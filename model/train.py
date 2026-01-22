from datetime import datetime
import os
import sys
import warnings
import time
os.environ['OMP_NUM_THREADS'] = '6'
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import argparse
import random
import numpy as np
import torch_scatter
from sklearn.cluster import KMeans
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from model.EnhancedHGNN import EnhancedHGNN
from model.Loss import Loss
from model.Evaluator import SimpleHypergraphEvaluator
from model.data_loader import DatasetLoader
from model.Augmentor import Augmentor
from model.ClusteringLayer import ClusteringPrototypes
from model.hgnn import HGNNEncoder


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_bytes(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def create_negative_samples(hyperedge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    node_idx, edge_idx = hyperedge_index
    num_edges = edge_idx.max().item() + 1
    ones = torch.ones_like(edge_idx, dtype=torch.long)
    edge_sizes = torch_scatter.scatter_add(ones, edge_idx, dim=0, dim_size=num_edges)
    num_to_replace_per_edge = torch.log2(edge_sizes.float()).floor().long()
    is_size_greater_than_one = (edge_sizes > 1)
    num_to_replace_per_edge = torch.max(num_to_replace_per_edge, is_size_greater_than_one.long())
    indices_to_replace_list = []
    for i in range(num_edges):
        num_replace = num_to_replace_per_edge[i].item()
        if num_replace > 0:
            current_edge_indices = torch.where(edge_idx == i)[0]
            sampled_indices = random.sample(current_edge_indices.tolist(), num_replace)
            indices_to_replace_list.extend(sampled_indices)
    indices_to_replace = torch.tensor(indices_to_replace_list, device=node_idx.device, dtype=torch.long)
    num_total_replacements = len(indices_to_replace)
    replacement_nodes = torch.randint(0, num_nodes, (num_total_replacements,), device=node_idx.device)
    hyperedge_index_aug = hyperedge_index.clone()
    hyperedge_index_aug[0, indices_to_replace] = replacement_nodes
    return hyperedge_index_aug

def train_step(
encoder: nn.Module,
project_head: nn.Module,
cluster_module: nn.Module,
optimizer: optim.Optimizer,
loss_fn: Loss,
features: torch.Tensor,
pos_hyperedge_index: torch.Tensor,
neg_hyperedge_index: torch.Tensor,
features_aug1: torch.Tensor,
hyperedge_index_aug1: torch.Tensor,
features_aug2: torch.Tensor,
hyperedge_index_aug2: torch.Tensor,
alpha: float,
beta: float,
gamma: float,
tau: float
) -> tuple[float, dict]:
    encoder.train()
    project_head.train()
    cluster_module.train()
    loss_fn.train()
    optimizer.zero_grad()
    pos_node_emb, _ = encoder(features, pos_hyperedge_index) 
    aug_emb_1, _ = encoder(features_aug1, hyperedge_index_aug1)
    aug_emb_2, _ = encoder(features_aug2, hyperedge_index_aug2)
    proj_emb_1 = project_head(aug_emb_1)
    proj_emb_2 = project_head(aug_emb_2)
    clustering_logits = cluster_module(pos_node_emb)
    loss, metrics = loss_fn(
        pos_node_embeds=pos_node_emb, 
        aug_node_embeds=proj_emb_1,
        other_aug_node_embeds=proj_emb_2,
        pos_hyperedge_index=pos_hyperedge_index, 
        neg_hyperedge_index=neg_hyperedge_index,
        clustering_logits=clustering_logits, 
        alpha=alpha, 
        beta=beta,
        gamma=gamma,
        tau=tau
    )
    loss.backward()
    optimizer.step()
    metrics['total_loss'] = loss.item()
    return metrics


def evaluate(
encoder: nn.Module,
cluster_module: nn.Module,
features: torch.Tensor,
hyperedge_index: torch.Tensor,
ground_truth: torch.Tensor,
evaluator: SimpleHypergraphEvaluator
) -> dict:
    encoder.eval()
    cluster_module.eval()
    with torch.no_grad():
        node_emb, _ = encoder(features, hyperedge_index)
        logits = cluster_module(node_emb)
        predicted_labels = torch.argmax(logits, dim=1)
    return evaluator.evaluate(predicted_labels, ground_truth)

def evaluate_with_kmeans(
    encoder: nn.Module,
    features: torch.Tensor,
    hyperedge_index: torch.Tensor,
    ground_truth: torch.Tensor,
    n_clusters: int,
    evaluator: SimpleHypergraphEvaluator,
    seed: int = 42
) -> dict:

    encoder.eval()
    with torch.no_grad():
        node_emb, _ = encoder(features, hyperedge_index)

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    kmeans.fit(node_emb.cpu().numpy())
    predicted_labels = torch.tensor(kmeans.labels_, device=ground_truth.device)
    return evaluator.evaluate(predicted_labels, ground_truth)

def main(config_override=None):
    parser = argparse.ArgumentParser(description='超图学习训练器')
    parser.add_argument('--dataset', type=str, default='cora_coauthor', choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 
                                                                       'dblp_coauthor', 'zoo', '20newsW100', 'Mushroom', 'NTU2012', 
                                                                       'ModelNet40','yelpRestaurant'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_config', action='store_true', help='保存当前配置')
    parser.add_argument('--load_config', action='store_true', help='加载已保存配置')
    parser.add_argument('--list_configs', action='store_true', help='列出已保存的配置')
    parser.add_argument('--load_model', action='store_true', help='直接加载已有模型而不进行训练')
    parser.add_argument('--seeds', type=str, default='0', help='随机种子,用逗号分隔多个种子')
    parser.add_argument('--runs', type=int, default=1, help='实验运行次数(当--seeds未指定多个种子时使用)')
    parser.add_argument('--encoder', type=str, default='enhanced', choices=['enhanced', 'hgnn'],help='选择编码器')
    parser.add_argument('--pretrain_only', action='store_true', help='只进行预训练，然后直接用K-means评估，用于消融实验。')
    args = parser.parse_args()
    
    if ',' in args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        base_seed = int(args.seeds)
        seeds = [base_seed + i for i in range(args.runs)]
    
    config = {
        'dataset_name': args.dataset ,                       
        'exp_name': args.dataset,
        'emb_dim': 512,               
        'num_layers': 1,                
        'lr': 5.0e-4,
        'finetune_lr_scale': 1,
        'weight_decay': 5.0e-04,
        'pretrain_epochs':180,
        'finetune_epochs':120,
        'epoch_eval' : 120,
        'seed': 42,
        'alpha': 1,
        'beta': 1,
        'tau': 0.5,
        'gamma': 1,
        'feat_mask_rate': 0.4,
        'edge_drop_rate': 0.4
    }

    if config_override:
        config.update(config_override)
        args.dataset = config['dataset_name']
    
    all_results = {
        'cluster_module': {'nmi': [], 'ari': [], 'acc': [], 'f1': []},
        'execution_times': [] 
    }
    
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*30} 运行 {run_idx+1}/{len(seeds)} (种子: {seed}) {'='*30}")
        config['seed'] = seed
        run_results = run_experiment(args, config)
        
        for method in ['cluster_module']:
            for metric in all_results[method].keys():
                all_results[method][metric].append(run_results[method][metric])
        if 'execution_time' in run_results:
            all_results['execution_times'].append(run_results['execution_time'])


    execution_times = all_results.pop('execution_times', [])


    mean_results = {
        method: {k: np.mean(v) for k, v in metrics.items()} 
        for method, metrics in all_results.items()
    }
    std_results = {
        method: {k: np.std(v) for k, v in metrics.items()} 
        for method, metrics in all_results.items()
    }
    
    all_results['execution_times'] = execution_times
    print("\n" + "="*50 + " 多次运行实验结果统计 " + "="*50)
    print(f"总运行次数: {len(seeds)}, 种子列表: {seeds}")
    print("\n基于簇分配矩阵的评估结果 (均值 ± 标准差):")
    for metric in ['nmi', 'ari', 'acc', 'f1']:
        print(f"  {metric.upper()}: {mean_results['cluster_module'][metric]:.4f} ± {std_results['cluster_module'][metric]:.4f}")
    
    if 'execution_times' in all_results and all_results['execution_times']:
        avg_time = np.mean(all_results['execution_times'])
        total_time = sum(all_results['execution_times'])
        avg_hours, avg_remainder = divmod(avg_time, 3600)
        avg_minutes, avg_seconds = divmod(avg_remainder, 60)
        total_hours, total_remainder = divmod(total_time, 3600)
        total_minutes, total_seconds = divmod(total_remainder, 60)
        print("\n时间统计:")
        print(f"  平均运行时间: {int(avg_hours):02d}小时{int(avg_minutes):02d}分钟{avg_seconds:.2f}秒")
        print(f"  总运行时间: {int(total_hours):02d}小时{int(total_minutes):02d}分钟{total_seconds:.2f}秒")
    
    return mean_results, std_results, all_results

def run_experiment(args, config):
    start_time=time.time()

    print("\n当前使用的配置:")
    for k, v in config.items(): print(f"  {k}: {v}")

    if args.pretrain_only:
        print("\n" + "="*20 + " 消融实验模式: 只进行预训练 " + "="*20)   

    fix_seed(config["seed"])
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)  
        torch.backends.cudnn.benchmark = True  
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats(device)
    else:
        print("没有可用的GPU, 将使用CPU")

    data = DatasetLoader().load(args.dataset).to(device)
    features, hyperedge_index, labels = data.features, data.hyperedge_index, data.labels
    node_dim = features.shape[1]
    num_nodes=features.shape[0]
    num_hyperedges = torch.unique(hyperedge_index[1]).shape[0]
    n_clusters = len(torch.unique(labels)) 
    print(f"数据集: {args.dataset}, 节点数: {features.shape[0]},超边数: {num_hyperedges}, 特征维度: {node_dim}")

    augmentor=Augmentor(config['feat_mask_rate'],config['edge_drop_rate'])
    if args.encoder == 'enhanced':
        encoder = EnhancedHGNN(node_dim, config['emb_dim'], config['num_layers']).to(device)
    elif args.encoder == 'hgnn':
        encoder = HGNNEncoder(node_dim, config['emb_dim'], config['num_layers']).to(device)
    cluster_module=ClusteringPrototypes(n_clusters,config['emb_dim'],config['tau']).to(device)
    project_head = nn.Sequential(nn.Linear(config['emb_dim'], config['emb_dim']),nn.ReLU(),nn.Linear(config['emb_dim'], config['emb_dim'])).to(device)
    loss_fn = Loss(config['emb_dim'],config['emb_dim']).to(device)
    if args.pretrain_only:
        all_params = list(encoder.parameters()) + list(project_head.parameters()) + list(loss_fn.parameters())
    else:
        all_params = list(encoder.parameters()) + list(project_head.parameters()) + list(loss_fn.parameters()) + list(cluster_module.parameters())
    optimizer = optim.Adam(all_params, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    evaluator = SimpleHypergraphEvaluator()
    total_epochs = config['pretrain_epochs'] + config['finetune_epochs']

    print("\n" + "="*50 + " STAGE 1: PRE-TRAINING " + "="*50)
    with tqdm(range(config['pretrain_epochs']), desc="Pre-training") as pbar:
        for epoch in pbar:
            current_gamma = 0.0 
            neg_hyperedge_index = create_negative_samples(hyperedge_index, num_nodes)
            features_aug1, he_index_aug1, features_aug2, he_index_aug2 = augmentor.augment(features, hyperedge_index)
            metrics = train_step(
                encoder, project_head, cluster_module, optimizer, loss_fn,
                features, hyperedge_index, neg_hyperedge_index,
                features_aug1, he_index_aug1, features_aug2, he_index_aug2,
                config['alpha'], config['beta'], current_gamma, config['tau']
            )
            pbar.set_postfix(total_loss=metrics['total_loss'])
    print("\nPre-training finished. The model has learned initial representations.")

    if args.pretrain_only:
        print("\n" + "="*50 + " PRE-TRAIN ONLY: FINAL EVALUATION " + "="*50)
        kmeans_results = evaluate_with_kmeans(encoder, features, hyperedge_index, labels, n_clusters, evaluator, config['seed'])

        print("\n基于预训练嵌入+K-means的评估结果:")
        print(f"  NMI: {kmeans_results['nmi']:.4f}")
        print(f"  ARI: {kmeans_results['ari']:.4f}")
        print(f"  ACC: {kmeans_results['acc']:.4f}")
        print(f"  F1:  {kmeans_results['f1']:.4f}")

        empty_results = {'nmi': 0, 'ari': 0, 'acc': 0, 'f1': 0}
        return {"cluster_module": empty_results, "kmeans": kmeans_results}

    encoder.eval() 
    with torch.no_grad():
        initial_node_emb, _ = encoder(features, hyperedge_index)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=config['seed'])
    kmeans.fit(initial_node_emb.cpu().numpy())
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=features.device)
    cluster_module.prototypes.data = cluster_centers


    print("\n" + "="*50 + " STAGE 2: FINE-TUNING " + "="*50)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= config['finetune_lr_scale']
    print(f"Learning rate adjusted for fine-tuning to: {optimizer.param_groups[0]['lr']}")
        
    with tqdm(range(config['finetune_epochs']), desc="Fine-tuning") as pbar:
        for epoch_offset in pbar:
            epoch = config['pretrain_epochs'] + epoch_offset 
            current_gamma = config['gamma']

            neg_hyperedge_index = create_negative_samples(hyperedge_index, num_nodes)
            features_aug1, he_index_aug1, features_aug2, he_index_aug2 = augmentor.augment(features, hyperedge_index)
            
            metrics = train_step(
                encoder, project_head, cluster_module, optimizer, loss_fn,
                features, hyperedge_index, neg_hyperedge_index,
                features_aug1, he_index_aug1, features_aug2, he_index_aug2,
                config['alpha'], config['beta'], current_gamma, config['tau']
            )
            
            scheduler.step(metrics['total_loss']) 
            pbar.set_postfix(total_loss=metrics['total_loss'])

            if (epoch_offset + 1) % config['epoch_eval'] == 0:
                print(f"\n--- Epoch {epoch+1}/{total_epochs}: Executing evaluation ---")
                eval_results = evaluate(
                    encoder, cluster_module, features, hyperedge_index, labels, evaluator
                )
                
                print(f"  Eval Results -> NMI: {eval_results['nmi']:.4f}, "
                    f"ARI: {eval_results['ari']:.4f}, "
                    f"ACC: {eval_results['acc']:.4f}, "
                    f"F1: {eval_results['f1']:.4f}")

    print("\n" + "="*50 + " FINAL EVALUATION " + "="*50)
    final_results = evaluate(encoder, cluster_module, features, hyperedge_index, labels, evaluator)

    print("\n基于簇分配矩阵的评估结果:")
    print(f"  NMI: {final_results['nmi']:.4f}")
    print(f"  ARI: {final_results['ari']:.4f}")
    print(f"  ACC: {final_results['acc']:.4f}")
    print(f"  F1:  {final_results['f1']:.4f}")

    if device.type == 'cuda':
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        print("\n" + "="*25 + " GPU Memory Analysis " + "="*25)
        print(f"  数据集 '{args.dataset}' 在此运行中的峰值GPU显存占用: {format_bytes(peak_memory_bytes)}")
        print("=" * 73)

    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n总运行时间: {int(hours):02d}小时{int(minutes):02d}分钟{seconds:.2f}秒")

    return {"cluster_module": final_results,"execution_time": execution_time}

if __name__ == '__main__':
    mean_results, std_results, all_results = main()
import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score
from scipy.optimize import linear_sum_assignment
from typing import Dict, Union

class SimpleHypergraphEvaluator:
    
    def __init__(self):
        pass

    def evaluate(self, 
                 predicted_labels: Union[torch.Tensor, np.ndarray],
                 ground_truth: Union[torch.Tensor, np.ndarray]) -> Dict:

        if isinstance(predicted_labels, torch.Tensor):y_pred = predicted_labels.cpu().numpy()
        else:y_pred = np.asarray(predicted_labels)
            
        if isinstance(ground_truth, torch.Tensor):y_true = ground_truth.cpu().numpy()
        else:y_true = np.asarray(ground_truth)

        acc = self._calculate_acc(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        f1 = self._calculate_f1(y_true, y_pred)
        
        return {'acc': acc, 'nmi': nmi, 'ari': ari, 'f1': f1}

    def _calculate_acc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return w[row_ind, col_ind].sum() / y_pred.size
    
    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        
        mapped_pred = np.zeros_like(y_pred)
        for i, cluster_id in enumerate(row_ind):
            mapped_pred[y_pred == cluster_id] = col_ind[i]
        
        return f1_score(y_true, mapped_pred, average='macro')
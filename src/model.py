import numpy as np
from typing import Tuple

def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 Sigmoid 函数"""
    res = np.zeros_like(z)
    mask_pos = (z >= 0)
    mask_neg = ~mask_pos
    res[mask_pos] = 1.0 / (1.0 + np.exp(-z[mask_pos]))
    exp_z = np.exp(z[mask_neg])
    res[mask_neg] = exp_z / (1.0 + exp_z)
    return res

def lr_loss(w: np.ndarray, A: np.ndarray, b: np.ndarray, lam: float) -> Tuple[float, np.ndarray]:
    """
    计算逻辑回归 Loss 和 Gradient。
    f(x) = (1/m) * sum(ln(1 + exp(-b * A * x))) + lambda * ||x||^2
    """
    m = A.shape[0]
    
    # 1. 向量化计算
    Aw = A @ w 
    margin = b * Aw
    
    # 2. 计算 Loss
    loss_term = np.sum(np.logaddexp(0, -margin))
    reg_term = lam * np.linalg.norm(w)**2
    f = (1.0 / m) * loss_term + reg_term
    
    # 3. 计算 Gradient
    sigmoid_val = _stable_sigmoid(-margin) 
    coeff = -b * sigmoid_val
    g = (1.0 / m) * (A.T @ coeff) + 2 * lam * w
    
    return f, g
import numpy as np
from typing import Tuple, Dict, Any, List
from .model import lr_loss

def gradient_descent_backtracking(
    w0: np.ndarray, A: np.ndarray, b: np.ndarray, lam: float, options: Dict[str, Any]
) -> Tuple[np.ndarray, List[float], List[float]]:
    """带有 Armijo 回溯线搜索的梯度下降"""
    
    max_iter = options.get('max_iter', 1000)
    tol = options.get('tol', 1e-6)
    alpha0 = options.get('alpha0', 1.0) 
    rho = options.get('rho', 0.5)       
    c = options.get('c', 1e-4)          
    
    w = w0.copy()
    f_hist, g_hist = [], []
    
    f_k, g_k = lr_loss(w, A, b, lam)
    
    print(f"{'Iter':<10} {'Loss':<15} {'Grad Norm':<15} {'Step Size':<15}")
    print("-" * 65)
    
    for k in range(max_iter):
        g_norm = np.linalg.norm(g_k)
        f_hist.append(f_k)
        g_hist.append(g_norm)
        
        if g_norm < tol:
            print(f"\n[Converged] Iteration: {k}, Gradient Norm: {g_norm:.2e}")
            break
        
        d = -g_k
        alpha = alpha0
        g_dot_d = - (g_norm ** 2)
        
        # === 回溯线搜索 ===
        while True:
            w_new = w + alpha * d
            f_new, _ = lr_loss(w_new, A, b, lam)
            
            # Armijo 条件
            if f_new <= f_k + c * alpha * g_dot_d:
                break
            
            # [证明]: 打印回溯动作
            if k == 0: 
                print(f"   [Backtracking] Alpha {alpha:.4f} too big (Loss: {f_new:.4f}), shrinking...")
            
            alpha *= rho
            if alpha < 1e-10: break
        
        w = w_new
        f_k, g_k = lr_loss(w, A, b, lam)
        
        if k == 0 or (k + 1) % 10 == 0:
            print(f"{k+1:<10} {f_k:<15.6f} {g_norm:<15.6f} {alpha:<15.6e}")
            
    return w, f_hist, g_hist
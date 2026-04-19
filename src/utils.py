import matplotlib.pyplot as plt
from typing import List

def plot_convergence(f_hist: List[float], g_hist: List[float], lambda_reg: float):
    """绘制 Loss 和梯度范数的收敛曲线"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(f_hist, linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.title(f'Loss Convergence (lambda={lambda_reg:.2e})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(g_hist, linewidth=2, color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
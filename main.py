import argparse
import numpy as np
import time

# 从拆分的模块中导入必要功能
from src.data_loader import load_a9a_data
from src.optimizer import gradient_descent_backtracking
from src.utils import plot_convergence

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Logistic Regression with Backtracking Line Search")
    parser.add_argument('--data_path', type=str, default='data/a9a.txt', help='Path to the dataset')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for gradient norm')
    parser.add_argument('--alpha0', type=float, default=100.0, help='Initial step size')
    parser.add_argument('--rho', type=float, default=0.5, help='Decay factor for backtracking')
    parser.add_argument('--c', type=float, default=1e-4, help='Armijo parameter')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载数据
    print(f"Loading dataset from '{args.data_path}'...")
    A, b = load_a9a_data(args.data_path)
    m, n = A.shape
    print(f"Data Loaded: m={m}, n={n} (including bias)")

    # 2. 参数设置 (根据 PDF)
    lambda_reg = 1.0 / (100.0 * m)
    print(f"Regularization parameter lambda = 1/(100*m) = {lambda_reg:.8f}")

    w0 = np.zeros((n, 1))
    
    solver_options = {
        'max_iter': args.max_iter,
        'tol': args.tol,
        'alpha0': args.alpha0,
        'rho': args.rho,
        'c': args.c
    }
    
    # 3. 运行优化
    print("\nStarting Optimization...")
    start_time = time.time()
    w_opt, f_hist, g_hist = gradient_descent_backtracking(w0, A, b, lambda_reg, solver_options)
    print(f"\nTime elapsed: {time.time() - start_time:.4f}s")
    
    # 4. 绘图
    plot_convergence(f_hist, g_hist, lambda_reg)

if __name__ == "__main__":
    main()
import numpy as np
from sklearn.datasets import load_svmlight_file
import os

def load_a9a_data(data_path: str):
    """加载并预处理 a9a 数据集"""

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_data_path = os.path.join(project_root, data_path)

    if not os.path.exists(absolute_data_path):
        print(f"Warning: 文件 '{absolute_data_path}' 未找到。使用模拟数据演示代码逻辑...")
        m, n = 32561, 123
        A = np.random.randn(m, n)
        b = np.sign(np.random.randn(m, 1))
        b[b == 0] = -1
    else:
        # 使用 sklearn 加载 libsvm 格式
        X_sparse, y = load_svmlight_file(absolute_data_path)
        A = X_sparse.toarray()
        b = y.reshape(-1, 1)
        
        # 处理标签: 确保是 +1/-1
        b[b == 0] = -1
        b[b == 2] = -1 # 有些版本的a9a包含2，视为-1
        
    # 增加偏置项 (Bias Trick): 在 A 末尾加一列 1
    m, n_features = A.shape
    A = np.hstack([A, np.ones((m, 1))])
    
    return A, b
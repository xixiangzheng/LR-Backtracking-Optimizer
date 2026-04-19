# 📈 Logistic Regression with Backtracking Line Search

This is a from-scratch implementation of a Logistic Regression optimization solver. This project does not use any high-level machine learning libraries (such as the underlying models of `scikit-learn`). Instead, it implements a **Gradient Descent algorithm with Armijo Backtracking Line Search** entirely from scratch using only `NumPy`.

Originally developed as a lab project for Convex Optimization and Machine Learning Basics, this project has been refactored into a modular, standard engineering structure to demonstrate low-level algorithmic capabilities and good software engineering practices.

## ✨ Core Highlights

* **Pure from-scratch implementation**: Independently derived and implemented the Loss function and analytical Gradient for Logistic Regression.
* **Intelligent step-size adaptation**: Implemented the Armijo backtracking line search criterion to automatically find the optimal learning rate, avoiding the oscillation or slow convergence problems caused by traditional fixed step sizes.
* **Industrial-grade engineering practices**:
  * **Numerical Stability**: The underlying implementation perfectly avoids overflow issues in exponential operations by piecewise processing the `sigmoid` function and introducing `np.logaddexp`.
  * **Vectorization**: Discarded inefficient `for` loops in favor of full matrix operations, achieving extremely fast convergence on the a9a dataset.
  * **Modular design**: Data loading, mathematical modeling, the optimizer, and the scheduling entry point are completely decoupled.

## 🧮 Mathematical Model

For a given binary classification dataset, where the feature vector is $a_i \in \mathbb{R}^n$ and the label is $b_i \in \{-1, +1\}$. The optimized objective function with L2 regularization is:

$$
\min_{w \in \mathbb{R}^n} f(w) = \frac{1}{m} \sum_{i=1}^{m} \ln(1 + \exp(-b_i a_i^T w)) + \lambda ||w||^2
$$

Its corresponding analytical gradient formula is:

$$
\nabla f(w) = \frac{1}{m} \sum_{i=1}^{m} -b_i a_i \sigma(-b_i a_i^T w) + 2\lambda w
$$

Where $\sigma(z) = \frac{1}{1 + \exp(-z)}$ is the Sigmoid function, and $\lambda$ is the regularization coefficient.

## 📂 Project Structure

```text
lr-backtracking-optimizer/
├── data/
│   └── a9a.txt                # Public dataset in LibSVM format (requires manual download)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and Bias Trick preprocessing
│   ├── model.py               # Loss function and numerically stable gradient computation
│   ├── optimizer.py           # Gradient descent and Armijo backtracking line search logic
│   └── utils.py               # Result visualization tools
├── main.py                    # Command-line entry point
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## 🚀 Quick Start

### 1. Environment Dependencies

Please ensure Python 3.8+ is installed on your machine. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Main dependencies are `numpy`, `matplotlib`, and the single-file data loading module from `scikit-learn`)*

### 2. Prepare the Dataset

This project uses the classic `a9a` (Adult) dataset by default.
Please go to the [LIBSVM Data official website](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) to download the `a9a` dataset, place it in the `data/` directory, and name it `a9a.txt`.

### 3. Run the Optimizer

You can directly run the entry file, which will start the optimization process using the default parameters:

```bash
python main.py
```

If you want to customize the hyperparameters of the optimizer, you can pass arguments via the command line:

```bash
python main.py --max_iter 500 --alpha0 100.0 --rho 0.5 --tol 1e-6
```

**Supported Parameters:**
| Parameter | Description | Default Value |
| :--- | :--- | :--- |
| `--data_path` | Dataset file path | `data/a9a.txt` |
| `--max_iter` | Maximum number of iterations | 200 |
| `--tol` | Convergence tolerance for gradient norm | 1e-6 |
| `--alpha0` | Initial trial step size | 100.0 |
| `--rho` | Step size decay ratio (0, 1) | 0.5 |
| `--c` | Armijo sufficient decrease parameter | 1e-4 |

## 📊 Experimental Results Analysis

After the program finishes running, the console will output the complete optimization trajectory (including the current iteration number, Loss value, gradient norm, and the currently selected step size Alpha), and automatically pop up the convergence trend chart.

  * **Effectiveness of backtracking line search**: When the initial trial step size is set too large (e.g., 100.0), the optimizer can automatically reject updates that cause the Loss to diverge via the Armijo criterion, and decay the step size to a reasonable range (e.g., 3.125), followed by smooth convergence.
  * **Convergence**: As the iterations proceed, the gradient norm decreases logarithmically, proving that the model has effectively converged to the global optimum.

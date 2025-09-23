# Optimization 1 Project – Support Vector Machines

This repository accompanies an Optimization 1 course project on soft-margin Support Vector Machines (SVMs). It contains both the LaTeX sources for the written report and reference Python implementations of primal and dual SVM solvers used to generate the numerical experiments.

## Repository layout
- `main.tex` and `chapters/`, `frontmatter/`, `appendices/`, `figures/`: LaTeX sources for the report. Compile `main.tex` to build the full document.
- `bib/`: Bibliography style file, references, and glossary definitions used by the report.
- `svm/`: Lightweight Python package with reusable components for the experiments (`PrimalSVM`, `DualSVM`, kernels, utilities, and the `demo.ipynb` notebook).
- `appendices/project_description.pdf`: Original project handout for reference.

## Python environment
The SVM code depends on a minimal scientific Python stack:

- Python 3.10+
- `numpy`
- `matplotlib` (only for plotting helpers in `utils.py`)
- `jupyter` (optional, to run the notebook)

Create an isolated environment and install the dependencies, for example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib jupyter
```

## Using the SVM implementations
The `svm` package contains gradient-based solvers for both the primal and dual formulations. A quick linear classification example:

```python
import numpy as np
from svm.svm_primal import PrimalSVM
from svm.svm_dual import DualSVM

X = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 3.0]])
y = np.array([1, 1, -1, -1])

primal = PrimalSVM(C=1.0, lr=1e-3, max_iter=5000).fit(X, y)
print("Primal accuracy:", primal.score(X, y))

dual = DualSVM(C=1.0, max_iter=2000).fit(X, y)
print("Dual accuracy:", dual.score(X, y))
```

Key utilities:
- `svm/kernels.py`: Linear, polynomial, RBF, and Laplacian kernels plus helpers to build parameterized kernels.
- `svm/utils.py`: Projection operator used by the dual solver, hinge loss helpers, synthetic data generation, and plotting utilities.
- `svm/demo.ipynb`: Jupyter notebook that demonstrates training, convergence behaviour, and reproduces plots used in the report.

## Reproducing figures and experiments
Most figures under `figures/` were produced with the notebook and helper functions in `svm/utils.py`. After activating your environment:

1. Launch Jupyter with `jupyter notebook svm/demo.ipynb`.
2. Run the notebook cells to regenerate training runs, convergence diagnostics, and comparison plots.
3. Update the LaTeX sources to include regenerated plots if you make changes (`figures/` is tracked so regenerated images can be dropped in place).

## Troubleshooting
- **LaTeX/biblatex errors**: Ensure `biber` is installed and available on your PATH. Running `latexmk -pdf main.tex` from the project root automatically invokes it.
- **Floating point or convergence issues in SVM solvers**: Reduce the learning rate (`lr`) for the primal solver or disable Barzilai-Borwein steps (`bb_steps=False`) if oscillations appear. For the dual solver, tighten the stopping tolerance (`tol`) or disable line search to test alternative step rules.

## License
No explicit license file is included. Treat the material as academic coursework; contact the authors before reusing the content outside the course context.

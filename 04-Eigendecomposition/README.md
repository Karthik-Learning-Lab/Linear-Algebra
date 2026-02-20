# Topic 04 — Eigendecomposition

---

## 0. PREREQUISITES (Precision Check)

**Required prerequisites:**
- Vectors and vector operations (Topic 01)
- Matrices and matrix operations, including inverse (Topic 02)
- Eigenvalues and eigenvectors (Topic 03)

**How this builds forward:**
Eigendecomposition is the direct application of eigenvalues/eigenvectors to factorise a matrix. It is the conceptual precursor to SVD (Topic 05) and directly enables PCA, spectral methods, and efficient computation of matrix powers.

---

## 1. INTUITION FIRST (Clarity Before Formalism)

### What is eigendecomposition?

If a matrix $\mathbf{A}$ has $n$ linearly independent eigenvectors, you can completely **break it apart** into three simple pieces:

$$\mathbf{A} = \mathbf{P} \boldsymbol{\Lambda} \mathbf{P}^{-1}$$

where:
- $\mathbf{P}$: matrix of eigenvectors (columns)
- $\boldsymbol{\Lambda}$: diagonal matrix of eigenvalues
- $\mathbf{P}^{-1}$: inverse of eigenvector matrix

**Real-world analogy:**

Imagine you have a complex, hard-to-use tool. You disassemble it into simple parts (screws, levers, springs), work with each part independently, then reassemble. Eigendecomposition does the same for a matrix — it finds the "natural axes" of the transformation, applies simple scaling along each axis, then returns to the original frame.

### Why is this useful?

- **Matrix powers**: $\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1}$ — trivially computed by raising diagonal entries to power $k$.
- **Matrix exponential**: $e^{\mathbf{A}} = \mathbf{P} e^{\boldsymbol{\Lambda}} \mathbf{P}^{-1}$ — used in differential equations and graph theory.
- **Understanding structure**: Reveals the "natural" directions and scales of a transformation.
- **Computational savings**: diagonal $\boldsymbol{\Lambda}$ is trivial to manipulate.

---

## 2. CORE THEORY (Rigorous but Clean)

### 2.1 Formal Derivation

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ with $n$ linearly independent eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$ and corresponding eigenvalues $\lambda_1, \ldots, \lambda_n$.

Define:
$$\mathbf{P} = \begin{bmatrix} \mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \end{bmatrix} \in \mathbb{R}^{n\times n}$$

$$\boldsymbol{\Lambda} = \begin{bmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_n \end{bmatrix} \in \mathbb{R}^{n\times n}$$

From $\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$ for all $i$:

$$\mathbf{A}\mathbf{P} = \begin{bmatrix} \lambda_1\mathbf{v}_1 & \cdots & \lambda_n\mathbf{v}_n \end{bmatrix} = \mathbf{P}\boldsymbol{\Lambda}$$

Since eigenvectors are linearly independent, $\mathbf{P}$ is invertible. Multiply right by $\mathbf{P}^{-1}$:

$$\boxed{\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}}$$

### 2.2 Diagonalisability Condition

A matrix $\mathbf{A} \in \mathbb{R}^{n\times n}$ is **diagonalisable** if and only if it has $n$ linearly independent eigenvectors.

**Sufficient conditions for diagonalisability:**
- $\mathbf{A}$ has $n$ **distinct** eigenvalues.
- $\mathbf{A}$ is **real symmetric** (spectral theorem — always diagonalisable with orthonormal eigenvectors).
- For every repeated eigenvalue, geometric multiplicity = algebraic multiplicity.

**Non-diagonalisable (defective) matrices:** When for some $\lambda_i$, geometric multiplicity < algebraic multiplicity. Example: the Jordan block $\mathbf{J}$ below has $\lambda = 1$ with multiplicity 2 but only one independent eigenvector:

$$\mathbf{J} = \begin{bmatrix}1 & 1\\0 & 1\end{bmatrix}$$

### 2.3 Orthogonal Diagonalisation (Symmetric Matrices)

For a real **symmetric** matrix $\mathbf{A} = \mathbf{A}^\top$, the **Spectral Theorem** guarantees:

$$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$$

where $\mathbf{Q}$ is **orthogonal** ($\mathbf{Q}^\top\mathbf{Q} = \mathbf{Q}\mathbf{Q}^\top = \mathbf{I}$) and $\boldsymbol{\Lambda}$ contains real eigenvalues.

This is a stronger and more stable result than general eigendecomposition:
- No need to invert $\mathbf{Q}$ — just transpose it.
- Eigenvectors are mutually orthogonal and can be made orthonormal.

### 2.4 Eigendecomposition as Sum of Rank-1 Matrices

An equivalent outer-product form:

$$\mathbf{A} = \sum_{i=1}^n \lambda_i \mathbf{v}_i \mathbf{v}_i^\top \quad \text{(for symmetric } \mathbf{A} \text{ with orthonormal eigenvectors)}$$

Each term $\lambda_i \mathbf{v}_i \mathbf{v}_i^\top$ is a rank-1 matrix — a projection onto the direction $\mathbf{v}_i$ scaled by $\lambda_i$.

This is the **spectral decomposition** — the matrix is built by superimposing rank-1 projections.

**Low-rank approximation:** Keep only the top $k$ terms:

$$\mathbf{A}_k = \sum_{i=1}^k \lambda_i \mathbf{v}_i \mathbf{v}_i^\top$$

This minimises the Frobenius norm error $\|\mathbf{A} - \mathbf{A}_k\|_F$ among all rank-$k$ approximations (when eigenvalues are sorted by $|\lambda_i|$).

### 2.5 Matrix Powers via Eigendecomposition

$$\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1}$$

where $\boldsymbol{\Lambda}^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$.

This makes computing $\mathbf{A}^{100}$ trivial — compute $\lambda_i^{100}$ and reassemble.

**Large-$k$ behaviour:**
- If $|\lambda_1| > |\lambda_i|$ for all $i > 1$ (dominant eigenvalue): $\mathbf{A}^k \approx \lambda_1^k \mathbf{v}_1 \mathbf{v}_1^\top / (\mathbf{v}_1^\top \mathbf{b})$ — converges to rank-1.
- If $|\lambda_i| < 1$ for all $i$: $\mathbf{A}^k \to \mathbf{0}$ (exponential decay — stable system).
- If $|\lambda_i| > 1$ for some $i$: $\mathbf{A}^k \to \infty$ in those directions (unstable).

### 2.6 Determinant and Trace via Eigendecomposition

$$\det(\mathbf{A}) = \det(\mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}) = \det(\mathbf{P})\det(\boldsymbol{\Lambda})\det(\mathbf{P}^{-1}) = \det(\boldsymbol{\Lambda}) = \prod_{i=1}^n \lambda_i$$

$$\text{tr}(\mathbf{A}) = \text{tr}(\mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}) = \text{tr}(\boldsymbol{\Lambda}\mathbf{P}^{-1}\mathbf{P}) = \text{tr}(\boldsymbol{\Lambda}) = \sum_{i=1}^n \lambda_i$$

These are immediate — eigendecomposition makes it transparent.

### 2.7 Connection to PCA

The covariance matrix $\boldsymbol{\Sigma}$ of centred data $\mathbf{X}$ is symmetric positive semi-definite:

$$\boldsymbol{\Sigma} = \frac{1}{N}\mathbf{X}^\top\mathbf{X} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$$

The columns of $\mathbf{Q}$ are **principal components** (directions of maximum variance). PCA is eigendecomposition of the covariance matrix.

---

## 3. WORKED EXAMPLES (Exactly Two)

### Example 1: Full Eigendecomposition and Matrix Power

$$\mathbf{A} = \begin{bmatrix}4 & 1\\2 & 3\end{bmatrix}$$

From Topic 03: $\lambda_1 = 5$, $\mathbf{v}_1 = (1,1)^\top$; $\lambda_2 = 2$, $\mathbf{v}_2 = (1,-2)^\top$.

**Step 1: Form $\mathbf{P}$ and $\boldsymbol{\Lambda}$:**

$$\mathbf{P} = \begin{bmatrix}1 & 1\\1 & -2\end{bmatrix}, \qquad \boldsymbol{\Lambda} = \begin{bmatrix}5 & 0\\0 & 2\end{bmatrix}$$

**Step 2: Compute $\mathbf{P}^{-1}$:**

$$\det(\mathbf{P}) = (1)(-2) - (1)(1) = -3$$

$$\mathbf{P}^{-1} = \frac{1}{-3}\begin{bmatrix}-2 & -1\\-1 & 1\end{bmatrix} = \begin{bmatrix}2/3 & 1/3\\1/3 & -1/3\end{bmatrix}$$

**Step 3: Verify $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$:**

$$\mathbf{P}\boldsymbol{\Lambda} = \begin{bmatrix}1\cdot5 & 1\cdot2\\1\cdot5 & -2\cdot2\end{bmatrix} = \begin{bmatrix}5 & 2\\5 & -4\end{bmatrix}$$

$$\mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1} = \begin{bmatrix}5 & 2\\5 & -4\end{bmatrix}\begin{bmatrix}2/3 & 1/3\\1/3 & -1/3\end{bmatrix} = \begin{bmatrix}10/3+2/3 & 5/3-2/3\\10/3-4/3 & 5/3+4/3\end{bmatrix} = \begin{bmatrix}4 & 1\\2 & 3\end{bmatrix} = \mathbf{A} \checkmark$$

**Step 4: Compute $\mathbf{A}^3$ efficiently:**

$$\boldsymbol{\Lambda}^3 = \begin{bmatrix}5^3 & 0\\0 & 2^3\end{bmatrix} = \begin{bmatrix}125 & 0\\0 & 8\end{bmatrix}$$

$$\mathbf{A}^3 = \mathbf{P}\boldsymbol{\Lambda}^3\mathbf{P}^{-1} = \begin{bmatrix}1&1\\1&-2\end{bmatrix}\begin{bmatrix}125 & 0\\0 & 8\end{bmatrix}\begin{bmatrix}2/3&1/3\\1/3&-1/3\end{bmatrix}$$

$$= \begin{bmatrix}125&8\\125&-16\end{bmatrix}\begin{bmatrix}2/3&1/3\\1/3&-1/3\end{bmatrix} = \begin{bmatrix}250/3+8/3 & 125/3-8/3\\250/3-16/3 & 125/3+16/3\end{bmatrix} = \begin{bmatrix}86 & 39\\78 & 47\end{bmatrix}$$

---

### Example 2: Spectral Decomposition of Symmetric Matrix (Exam-level)

$$\mathbf{S} = \begin{bmatrix}3 & 1\\1 & 3\end{bmatrix}$$

**Step 1: Eigenvalues:**

$$\det(\mathbf{S} - \lambda\mathbf{I}) = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda-4)(\lambda-2) = 0$$

$$\lambda_1 = 4, \quad \lambda_2 = 2$$

**Step 2: Eigenvectors:**

For $\lambda_1 = 4$:

$$(\mathbf{S}-4\mathbf{I})\mathbf{v} = \begin{bmatrix}-1&1\\1&-1\end{bmatrix}\mathbf{v} = \mathbf{0} \implies \mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$$

For $\lambda_2 = 2$:

$$(\mathbf{S}-2\mathbf{I})\mathbf{v} = \begin{bmatrix}1&1\\1&1\end{bmatrix}\mathbf{v} = \mathbf{0} \implies \mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$$

**Verify orthogonality:** $\mathbf{v}_1 \cdot \mathbf{v}_2 = \frac{1}{2}(1 - 1) = 0 \checkmark$

**Step 3: Orthogonal decomposition $\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$:**

$$\mathbf{Q} = \frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}, \quad \boldsymbol{\Lambda} = \begin{bmatrix}4&0\\0&2\end{bmatrix}$$

**Step 4: Spectral outer-product form:**

$$\mathbf{S} = 4\mathbf{v}_1\mathbf{v}_1^\top + 2\mathbf{v}_2\mathbf{v}_2^\top = 4\cdot\frac{1}{2}\begin{bmatrix}1&1\\1&1\end{bmatrix} + 2\cdot\frac{1}{2}\begin{bmatrix}1&-1\\-1&1\end{bmatrix}$$

$$= \begin{bmatrix}2&2\\2&2\end{bmatrix} + \begin{bmatrix}1&-1\\-1&1\end{bmatrix} = \begin{bmatrix}3&1\\1&3\end{bmatrix} = \mathbf{S} \checkmark$$

---

## 4. IMPLEMENTATION (Notebook Style Only)

### Importing libraries

```python
import numpy as np
import matplotlib.pyplot as plt
```

### Eigendecomposition and verification

```python
A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigenvalues, P = np.linalg.eig(A)
Lambda = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

A_reconstructed = P @ Lambda @ P_inv
print("A reconstructed:\n", np.round(A_reconstructed, 6))
print("Matches original:", np.allclose(A, A_reconstructed))
```

### Computing matrix powers via eigendecomposition

```python
def matrix_power_eig(A, k):
    """Compute A^k using eigendecomposition."""
    eigenvalues, P = np.linalg.eig(A)
    Lambda_k = np.diag(eigenvalues ** k)
    return P @ Lambda_k @ np.linalg.inv(P)

A_cubed_eig = matrix_power_eig(A, 3)
A_cubed_direct = np.linalg.matrix_power(A, 3)

print("A^3 via eigendecomposition:\n", np.round(A_cubed_eig.real, 6))
print("A^3 directly:\n", A_cubed_direct)
print("Match:", np.allclose(A_cubed_eig.real, A_cubed_direct))
```

### Orthogonal diagonalisation for symmetric matrix

```python
S = np.array([[3, 1],
              [1, 3]], dtype=float)

# eigh guarantees real eigenvalues and orthonormal eigenvectors
eigenvalues_s, Q = np.linalg.eigh(S)
Lambda_s = np.diag(eigenvalues_s)

print("Eigenvalues:", eigenvalues_s)
print("Q (orthonormal eigenvectors):\n", Q)
print("Q^T Q (should be I):\n", np.round(Q.T @ Q, 6))
print("S = Q Λ Q^T:\n", np.round(Q @ Lambda_s @ Q.T, 6))
```

### Spectral decomposition as sum of rank-1 matrices

```python
S_approx = np.zeros_like(S)
for i in range(len(eigenvalues_s)):
    v = Q[:, i:i+1]                    # column vector
    rank1 = eigenvalues_s[i] * (v @ v.T)
    S_approx += rank1
    print(f"After adding term {i+1} (λ={eigenvalues_s[i]:.2f}):\n{np.round(S_approx, 4)}")

print("\nFull reconstruction matches S:", np.allclose(S_approx, S))
```

### Visualising eigendecomposition as change of basis

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
S = np.array([[3, 1], [1, 3]], dtype=float)
evals, Q = np.linalg.eigh(S)

# Create unit circle and apply transformations
theta = np.linspace(0, 2*np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

# Step 1: Original circle
axes[0].plot(circle[0], circle[1], 'b-')
axes[0].set_title("Input: unit circle")

# Step 2: Rotate by Q^T (change to eigenbasis)
rotated = Q.T @ circle
axes[1].plot(rotated[0], rotated[1], 'g-')
axes[1].set_title("After Q^T (eigenbasis)")

# Step 3: Scale by Λ, then rotate back by Q
final = S @ circle
axes[2].plot(final[0], final[1], 'r-')
axes[2].set_title(f"After S=QΛQ^T\nλ={evals.round(1)}")

for ax, v in zip(axes[1:], [Q.T @ circle, final]):
    lim = np.max(np.abs(v)) * 1.2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

for ax in axes:
    ax.set_aspect('equal'); ax.grid(True)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    lim = axes[1].get_xlim()

plt.tight_layout()
plt.savefig("outputs/output_1.png")   # saved to outputs/
```

### PCA via eigendecomposition of covariance matrix

```python
np.random.seed(42)

# Generate correlated 2D data
mean = [0, 0]
cov_true = [[3, 2], [2, 2]]
data = np.random.multivariate_normal(mean, cov_true, 200)
data -= data.mean(axis=0)   # centre

# Covariance matrix
Sigma = (data.T @ data) / (len(data) - 1)
print("Covariance matrix:\n", np.round(Sigma, 4))

# Eigendecomposition
evals, evecs = np.linalg.eigh(Sigma)
idx = np.argsort(evals)[::-1]   # sort descending
evals, evecs = evals[idx], evecs[:, idx]

print("Eigenvalues (principal variances):", np.round(evals, 4))
print("Explained variance ratio:", np.round(evals / evals.sum(), 4))

# Plot
plt.figure(figsize=(7, 7))
plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10)
for i, (lam, v) in enumerate(zip(evals, evecs.T)):
    plt.quiver(0, 0, v[0]*lam, v[1]*lam, color=['red', 'blue'][i],
               scale=1, scale_units='xy', angles='xy',
               label=f'PC{i+1} (λ={lam:.2f})', width=0.02)
plt.legend(); plt.grid(True)
plt.set_aspect = 'equal'
plt.title("PCA via Eigendecomposition of Covariance Matrix")
plt.tight_layout()
plt.savefig("outputs/output_2.png")   # saved to outputs/
```

```requirements.txt
# ── shared base ──────────────────────────────────────
numpy>=1.26,<2.0
matplotlib>=3.8,<4.0
scipy>=1.12,<2.0
scikit-learn>=1.4,<2.0
torch>=2.2,<3.0
torchvision>=0.17,<1.0
tqdm>=4.66,<5.0
Pillow>=10.2,<11.0
pandas>=2.2,<3.0
# ── add topic-specific extras below ───────────────────
```

---

## 5. CONNECTIONS & RESEARCH CONTEXT

### In AI and ML:

| Area                    | Role of eigendecomposition                                                              |
| ----------------------- | --------------------------------------------------------------------------------------- |
| **PCA**                 | Eigenvectors of covariance matrix = principal components                                |
| **Spectral clustering** | Eigenvectors of graph Laplacian define cluster assignments                              |
| **Kernel methods**      | Representer theorem and Nyström approximation use eigendecomposition of kernel matrices |
| **Matrix completion**   | Low-rank matrix approximation via top eigenvectors                                      |
| **LSTM/RNN stability**  | Eigenvalues of weight matrices determine gradient flow                                  |
| **Markov chains**       | Stationary distribution = eigenvector for $\lambda=1$                                   |

### Limitations:
- Only applicable to **square matrices** — rectangular matrices require SVD instead.
- Non-diagonalisable matrices require Jordan Normal Form (rarely used in practice).
- Computing all $n$ eigenvalues costs $O(n^3)$ — expensive for large matrices.
- **Numerical instability** when eigenvalues are close or the eigenvectors matrix is ill-conditioned.

### Research context:
- **Hessian eigendecomposition** in deep learning: the curvature landscape of the loss is characterised by Hessian eigenvalues. Sharp minima (large positive eigenvalues) generalise worse than flat minima — an active research area (Keskar et al., 2017; Foret et al., 2021 SAM optimiser).
- **Spectral graph theory** in GNNs: Chebyshev polynomial approximations replace full eigendecomposition for scalable spectral convolutions.

---

## 6. COMMON EXAM QUESTIONS (5 Questions + Answers)

### Q1 (Conceptual): When does eigendecomposition fail to exist? What is the remedy?

**Answer:** Eigendecomposition $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$ fails when $\mathbf{A}$ does not have $n$ linearly independent eigenvectors (i.e., $\mathbf{A}$ is **defective**). This happens when an eigenvalue's geometric multiplicity is less than its algebraic multiplicity. The remedy is **Jordan Normal Form** ($\mathbf{A} = \mathbf{P}\mathbf{J}\mathbf{P}^{-1}$ where $\mathbf{J}$ is block-diagonal with Jordan blocks). For non-square or numerical work, **SVD** is always applicable and more robust.

---

### Q2 (Conceptual): Why does $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ for symmetric $\mathbf{A}$ but $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$ in general?

**Answer:** For symmetric $\mathbf{A}$, the eigenvectors are orthonormal (by the Spectral Theorem), so $\mathbf{P} = \mathbf{Q}$ is orthogonal: $\mathbf{Q}^{-1} = \mathbf{Q}^\top$. This makes the decomposition simpler and more numerically stable — transposing vs. inverting. For non-symmetric matrices, eigenvectors are generally not orthogonal, so explicit inversion of $\mathbf{P}$ is required.

---

### Q3 (Mathematical): Compute $\mathbf{A}^{10}$ for the diagonal matrix $\mathbf{A}$ given below.

**Answer:**

$$\mathbf{A} = \begin{bmatrix}2 & 0\\ 0 & 3\end{bmatrix}$$

$\mathbf{A}$ is already diagonal, so eigenvalues are $\lambda_1 = 2$, $\lambda_2 = 3$:

$$\mathbf{A}^{10} = \begin{bmatrix}2^{10} & 0\\0 & 3^{10}\end{bmatrix} = \begin{bmatrix}1024 & 0\\0 & 59049\end{bmatrix}$$

---

### Q4 (Mathematical): Given spectral decomposition $\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$, derive expressions for $\mathbf{S}^{-1}$ and $\mathbf{S}^{1/2}$.

**Answer:**

**Inverse:** $\mathbf{S}^{-1} = (\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top)^{-1} = (\mathbf{Q}^\top)^{-1}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^{-1} = \mathbf{Q}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^\top$

where $\boldsymbol{\Lambda}^{-1} = \text{diag}(1/\lambda_1, \ldots, 1/\lambda_n)$ (requires all $\lambda_i > 0$).

**Square root:** $\mathbf{S}^{1/2} = \mathbf{Q}\boldsymbol{\Lambda}^{1/2}\mathbf{Q}^\top$ where $\boldsymbol{\Lambda}^{1/2} = \text{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n})$ (requires all $\lambda_i \geq 0$, i.e., PSD matrix).

**Verify:** $\mathbf{S}^{1/2}\mathbf{S}^{1/2} = \mathbf{Q}\boldsymbol{\Lambda}^{1/2}\underbrace{\mathbf{Q}^\top\mathbf{Q}}_{=\mathbf{I}}\boldsymbol{\Lambda}^{1/2}\mathbf{Q}^\top = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top = \mathbf{S} \checkmark$

---

### Q5 (Applied): In PCA, we project data $\mathbf{X}$ onto the top eigenvector of the covariance matrix $\boldsymbol{\Sigma}$. What fraction of total variance is captured?

**Answer:** The covariance matrix $\boldsymbol{\Sigma}$ has eigendecomposition $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ where eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n \geq 0$.

The **total variance** is $\text{tr}(\boldsymbol{\Sigma}) = \sum_{i=1}^n \lambda_i$ (sum of diagonal = sum of eigenvalues).

The variance captured by the top $k$ components is $\sum_{i=1}^k \lambda_i$.

**Fraction of variance explained** by top $k$ components:

$$\text{EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^n \lambda_i}$$

For a single top eigenvector: $\text{EVR}_1 = \lambda_1 / \sum_i \lambda_i$.

---

## 7. COMMON MISTAKES

**Mistake 1: Applying eigendecomposition to non-square or defective matrices**
- Eigendecomposition requires square matrices. For non-square, defective, or numerically ill-conditioned matrices, use SVD instead.

**Mistake 2: Confusing $\mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$ with $\mathbf{P}^\top\boldsymbol{\Lambda}\mathbf{P}$**
- The latter is only valid for orthogonal $\mathbf{P}$ (symmetric matrices). In the general case, you must explicitly invert $\mathbf{P}$.

**Mistake 3: Forgetting to sort eigenvalues when used for PCA**
- NumPy does not guarantee eigenvalue ordering. Always sort eigenvalues (and corresponding eigenvectors) by descending absolute value.

**Mistake 4: Expecting unique eigendecomposition**
- Eigenvectors are only unique up to sign (and scaling). Multiple valid decompositions exist; the eigenvalues are unique (up to ordering).

**Mistake 5: Using `np.linalg.eig` for symmetric matrices in production**
- `np.linalg.eigh` is specialised for symmetric/Hermitian matrices. It is faster, more numerically stable, and guarantees real eigenvalues.

---

## 8. SUMMARY FLASHCARD

- Eigendecomposition: $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$, where $\mathbf{P}$ = eigenvectors (columns), $\boldsymbol{\Lambda}$ = diagonal eigenvalues.
- Exists iff $\mathbf{A}$ has $n$ linearly independent eigenvectors (diagonalisable).
- Symmetric: $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ with orthonormal $\mathbf{Q}$ — always diagonalisable (Spectral Theorem).
- Matrix power: $\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1}$ — trivial on the diagonal.
- Spectral decomposition: $\mathbf{A} = \sum_i \lambda_i \mathbf{v}_i\mathbf{v}_i^\top$ — sum of rank-1 projections.
- PCA = eigendecomposition of covariance matrix $\boldsymbol{\Sigma}$; EVR = $\lambda_k / \sum \lambda_i$.
- Use `np.linalg.eigh` for symmetric matrices; `np.linalg.eig` for general.

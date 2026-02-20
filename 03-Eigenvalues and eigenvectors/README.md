# Topic 03 — Eigenvalues and Eigenvectors

---

## 0. PREREQUISITES (Precision Check)

**Required prerequisites:**
- Vectors and vector operations (Topic 01)
- Matrices and matrix operations, including determinant and inverse (Topic 02)
- Solving polynomial equations

**How this builds forward:**
Eigenvalues and eigenvectors are the foundation of eigendecomposition (Topic 04), SVD (Topic 05), PCA, spectral clustering, and stability analysis in dynamical systems. They reveal the intrinsic geometric structure of a matrix.

---

## 1. INTUITION FIRST (Clarity Before Formalism)

### What is an eigenvector?

When a matrix transforms a vector, it usually changes both the **magnitude** and **direction** of the vector.

But some special vectors are only scaled — their direction doesn't change. These are **eigenvectors**.

**Real-world analogy:**

Imagine stretching a rubber sheet. Most arrows drawn on it get rotated and stretched. But imagine an arrow drawn along the axis of stretching — it only gets longer (or shorter), never rotated. That arrow is an eigenvector. The amount it stretches is the eigenvalue.

### Formal statement:

For a square matrix $\mathbf{A}$, a non-zero vector $\mathbf{v}$ is an **eigenvector** if:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

The scalar $\lambda$ is the corresponding **eigenvalue** — it tells you **how much** the vector was scaled.

### Why does this matter in AI?

- **PCA**: The directions of maximum variance are the eigenvectors of the covariance matrix.
- **PageRank**: The web link graph's dominant eigenvector determines which pages are most important.
- **Graph neural networks**: Spectral filters operate on eigenvectors of the graph Laplacian.
- **Stability of dynamical systems**: A system $\mathbf{x}_{t+1} = \mathbf{A}\mathbf{x}_t$ is stable if all eigenvalues satisfy $|\lambda| < 1$.

---

## 2. CORE THEORY (Rigorous but Clean)

### 2.1 Definition

Let $\mathbf{A} \in \mathbb{R}^{n \times n}$. A non-zero vector $\mathbf{v} \in \mathbb{R}^n$ and scalar $\lambda \in \mathbb{C}$ form an **eigenpair** if:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

Rearranged:
$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = \mathbf{0}$$

For a non-trivial (non-zero) solution $\mathbf{v}$ to exist, $(\mathbf{A} - \lambda\mathbf{I})$ must be **singular**:

$$\det(\mathbf{A} - \lambda \mathbf{I}) = 0$$

### 2.2 Characteristic Polynomial

$$p(\lambda) = \det(\mathbf{A} - \lambda \mathbf{I})$$

This is a degree-$n$ polynomial in $\lambda$. Its roots (possibly complex) are the eigenvalues.

**For a 2×2 matrix $\mathbf{A}$:**

$$\mathbf{A} = \begin{bmatrix}a & b \\ c & d\end{bmatrix}$$

$$\det(\mathbf{A} - \lambda\mathbf{I}) = (a-\lambda)(d-\lambda) - bc = \lambda^2 - (a+d)\lambda + (ad-bc)$$
$$= \lambda^2 - \text{tr}(\mathbf{A})\lambda + \det(\mathbf{A}) = 0$$

### 2.3 Key Relationships

For any $n\times n$ matrix $\mathbf{A}$ with eigenvalues $\lambda_1, \ldots, \lambda_n$:

$$\text{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i$$
$$\det(\mathbf{A}) = \prod_{i=1}^n \lambda_i$$

These are always true (Vieta's formulas applied to the characteristic polynomial).

### 2.4 Eigenspace

For each eigenvalue $\lambda_i$, the corresponding **eigenspace** is:

$$E_{\lambda_i} = \ker(\mathbf{A} - \lambda_i \mathbf{I}) = \{\mathbf{v} : (\mathbf{A} - \lambda_i\mathbf{I})\mathbf{v} = \mathbf{0}\}$$

This is a subspace. Its **geometric multiplicity** is $\dim(E_{\lambda_i})$.

The **algebraic multiplicity** is the multiplicity of $\lambda_i$ as a root of $p(\lambda)$.

> **Key fact:** Geometric multiplicity $\leq$ Algebraic multiplicity.

### 2.5 Properties of Eigenvalues

For a square matrix $\mathbf{A}$:

| Property                        | Statement                            |
| ------------------------------- | ------------------------------------ |
| Real symmetric matrices         | All eigenvalues are real             |
| Positive definite matrices      | All eigenvalues $> 0$                |
| Positive semi-definite matrices | All eigenvalues $\geq 0$             |
| Orthogonal matrices             | All eigenvalues have $\lvert\lambda\rvert = 1$ |
| Triangular matrices             | Eigenvalues are the diagonal entries |

**Invariance under similarity:** If $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$, then $\mathbf{A}$ and $\mathbf{B}$ have the same eigenvalues.

### 2.6 Eigenvectors of Symmetric Matrices

**Spectral theorem:** If $\mathbf{A}$ is real symmetric ($\mathbf{A} = \mathbf{A}^\top$):

1. All eigenvalues are **real**.
2. Eigenvectors corresponding to **distinct** eigenvalues are **orthogonal**.
3. $\mathbf{A}$ can be decomposed as $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ with orthonormal $\mathbf{Q}$ (eigendecomposition — see Topic 04).

### 2.7 Power Iteration (Computing the Dominant Eigenvector)

The eigenvalue with the largest absolute value is the **dominant eigenvalue**. It can be found iteratively:

1. Start with a random vector $\mathbf{b}_0$.
2. Iterate: $\mathbf{b}_{k+1} = \dfrac{\mathbf{A}\mathbf{b}_k}{\|\mathbf{A}\mathbf{b}_k\|}$
3. As $k \to \infty$, $\mathbf{b}_k \to \hat{\mathbf{v}}_1$ (dominant eigenvector).

Corresponding eigenvalue: $\lambda_1 = \mathbf{b}_k^\top \mathbf{A} \mathbf{b}_k$ (Rayleigh quotient).

### 2.8 Geometric Interpretation

Eigenvalues describe how the matrix stretches/compresses space along its eigenvector directions:

- $\lambda > 1$: stretches in that direction
- $0 < \lambda < 1$: compresses
- $\lambda = 1$: no change (fixed direction and magnitude)
- $\lambda = 0$: projects onto a hyperplane (rank loss)
- $\lambda < 0$: flips direction and scales
- Complex $\lambda$: involves rotation

---

## 3. WORKED EXAMPLES (Exactly Two)

### Example 1: Finding Eigenvalues and Eigenvectors of a 2×2 Matrix

$$\mathbf{A} = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

**Step 1: Characteristic polynomial:**
$$\det(\mathbf{A} - \lambda\mathbf{I}) = \det\begin{bmatrix}4-\lambda & 1\\2 & 3-\lambda\end{bmatrix} = (4-\lambda)(3-\lambda) - (1)(2)$$
$$= 12 - 7\lambda + \lambda^2 - 2 = \lambda^2 - 7\lambda + 10 = 0$$

**Step 2: Solve for eigenvalues:**
$$(\lambda - 5)(\lambda - 2) = 0 \implies \lambda_1 = 5, \quad \lambda_2 = 2$$

**Verify:** $\lambda_1 + \lambda_2 = 7 = \text{tr}(\mathbf{A}) = 4 + 3 \checkmark$; $\lambda_1 \lambda_2 = 10 = \det(\mathbf{A}) = 12-2 \checkmark$

**Step 3: Eigenvector for $\lambda_1 = 5$:**
$$(\mathbf{A} - 5\mathbf{I})\mathbf{v} = \begin{bmatrix}-1 & 1\\2 & -2\end{bmatrix}\begin{bmatrix}v_1\\v_2\end{bmatrix} = \mathbf{0}$$

Row 1: $-v_1 + v_2 = 0 \implies v_2 = v_1$

$$\mathbf{v}_1 = \begin{bmatrix}1\\1\end{bmatrix} \quad \text{(or any scalar multiple)}$$

**Step 4: Eigenvector for $\lambda_2 = 2$:**
$$(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \begin{bmatrix}2 & 1\\2 & 1\end{bmatrix}\begin{bmatrix}v_1\\v_2\end{bmatrix} = \mathbf{0}$$

Row 1: $2v_1 + v_2 = 0 \implies v_2 = -2v_1$

$$\mathbf{v}_2 = \begin{bmatrix}1\\-2\end{bmatrix}$$

**Verification for $\lambda_1 = 5$:**
$$\mathbf{A}\mathbf{v}_1 = \begin{bmatrix}4+1\\2+3\end{bmatrix} = \begin{bmatrix}5\\5\end{bmatrix} = 5\begin{bmatrix}1\\1\end{bmatrix} \checkmark$$

---

### Example 2: 3×3 Matrix with Repeated Eigenvalue (Exam-level)

$$\mathbf{A} = \begin{bmatrix} 2 & 0 & 1 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}$$

**Step 1: Characteristic polynomial:**

$\mathbf{A}$ is upper triangular, so eigenvalues are the diagonal entries:

$$\lambda_1 = 2 \text{ (algebraic multiplicity 2)}, \quad \lambda_2 = 3$$

Formally: $\det(\mathbf{A} - \lambda\mathbf{I}) = (2-\lambda)^2(3-\lambda)$

**Step 2: Eigenvectors for $\lambda = 2$:**
$$\mathbf{A} - 2\mathbf{I} = \begin{bmatrix}0 & 0 & 1\\0 & 0 & 0\\0 & 0 & 1\end{bmatrix}$$

Row reduction gives: $v_3 = 0$; $v_1, v_2$ are free.

$$\mathbf{v}_1 = \begin{bmatrix}1\\0\\0\end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix}0\\1\\0\end{bmatrix}$$

Geometric multiplicity = 2 = Algebraic multiplicity $\Rightarrow$ matrix is **diagonalisable** at this eigenvalue.

**Step 3: Eigenvector for $\lambda = 3$:**
$$\mathbf{A} - 3\mathbf{I} = \begin{bmatrix}-1 & 0 & 1\\0 & -1 & 0\\0 & 0 & 0\end{bmatrix}$$

From row 1: $v_1 = v_3$. From row 2: $v_2 = 0$. Set $v_3 = 1$:

$$\mathbf{v}_3 = \begin{bmatrix}1\\0\\1\end{bmatrix}$$

---

## 4. IMPLEMENTATION (Notebook Style Only)

### Importing libraries

```python
import numpy as np
import matplotlib.pyplot as plt
```

### Computing eigenvalues and eigenvectors

```python
A = np.array([[4, 1],
              [2, 3]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)
```

### Verifying Av = lambda*v

```python
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]   # columns are eigenvectors
    lhs = A @ v
    rhs = lam * v
    print(f"lambda_{i+1} = {lam:.4f}")
    print(f"  Av   = {lhs}")
    print(f"  λv   = {rhs}")
    print(f"  Match: {np.allclose(lhs, rhs)}")
```

### Verifying trace and determinant relationships

```python
print(f"tr(A)            = {np.trace(A):.4f}")
print(f"sum(eigenvalues) = {np.sum(eigenvalues):.4f}")
print(f"det(A)           = {np.linalg.det(A):.4f}")
print(f"prod(eigenvalues)= {np.prod(eigenvalues):.4f}")
```

### Symmetric matrix — real eigenvalues and orthogonal eigenvectors

```python
S = np.array([[4, 2],
              [2, 3]], dtype=float)   # symmetric

evals_s, evecs_s = np.linalg.eigh(S)   # eigh for symmetric/Hermitian — more stable
print("Eigenvalues (real):", evals_s)
print("Eigenvectors:\n", evecs_s)

# Check orthogonality
dot_product = np.dot(evecs_s[:, 0], evecs_s[:, 1])
print(f"Dot product of eigenvectors: {dot_product:.6f}  (should be ~0)")
```

### Power iteration for dominant eigenvector

```python
def power_iteration(A, num_iter=100):
    n = A.shape[0]
    b = np.random.rand(n)
    b = b / np.linalg.norm(b)
    for _ in range(num_iter):
        b_new = A @ b
        b = b_new / np.linalg.norm(b_new)
    eigenvalue = b @ A @ b   # Rayleigh quotient
    return eigenvalue, b

dominant_eval, dominant_evec = power_iteration(A)
print(f"Dominant eigenvalue (power iter): {dominant_eval:.6f}")
print(f"Dominant eigenvector:             {dominant_evec}")
print(f"numpy eig dominant eigenvalue:    {max(eigenvalues):.6f}")
```

### Visualising eigenvectors as transformation directions

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, ax in enumerate(axes):
    A_plot = np.array([[4, 1], [2, 3]]) if idx == 0 else np.array([[3, 1], [1, 3]])
    evals, evecs = np.linalg.eig(A_plot)
    title = "Non-symmetric A" if idx == 0 else "Symmetric A"

    # Draw a circle of vectors and their images
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    transformed = A_plot @ circle

    ax.plot(circle[0], circle[1], 'b-', alpha=0.4, label='Unit circle')
    ax.plot(transformed[0], transformed[1], 'r-', alpha=0.4, label='Transformed')

    # Draw eigenvectors
    for i in range(len(evals)):
        if np.isreal(evals[i]):
            v = np.real(evecs[:, i])
            lam = np.real(evals[i])
            ax.annotate('', xy=lam*v, xytext=[0,0],
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
            ax.annotate('', xy=v, xytext=[0,0],
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_aspect('equal'); ax.grid(True)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.legend(); ax.set_title(f'{title}\nλ = {np.real(evals).round(2)}')

plt.tight_layout()
plt.savefig("outputs/output_1.png")   # saved to outputs/
```

### Eigenvalue spectrum for a larger random symmetric matrix

```python
np.random.seed(42)
M = np.random.randn(50, 50)
S_large = M @ M.T   # symmetric positive semi-definite

evals_large = np.linalg.eigvalsh(S_large)

plt.figure(figsize=(8, 4))
plt.plot(sorted(evals_large, reverse=True), 'o-', markersize=4)
plt.xlabel("Index"); plt.ylabel("Eigenvalue")
plt.title("Eigenvalue Spectrum of a 50×50 Symmetric PSD Matrix")
plt.grid(True)
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

| Area                  | Role of eigenvalues/eigenvectors                                        |
| --------------------- | ----------------------------------------------------------------------- |
| PCA                   | Eigenvectors of covariance $\boldsymbol{\Sigma}$ = principal components |
| Spectral clustering   | Eigenvectors of graph Laplacian define cluster structure                |
| PageRank              | Dominant eigenvector of the web transition matrix                       |
| Stability analysis    | Eigenvalues of Jacobian determine stability of neural network dynamics  |
| Markov chains         | Stationary distribution = dominant eigenvector                          |
| Graph neural networks | Spectral convolutions: $g(\boldsymbol{\Lambda})$ applied in eigenbasis  |
| Transformer attention | Attention matrix eigenstructure controls information flow               |

### Research frontier:
- **Neural Tangent Kernel (NTK)**: The training dynamics of infinitely wide neural networks are governed by eigenvalues of the NTK matrix.
- **Hessian eigenspectrum**: Sharpness of the loss landscape is characterised by the top eigenvalues of the Hessian — directly affects optimisation (flat minima generalise better).
- **Random Matrix Theory**: Explains the distribution of eigenvalues for large random weight matrices — relevant to initialisation and pruning research.

### Historical context:
- Euler studied special directions of rotation in the 18th century.
- Cauchy formally defined eigenvalues in 1829.
- The term "eigenwert" (own value) comes from German — Von Mises named it.

---

## 6. COMMON EXAM QUESTIONS (5 Questions + Answers)

### Q1 (Conceptual): Can a matrix have more eigenvectors than eigenvalues?

**Answer:** Yes. Each eigenvalue has an associated **eigenspace** (a subspace) which may have dimension greater than 1. For example, the identity matrix $\mathbf{I}_n$ has a single eigenvalue $\lambda = 1$ but **every non-zero vector** is an eigenvector — the eigenspace is all of $\mathbb{R}^n$.

---

### Q2 (Conceptual): Why must the eigenvectors of a symmetric matrix for distinct eigenvalues be orthogonal?

**Answer:** Let $\mathbf{A}\mathbf{u} = \lambda\mathbf{u}$ and $\mathbf{A}\mathbf{v} = \mu\mathbf{v}$ with $\lambda \neq \mu$.

$$\lambda(\mathbf{u}^\top\mathbf{v}) = (\lambda\mathbf{u})^\top\mathbf{v} = (\mathbf{A}\mathbf{u})^\top\mathbf{v} = \mathbf{u}^\top\mathbf{A}^\top\mathbf{v} = \mathbf{u}^\top\mathbf{A}\mathbf{v} = \mu(\mathbf{u}^\top\mathbf{v})$$

So $(\lambda - \mu)(\mathbf{u}^\top\mathbf{v}) = 0$. Since $\lambda \neq \mu$, we get $\mathbf{u}^\top\mathbf{v} = 0$. $\checkmark$

---

### Q3 (Mathematical): Find the eigenvalues of the matrix $\mathbf{A}$ given below.

**Answer:**

$$\mathbf{A} = \begin{bmatrix}3 & -2\\1 & 0\end{bmatrix}$$

$$\det(\mathbf{A} - \lambda\mathbf{I}) = (3-\lambda)(0-\lambda) - (-2)(1) = -\lambda(3-\lambda) + 2 = \lambda^2 - 3\lambda + 2 = 0$$
$$(\lambda-1)(\lambda-2) = 0 \implies \lambda_1 = 1, \quad \lambda_2 = 2$$

Check: $\text{tr}(\mathbf{A}) = 3 = 1+2$ $\checkmark$; $\det(\mathbf{A}) = 0+2 = 2 = 1\times2$ $\checkmark$

---

### Q4 (Mathematical): If $\lambda$ is an eigenvalue of $\mathbf{A}$, what can you say about the eigenvalues of $\mathbf{A}^k$ and $\mathbf{A}^{-1}$?

**Answer:**

**For $\mathbf{A}^k$:** By induction, $\mathbf{A}^k\mathbf{v} = \lambda^k\mathbf{v}$, so eigenvalues of $\mathbf{A}^k$ are $\lambda^k$.

**For $\mathbf{A}^{-1}$:** $\mathbf{A}\mathbf{v} = \lambda\mathbf{v} \Rightarrow \mathbf{v} = \lambda\mathbf{A}^{-1}\mathbf{v} \Rightarrow \mathbf{A}^{-1}\mathbf{v} = \frac{1}{\lambda}\mathbf{v}$, so eigenvalues of $\mathbf{A}^{-1}$ are $1/\lambda$ (requires $\lambda \neq 0$, i.e., $\mathbf{A}$ invertible).

---

### Q5 (Applied): PCA uses eigenvalues of the covariance matrix. Explain why the eigenvector corresponding to the largest eigenvalue gives the direction of maximum variance.

**Answer:** The covariance matrix $\boldsymbol{\Sigma}$ is symmetric positive semi-definite. The variance of the data projected onto a unit vector $\mathbf{w}$ is:

$$\text{Var}(\mathbf{w}^\top\mathbf{X}) = \mathbf{w}^\top\boldsymbol{\Sigma}\mathbf{w}$$

Maximising this subject to $\|\mathbf{w}\|=1$ via Lagrange multipliers yields the condition $\boldsymbol{\Sigma}\mathbf{w} = \lambda\mathbf{w}$ — an eigenvalue equation. The maximum variance equals the largest eigenvalue $\lambda_1$, achieved along its corresponding eigenvector $\mathbf{w}_1$. Each subsequent principal component is the eigenvector with the next largest eigenvalue.

---

## 7. COMMON MISTAKES

**Mistake 1: Assuming all matrices have $n$ distinct eigenvalues**
- Repeated eigenvalues (algebraic multiplicity > 1) are common. The matrix may still be diagonalisable if geometric multiplicity equals algebraic multiplicity.

**Mistake 2: Normalising eigenvectors by default**
- Eigenvectors are defined up to a scalar multiple. Any scaling is valid. `np.linalg.eig` returns unit-normalised columns, but they could have arbitrary sign.

**Mistake 3: Using `np.linalg.eig` for symmetric matrices**
- For symmetric/Hermitian matrices, always use `np.linalg.eigh` — it guarantees real outputs and is more numerically stable.

**Mistake 4: Confusing algebraic and geometric multiplicity**
- A matrix is diagonalisable if and only if, for every eigenvalue, geometric multiplicity = algebraic multiplicity. Defective matrices (where they differ) cannot be diagonalised.

**Mistake 5: Thinking eigenvalues are always real**
- For non-symmetric real matrices, eigenvalues can be complex. Symmetric matrices always have real eigenvalues (guaranteed by the Spectral Theorem).

---

## 8. SUMMARY FLASHCARD

- Eigenpair: $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$; eigenvector $\mathbf{v}$ only scales, never rotates.
- Eigenvalues from: $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$ (characteristic polynomial).
- $\text{tr}(\mathbf{A}) = \sum \lambda_i$; $\det(\mathbf{A}) = \prod \lambda_i$.
- Symmetric matrices: all eigenvalues real; eigenvectors for distinct $\lambda$ are orthogonal.
- Eigenspace $E_\lambda = \ker(\mathbf{A} - \lambda\mathbf{I})$; dimension = geometric multiplicity.
- Geometric multiplicity $\leq$ Algebraic multiplicity; equality $\Rightarrow$ diagonalisable.
- In ML: PCA eigenvectors = principal components; PageRank = dominant eigenvector.

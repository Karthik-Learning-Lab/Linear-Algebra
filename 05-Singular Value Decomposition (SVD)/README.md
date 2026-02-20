# Topic 05 — Singular Value Decomposition (SVD)

---

## 0. PREREQUISITES (Precision Check)

**Required prerequisites:**
- Vectors and norms (Topic 01)
- Matrices, transpose, inverse, rank (Topic 02)
- Eigenvalues and eigenvectors (Topic 03)
- Eigendecomposition and spectral theorem (Topic 04)

**How this builds forward:**
SVD is the generalisation of eigendecomposition to **any rectangular matrix**. It underpins PCA, LSA, matrix completion, low-rank approximation, pseudo-inverse, and is the workhorse of numerical linear algebra in AI.

---

## 1. INTUITION FIRST (Clarity Before Formalism)

### What is SVD?

SVD says: **every matrix**, no matter what shape, can be decomposed into three simple operations:

$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$

1. **$\mathbf{V}^\top$** — Rotate/reflect the input space
2. **$\boldsymbol{\Sigma}$** — Stretch along each axis (by singular values)
3. **$\mathbf{U}$** — Rotate/reflect the output space

**Real-world analogy:**

Imagine taking a photo of a circle drawn on a rubber sheet. No matter how you warp the rubber sheet (rotate → stretch → rotate), the resulting ellipse can be described by SVD. The singular values tell you how much each axis got stretched; $\mathbf{U}$ and $\mathbf{V}$ tell you the orientations.

### Why is SVD more powerful than eigendecomposition?

| Property   | Eigendecomposition                        | SVD                                                  |
| ---------- | ----------------------------------------- | ---------------------------------------------------- |
| Works for  | Square matrices only                      | **Any** $m \times n$ matrix                          |
| Components | Eigenvectors (not necessarily orthogonal) | Left/right singular vectors (**always orthonormal**) |
| Values     | Can be complex                            | Singular values always **real, non-negative**        |
| Existence  | Not always (defective matrices)           | **Always exists**                                    |

SVD is the universal factorisation.

---

## 2. CORE THEORY (Rigorous but Clean)

### 2.1 Formal Definition

For any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ with $r = \text{rank}(\mathbf{A})$, the SVD is:

$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: **left singular vectors** (orthonormal columns; $\mathbf{U}^\top\mathbf{U} = \mathbf{U}\mathbf{U}^\top = \mathbf{I}_m$)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$: **singular values** on diagonal, zeros elsewhere; $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: **right singular vectors** (orthonormal columns; $\mathbf{V}^\top\mathbf{V} = \mathbf{V}\mathbf{V}^\top = \mathbf{I}_n$)

**Full SVD form:**
$$\boldsymbol{\Sigma} = \begin{bmatrix} \sigma_1 & & & \\ & \sigma_2 & & \\ & & \ddots & \\ & & & \sigma_r & \mathbf{0} \\ & \mathbf{0} & & & \mathbf{0} \end{bmatrix}_{m \times n}$$

### 2.2 Derivation

**Step 1:** Form $\mathbf{A}^\top\mathbf{A} \in \mathbb{R}^{n \times n}$. This is symmetric positive semi-definite.

**Eigendecompose:**
$$\mathbf{A}^\top\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n), \quad \lambda_i \geq 0$$

**Step 2:** Singular values:
$$\sigma_i = \sqrt{\lambda_i}$$

The columns of $\mathbf{V}$ are the **right singular vectors**.

**Step 3:** Left singular vectors (for $\sigma_i > 0$):
$$\mathbf{u}_i = \frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i$$

This ensures $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$.

**Equivalently** from $\mathbf{A}\mathbf{A}^\top \in \mathbb{R}^{m \times m}$:
$$\mathbf{A}\mathbf{A}^\top = \mathbf{U}\boldsymbol{\Lambda}'\mathbf{U}^\top$$

where $\boldsymbol{\Lambda}'$ contains the same non-zero values as $\boldsymbol{\Lambda}$ (the non-zero eigenvalues of $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$ are identical).

### 2.3 Compact (Thin/Economy) SVD

When $r < \min(m,n)$, retain only non-zero singular values:

$$\mathbf{A} = \mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top$$

where $\mathbf{U}_r \in \mathbb{R}^{m \times r}$, $\boldsymbol{\Sigma}_r \in \mathbb{R}^{r \times r}$, $\mathbf{V}_r \in \mathbb{R}^{n \times r}$.

This is the **numerically practical form** — no need to store the zero-padded parts.

### 2.4 SVD as Sum of Rank-1 Matrices

$$\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

Each term $\sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ is a rank-1 matrix. SVD breaks $\mathbf{A}$ into $r$ rank-1 "layers", each with importance $\sigma_i$ (since $\sigma_1 \geq \sigma_2 \geq \cdots$).

### 2.5 Best Low-Rank Approximation (Eckart–Young Theorem)

**Theorem (Eckart–Young, 1936):** The best rank-$k$ approximation to $\mathbf{A}$ in both Frobenius and spectral norms is:

$$\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top$$

**Approximation errors:**
$$\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$$
$$\|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}$$

This is the theoretical foundation of **image compression**, **PCA**, and **latent semantic analysis**.

### 2.6 Connection to Eigendecomposition

For symmetric positive semi-definite $\mathbf{S}$:
- Eigendecomposition: $\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$
- SVD: $\mathbf{S} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$

They coincide: $\mathbf{U} = \mathbf{V} = \mathbf{Q}$, $\boldsymbol{\Sigma} = \boldsymbol{\Lambda}$ (since all eigenvalues are non-negative).

For general square matrices, SVD $\neq$ eigendecomposition.

### 2.7 Four Fundamental Subspaces via SVD

SVD reveals the four fundamental subspaces of $\mathbf{A}$:

| Subspace                             | Basis                                    | Dimension |
| ------------------------------------ | ---------------------------------------- | --------- |
| Column space (range) of $\mathbf{A}$ | $\mathbf{u}_1, \ldots, \mathbf{u}_r$     | $r$       |
| Left null space of $\mathbf{A}$      | $\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m$ | $m-r$     |
| Row space of $\mathbf{A}$            | $\mathbf{v}_1, \ldots, \mathbf{v}_r$     | $r$       |
| Null space of $\mathbf{A}$           | $\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n$ | $n-r$     |

### 2.8 Moore-Penrose Pseudo-Inverse

For any $\mathbf{A}$, the pseudo-inverse $\mathbf{A}^+$ is:

$$\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$$

where $\boldsymbol{\Sigma}^+$ replaces each non-zero $\sigma_i$ with $1/\sigma_i$ (zeros stay zero).

This gives the **least-squares solution** to $\mathbf{A}\mathbf{x} = \mathbf{b}$:

$$\mathbf{x}^* = \mathbf{A}^+\mathbf{b}$$

- Unique minimum-norm solution when the system is underdetermined.
- Minimum-residual solution when overdetermined.

### 2.9 Matrix Norms via SVD

$$\|\mathbf{A}\|_2 = \sigma_1 \quad \text{(spectral/operator norm)}$$
$$\|\mathbf{A}\|_F = \sqrt{\sum_{i=1}^r \sigma_i^2} = \sqrt{\text{tr}(\mathbf{A}^\top\mathbf{A})} \quad \text{(Frobenius norm)}$$
$$\|\mathbf{A}\|_* = \sum_{i=1}^r \sigma_i \quad \text{(nuclear/trace norm — promotes low rank in optimisation)}$$

### 2.10 Numerical Stability

SVD is the most **numerically stable** matrix decomposition. It handles:
- Rank-deficient matrices
- Ill-conditioned matrices
- Rectangular matrices

The **condition number** of $\mathbf{A}$:

$$\kappa(\mathbf{A}) = \frac{\sigma_1}{\sigma_r}$$

Large $\kappa$ means the system is ill-conditioned — small perturbations in $\mathbf{b}$ cause large changes in $\mathbf{x}$.

---

## 3. WORKED EXAMPLES (Exactly Two)

### Example 1: Computing SVD of a Small Matrix

$$\mathbf{A} = \begin{bmatrix}3 & 0\\4 & 5\end{bmatrix}$$

**Step 1: Compute $\mathbf{A}^\top\mathbf{A}$:**
$$\mathbf{A}^\top\mathbf{A} = \begin{bmatrix}3 & 4\\0 & 5\end{bmatrix}\begin{bmatrix}3 & 0\\4 & 5\end{bmatrix} = \begin{bmatrix}9+16 & 0+20\\20+0 & 0+25\end{bmatrix} = \begin{bmatrix}25 & 20\\20 & 25\end{bmatrix}$$

**Step 2: Eigenvalues of $\mathbf{A}^\top\mathbf{A}$:**
$$\det(\mathbf{A}^\top\mathbf{A} - \lambda\mathbf{I}) = (25-\lambda)^2 - 400 = 0$$
$$(25-\lambda)^2 = 400 \implies 25-\lambda = \pm 20$$
$$\lambda_1 = 45, \quad \lambda_2 = 5$$

**Step 3: Singular values:**
$$\sigma_1 = \sqrt{45} = 3\sqrt{5}, \quad \sigma_2 = \sqrt{5}$$

**Step 4: Right singular vectors (eigenvectors of $\mathbf{A}^\top\mathbf{A}$):**

For $\lambda_1 = 45$:

$$(\mathbf{A}^\top\mathbf{A} - 45\mathbf{I})\mathbf{v} = \begin{bmatrix}-20&20\\20&-20\end{bmatrix}\mathbf{v} = \mathbf{0} \implies v_1 = v_2$$

$$\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix}$$

For $\lambda_2 = 5$:

$$(\mathbf{A}^\top\mathbf{A} - 5\mathbf{I})\mathbf{v} = \begin{bmatrix}20&20\\20&20\end{bmatrix}\mathbf{v} = \mathbf{0} \implies v_2 = -v_1$$

$$\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix}$$

**Step 5: Left singular vectors** $\mathbf{u}_i = \frac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i$:

$$\mathbf{u}_1 = \frac{1}{3\sqrt{5}} \cdot \begin{bmatrix}3&0\\4&5\end{bmatrix}\cdot\frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix} = \frac{1}{3\sqrt{10}}\begin{bmatrix}3\\9\end{bmatrix} = \frac{1}{\sqrt{10}}\begin{bmatrix}1\\3\end{bmatrix}$$

$$\mathbf{u}_2 = \frac{1}{\sqrt{5}} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix}3&0\\4&5\end{bmatrix}\begin{bmatrix}1\\-1\end{bmatrix} = \frac{1}{\sqrt{10}}\begin{bmatrix}3\\-1\end{bmatrix}$$

**Result:**
$$\mathbf{A} = \underbrace{\frac{1}{\sqrt{10}}\begin{bmatrix}1&3\\3&-1\end{bmatrix}}_{\mathbf{U}} \underbrace{\begin{bmatrix}3\sqrt{5}&0\\0&\sqrt{5}\end{bmatrix}}_{\boldsymbol{\Sigma}} \underbrace{\frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}}_{\mathbf{V}^\top}$$

---

### Example 2: Image Compression via SVD Low-Rank Approximation (Exam-level)

**Problem:** Explain how to compress a grayscale image $\mathbf{A} \in \mathbb{R}^{m \times n}$ using SVD. How many values are stored for a rank-$k$ approximation? What percentage compression at rank $k=10$ for a $256 \times 256$ image?

**Solution:**

**Full SVD of image:**
$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \sum_{i=1}^{\min(m,n)} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

**Rank-$k$ approximation:**
$$\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

**Storage count:** For each kept component:
- $\mathbf{u}_i \in \mathbb{R}^m$: $m$ values
- $\sigma_i \in \mathbb{R}$: 1 value
- $\mathbf{v}_i \in \mathbb{R}^n$: $n$ values

Total for rank-$k$: $k(m + 1 + n)$ values.

**For $256 \times 256$ image, rank-10:**
- Full image storage: $256 \times 256 = 65536$ values
- Compressed: $10 \times (256 + 1 + 256) = 10 \times 513 = 5130$ values
- **Compression ratio:** $5130 / 65536 \approx 7.8\%$ of original storage (13.6× compression)

**Quality measure:** Fraction of variance (Frobenius norm) retained:

$$\text{Quality} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$$

In practice, images have rapidly decaying singular values — rank-10 or rank-20 often captures >95% of the variance.

---

## 4. IMPLEMENTATION (Notebook Style Only)

### Importing libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
```

### Computing SVD

```python
A = np.array([[3, 0],
              [4, 5]], dtype=float)

U, sigma, Vt = np.linalg.svd(A)

print("U  =\n", U)
print("σ  =", sigma)
print("Vt =\n", Vt)

# Reconstruct
Sigma = np.zeros_like(A)
np.fill_diagonal(Sigma, sigma)
A_reconstructed = U @ Sigma @ Vt
print("\nReconstructed:\n", np.round(A_reconstructed, 6))
print("Matches:", np.allclose(A, A_reconstructed))
```

### Verifying A^T A = V Λ V^T

```python
ATA = A.T @ A
eigenvalues_ATA, V = np.linalg.eigh(ATA)
print("σ² from SVD:       ", sigma**2)
print("Eigenvalues of A^TA:", sorted(eigenvalues_ATA, reverse=True))
```

### Low-rank approximation

```python
def svd_low_rank(A, k):
    """Rank-k approximation of A using SVD."""
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

A_orig = np.random.randn(50, 30)   # random matrix
ranks = [1, 5, 10, 20]

fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(16, 3))
axes[0].imshow(A_orig, cmap='gray', aspect='auto')
axes[0].set_title("Original"); axes[0].axis('off')

U_full, sigma_full, _ = np.linalg.svd(A_orig, full_matrices=False)
total_var = np.sum(sigma_full**2)

for ax, k in zip(axes[1:], ranks):
    A_k = svd_low_rank(A_orig, k)
    captured = np.sum(sigma_full[:k]**2) / total_var * 100
    ax.imshow(A_k, cmap='gray', aspect='auto')
    ax.set_title(f"Rank {k}\n{captured:.1f}% var")
    ax.axis('off')

plt.tight_layout()
plt.savefig("outputs/output_1.png")   # saved to outputs/
```

### Image compression with SVD

```python
# Create a synthetic test image (gradient + noise)
np.random.seed(42)
x = np.linspace(0, 1, 128)
img = np.outer(np.sin(np.pi * x), np.cos(np.pi * x))
img += 0.1 * np.random.randn(*img.shape)
img = (img - img.min()) / (img.max() - img.min())   # normalise

U_img, s_img, Vt_img = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title("Original")

for i, k in enumerate([1, 5, 15, 30]):
    ax = axes.flat[i+1]
    img_k = U_img[:, :k] @ np.diag(s_img[:k]) @ Vt_img[:k, :]
    pct = np.sum(s_img[:k]**2) / np.sum(s_img**2) * 100
    err = np.linalg.norm(img - img_k, 'fro')
    ax.imshow(img_k, cmap='gray')
    ax.set_title(f"Rank {k} | {pct:.1f}% var\nFrob err={err:.4f}")

# Last plot: singular value decay
axes.flat[-1].semilogy(s_img, 'b.-', markersize=4)
axes.flat[-1].set_xlabel("Index"); axes.flat[-1].set_ylabel("σᵢ (log)")
axes.flat[-1].set_title("Singular Value Decay"); axes.flat[-1].grid(True)

plt.tight_layout()
plt.savefig("outputs/output_2.png")   # saved to outputs/
```

### Pseudo-inverse and least-squares

```python
# Overdetermined system: more equations than unknowns
A_over = np.random.randn(10, 3)
b_over = np.random.randn(10)

# Solve via pseudo-inverse (minimum norm least-squares)
A_pinv = np.linalg.pinv(A_over)
x_ls = A_pinv @ b_over

# Verify this matches np.linalg.lstsq
x_lstsq, residuals, rank, sv = np.linalg.lstsq(A_over, b_over, rcond=None)

print("Solution via pinv:   ", np.round(x_ls, 6))
print("Solution via lstsq:  ", np.round(x_lstsq, 6))
print("Match:", np.allclose(x_ls, x_lstsq))

# Residual
print(f"Residual ||Ax-b||: {np.linalg.norm(A_over @ x_ls - b_over):.6f}")
```

### Condition number

```python
A_ill = np.array([[1, 1], [1, 1.001]])   # nearly singular
A_well = np.eye(2)

cond_ill = np.linalg.cond(A_ill)
cond_well = np.linalg.cond(A_well)

print(f"Condition number (ill):  {cond_ill:.2f}")
print(f"Condition number (well): {cond_well:.2f}")
# High condition number → numerically unstable inversions
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

| Area                               | How SVD is used                                                                                                                         |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **PCA**                            | SVD of the data matrix $\mathbf{X}$ directly gives principal components (equivalent to eigendecomposition of $\boldsymbol{\Sigma}$)     |
| **Latent Semantic Analysis (LSA)** | SVD of term-document matrix extracts semantic topics                                                                                    |
| **Collaborative filtering**        | Matrix factorisation (Netflix Prize): approximate rating matrix with $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$                     |
| **Image compression / denoising**  | Truncated SVD retains signal, discards noise (small $\sigma_i$)                                                                         |
| **Pseudo-inverse / least squares** | $\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$ — robust solution to $\mathbf{A}\mathbf{x}=\mathbf{b}$                  |
| **Low-rank fine-tuning (LoRA)**    | Fine-tune LLMs by restricting weight updates to low-rank: $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$, inspired by low-rank SVD structure |
| **Nuclear norm regularisation**    | Promotes low-rank solutions in matrix completion by penalising $\|\mathbf{A}\|_* = \sum \sigma_i$                                       |
| **Signal processing**              | SVD filters noise by zeroing small singular values (truncated SVD)                                                                      |

### Modern Research:
- **LoRA (Low-Rank Adaptation)** (Hu et al., 2022): Fine-tunes large language models by decomposing weight updates as products of low-rank matrices — directly motivated by SVD structure. Now used in almost all LLM fine-tuning.
- **Randomised SVD**: Halko, Martinsson & Tropp (2011) — compute approximate SVD in $O(mn\log k)$ instead of $O(mn\min(m,n))$ using random projections. Used in scikit-learn's PCA.
- **Robust PCA**: Decomposes $\mathbf{M} = \mathbf{L} + \mathbf{S}$ (low-rank + sparse) for corrupted data recovery.

### Historical evolution:
- Beltrami and Jordan independently discovered SVD in 1873.
- Golub-Reinsch algorithm (1970) made numerical SVD practical.
- Eckart-Young theorem (1936) proved SVD gives the best low-rank approximation.

---

## 6. COMMON EXAM QUESTIONS (5 Questions + Answers)

### Q1 (Conceptual): Why is SVD preferred over eigendecomposition in numerical practice?

**Answer:** SVD is universally applicable — it works for any $m \times n$ matrix, including non-square and rank-deficient matrices, and always produces real non-negative singular values with orthonormal factors. Eigendecomposition requires a square matrix and can fail for defective matrices (insufficient independent eigenvectors). SVD is also more numerically stable since $\mathbf{U}$ and $\mathbf{V}$ are orthogonal (condition number = 1), making it robust to floating-point errors.

---

### Q2 (Conceptual): What is the relationship between SVD and PCA?

**Answer:** Both are equivalent for centred data. For centred data matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$, the covariance matrix is $\boldsymbol{\Sigma} = \frac{1}{N}\mathbf{X}^\top\mathbf{X}$. The SVD of $\mathbf{X}$ is $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}_X\mathbf{V}^\top$. The right singular vectors $\mathbf{V}$ are exactly the eigenvectors of $\boldsymbol{\Sigma}$ (principal components), and the squared singular values $\sigma_i^2/N$ are the corresponding variances. PCA via SVD of $\mathbf{X}$ is numerically preferred over eigendecomposition of $\boldsymbol{\Sigma}$ because it avoids squaring the matrix (which can worsen conditioning).

---

### Q3 (Mathematical): Write the full SVD of the rectangular matrix $\mathbf{A}$ given below.

**Answer:**

$$\mathbf{A} = \begin{bmatrix}2&0\\0&3\\0&0\end{bmatrix} \in \mathbb{R}^{3\times 2}$$

The singular values are the non-zero entries: $\sigma_1 = 3$, $\sigma_2 = 2$ (sorted descending).

$$\mathbf{U} = \begin{bmatrix}0&1&0\\1&0&0\\0&0&1\end{bmatrix}, \quad \boldsymbol{\Sigma} = \begin{bmatrix}3&0\\0&2\\0&0\end{bmatrix}, \quad \mathbf{V}^\top = \begin{bmatrix}0&1\\1&0\end{bmatrix}$$

**Verify:**

$$\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \begin{bmatrix}0&1&0\\1&0&0\\0&0&1\end{bmatrix}\begin{bmatrix}3&0\\0&2\\0&0\end{bmatrix}\begin{bmatrix}0&1\\1&0\end{bmatrix} = \begin{bmatrix}0&2\\3&0\\0&0\end{bmatrix}\begin{bmatrix}0&1\\1&0\end{bmatrix} = \begin{bmatrix}2&0\\0&3\\0&0\end{bmatrix} = \mathbf{A} \checkmark$$

---

### Q4 (Mathematical): Prove that the rank-$k$ truncated SVD is the best rank-$k$ approximation in Frobenius norm.

**Answer (Eckart-Young Theorem sketch):**

Let $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ and $\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i\mathbf{v}_i^\top$.

$$\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \left\|\sum_{i=k+1}^r \sigma_i\mathbf{u}_i\mathbf{v}_i^\top\right\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$$

For any other rank-$k$ matrix $\mathbf{B}$: by the min-max characterisation of singular values:

$$\sigma_{k+1} = \min_{\text{rank}(\mathbf{C}) \leq k} \|\mathbf{A} - \mathbf{C}\|_2$$

The Frobenius norm bound follows: no rank-$k$ matrix can reduce the residual below $\sqrt{\sum_{i=k+1}^r \sigma_i^2}$.

---

### Q5 (Applied): LoRA fine-tunes LLMs by updating weights as $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ where $\mathbf{B} \in \mathbb{R}^{d \times r}$, $\mathbf{A} \in \mathbb{R}^{r \times d}$, $r \ll d$. Why is this related to SVD and what is the parameter savings?

**Answer:** The product $\mathbf{B}\mathbf{A}$ is a rank-$r$ matrix — exactly the form of the top-$r$ truncated SVD. The hypothesis is that **task-specific weight updates lie in a low-dimensional subspace** (supported empirically by the observation that pre-trained weight matrices have low "intrinsic rank"). By factoring the update this way, the number of trainable parameters drops from $d^2$ (full matrix) to $2dr$ (two low-rank matrices). For $d=4096$, $r=16$: $d^2 = 16{,}777{,}216$ vs $2 \times 4096 \times 16 = 131{,}072$ — a **128× reduction** in trainable parameters while achieving near-full fine-tuning performance.

---

## 7. COMMON MISTAKES

**Mistake 1: Assuming $\mathbf{U}$ and $\mathbf{V}$ are the same**
- $\mathbf{U} \in \mathbb{R}^{m \times m}$ spans the output (column) space; $\mathbf{V} \in \mathbb{R}^{n \times n}$ spans the input (row) space. They are different unless $m = n$ and special structure exists.

**Mistake 2: Confusing `full_matrices=True` vs `full_matrices=False` in NumPy**
- `np.linalg.svd(A, full_matrices=True)`: returns full square $\mathbf{U}$ and $\mathbf{V}$ (standard SVD).
- `np.linalg.svd(A, full_matrices=False)`: returns compact/thin SVD with dimensions $m \times r$ and $r \times n$. Use compact for practical work.

**Mistake 3: Setting up $\boldsymbol{\Sigma}$ incorrectly for reconstruction**
- `np.linalg.svd` returns `sigma` as a 1D array, not a matrix. You must use `np.diag(sigma)` and account for the correct shape for non-square matrices.

**Mistake 4: Confusing singular values with eigenvalues**
- Singular values of $\mathbf{A}$ are the square roots of the eigenvalues of $\mathbf{A}^\top\mathbf{A}$. They are not eigenvalues of $\mathbf{A}$ itself (unless $\mathbf{A}$ is symmetric PSD).

**Mistake 5: Using full SVD when truncated is needed**
- For image compression or PCA, you only need the top-$k$ components. Computing full SVD when $k \ll \min(m,n)$ wastes time. Use `scipy.sparse.linalg.svds` or randomised SVD (`sklearn.utils.extmath.randomized_svd`) for large matrices.

---

## 8. SUMMARY FLASHCARD

- SVD: $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ — exists for **any** $m \times n$ matrix; always real, always orthonormal factors.
- $\mathbf{U}$ = left singular vectors (column space); $\mathbf{V}$ = right singular vectors (row space); $\sigma_i \geq 0$ sorted descending.
- Singular values: $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top\mathbf{A})}$; $\text{rank}(\mathbf{A})$ = number of non-zero $\sigma_i$.
- Eckart-Young: best rank-$k$ approx is $\mathbf{A}_k = \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$; error = $\sqrt{\sum_{i>k}\sigma_i^2}$.
- Pseudo-inverse: $\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top$ — gives minimum-norm least-squares solution.
- Condition number: $\kappa = \sigma_1/\sigma_r$ — measures numerical sensitivity.
- Key applications: PCA, image compression, recommender systems, LoRA, LSA, denoising.

# Topic 02 — Matrices and Matrix Operations

---

## 0. PREREQUISITES (Precision Check)

**Required prerequisites:**
- Vectors and vector operations (Topic 01)
- Basic scalar arithmetic and algebra
- Summation notation $\sum$

**How this builds forward:**
Matrices are the central objects of linear algebra. Every linear transformation, every neural network layer, every dataset is represented as a matrix. Eigendecomposition, SVD, and all downstream AI algorithms operate on matrices.

---

## 1. INTUITION FIRST (Clarity Before Formalism)

### What is a matrix?

A matrix is a **rectangular grid of numbers**. Think of it as a spreadsheet — rows and columns.

**Real-world analogy:**

A cinema has 5 rows and 8 seats per row. The seating plan is a $5 \times 8$ matrix. Each entry is either 0 (empty) or 1 (occupied).

In AI: a grayscale image of $28 \times 28$ pixels is a matrix where each entry is a pixel intensity value.

### Why do matrices matter?

Matrices represent **linear transformations** — they describe how to rotate, scale, shear, or project vectors. When you multiply a matrix by a vector $\mathbf{A}\mathbf{x}$, you're applying a transformation that maps $\mathbf{x}$ to a new vector.

Every layer of a neural network performs $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ — a matrix-vector multiplication.

---

## 2. CORE THEORY (Rigorous but Clean)

### 2.1 Formal Definition

A matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

Notation: $a_{ij}$ denotes the entry in row $i$, column $j$.

### 2.2 Special Matrices

| Matrix                   | Definition                                |
| ------------------------ | ----------------------------------------- |
| Square                   | $m = n$                                   |
| Zero matrix $\mathbf{0}$ | All entries are 0                         |
| Identity $\mathbf{I}_n$  | $a_{ij} = 1$ if $i=j$, else 0             |
| Diagonal                 | $a_{ij} = 0$ for $i \neq j$               |
| Upper triangular         | $a_{ij} = 0$ for $i > j$                  |
| Lower triangular         | $a_{ij} = 0$ for $i < j$                  |
| Symmetric                | $\mathbf{A} = \mathbf{A}^\top$            |
| Skew-symmetric           | $\mathbf{A} = -\mathbf{A}^\top$           |
| Orthogonal               | $\mathbf{A}^\top \mathbf{A} = \mathbf{I}$ |

### 2.3 Matrix Addition

For $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$:

$$(\mathbf{A} + \mathbf{B})_{ij} = a_{ij} + b_{ij}$$

**Requires same dimensions.** Element-wise. Commutative and associative.

### 2.4 Scalar Multiplication

$$(\alpha \mathbf{A})_{ij} = \alpha \cdot a_{ij}$$

Every entry is multiplied by $\alpha$.

### 2.5 Matrix Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$, the product $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$:

$$c_{ij} = \sum_{l=1}^k a_{il} \cdot b_{lj}$$

The $(i,j)$ entry is the **dot product** of row $i$ of $\mathbf{A}$ with column $j$ of $\mathbf{B}$.

**Requirement:** inner dimensions must match. $\mathbf{A}_{m \times k} \cdot \mathbf{B}_{k \times n}$ is valid; $\mathbf{A}_{m \times k} \cdot \mathbf{B}_{p \times n}$ (with $k \neq p$) is not.

**Properties:**
- **Not commutative**: $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ in general
- Associative: $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$
- Distributive: $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{A}\mathbf{B} + \mathbf{A}\mathbf{C}$
- $\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}$

**Computational complexity:** Naive matrix multiplication of $\mathbf{A}_{m\times k}$ and $\mathbf{B}_{k \times n}$ costs $O(mkn)$. For square $n \times n$ matrices: $O(n^3)$. (Strassen's algorithm achieves $O(n^{2.807})$.)

### 2.6 Transpose

$$(\mathbf{A}^\top)_{ij} = a_{ji}$$

Rows become columns and vice versa.

**Properties:**
- $(\mathbf{A}^\top)^\top = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top$
- $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top$ (**order reverses**)
- $(\mathbf{A}\mathbf{B}\mathbf{C})^\top = \mathbf{C}^\top \mathbf{B}^\top \mathbf{A}^\top$

### 2.7 Trace

The **trace** of a square matrix:

$$\text{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}$$

**Properties:**
- $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$
- $\text{tr}(\alpha \mathbf{A}) = \alpha \, \text{tr}(\mathbf{A})$
- $\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})$ (**cyclic property**)
- $\text{tr}(\mathbf{A}) = \sum_i \lambda_i$ (sum of eigenvalues)

### 2.8 Determinant

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, the determinant $\det(\mathbf{A})$ (or $|\mathbf{A}|$) is a scalar encoding the **signed volume scaling factor** of the transformation.

**2×2 case:**
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**3×3 case (cofactor expansion):**
$$\det(\mathbf{A}) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

**Properties:**
- $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B})$
- $\det(\mathbf{A}^\top) = \det(\mathbf{A})$
- $\det(\mathbf{A}^{-1}) = 1/\det(\mathbf{A})$
- $\det(\alpha \mathbf{A}) = \alpha^n \det(\mathbf{A})$ for $\mathbf{A} \in \mathbb{R}^{n \times n}$
- $\det(\mathbf{A}) = \prod_i \lambda_i$ (product of eigenvalues)
- $\det(\mathbf{A}) = 0 \Leftrightarrow \mathbf{A}$ is singular (non-invertible)

### 2.9 Matrix Inverse

For a square matrix $\mathbf{A}$, the inverse $\mathbf{A}^{-1}$ satisfies:

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

**Existence condition:** $\mathbf{A}^{-1}$ exists $\Leftrightarrow \det(\mathbf{A}) \neq 0$ $\Leftrightarrow \mathbf{A}$ is full rank.

**2×2 closed form:**
$$\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

**Properties:**
- $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
- $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$ (order reverses)
- $(\mathbf{A}^\top)^{-1} = (\mathbf{A}^{-1})^\top$

### 2.10 Geometric Interpretation of Matrix-Vector Multiplication

$\mathbf{A}\mathbf{x}$ transforms the vector $\mathbf{x}$. The columns of $\mathbf{A}$ are the images of the standard basis vectors:

$$\mathbf{A}\mathbf{e}_j = \mathbf{A}_{:,j} \quad \text{(the $j$-th column of $\mathbf{A}$)}$$

So the matrix encodes where each basis vector lands after the transformation.

### 2.11 Rank

The **rank** of $\mathbf{A}$ is the number of linearly independent rows (= number of linearly independent columns).

$$\text{rank}(\mathbf{A}) \leq \min(m, n)$$

- **Full rank**: $\text{rank}(\mathbf{A}) = \min(m,n)$ — no redundant information
- **Rank-deficient**: $\text{rank}(\mathbf{A}) < \min(m,n)$ — rows/columns are linearly dependent

---

## 3. WORKED EXAMPLES (Exactly Two)

### Example 1: Matrix Operations

Let:
$$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

**a) Addition:**
$$\mathbf{A} + \mathbf{B} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$$

**b) Multiplication $\mathbf{A}\mathbf{B}$:**
$$c_{11} = (1)(5) + (2)(7) = 19, \quad c_{12} = (1)(6) + (2)(8) = 22$$
$$c_{21} = (3)(5) + (4)(7) = 43, \quad c_{22} = (3)(6) + (4)(8) = 50$$
$$\mathbf{A}\mathbf{B} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

**c) Transpose:**
$$\mathbf{A}^\top = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$$

**d) Determinant of $\mathbf{A}$:**
$$\det(\mathbf{A}) = (1)(4) - (2)(3) = 4 - 6 = -2$$

**e) Inverse of $\mathbf{A}$:**
$$\mathbf{A}^{-1} = \frac{1}{-2}\begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\ 3/2 & -1/2 \end{bmatrix}$$

**Verification:**
$$\mathbf{A}\mathbf{A}^{-1} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix} -2 & 1 \\ 3/2 & -1/2 \end{bmatrix} = \begin{bmatrix} -2+3 & 1-1 \\ -6+6 & 3-2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \checkmark$$

---

### Example 2: Solving a Linear System via Matrix Operations (Exam-level)

**Problem:** Solve the system:
$$2x_1 + x_2 = 5$$
$$5x_1 + 3x_2 = 13$$

**Matrix form:** $\mathbf{A}\mathbf{x} = \mathbf{b}$ where:
$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 5 \\ 13 \end{bmatrix}$$

**Step 1:** Compute $\det(\mathbf{A})$:
$$\det(\mathbf{A}) = (2)(3) - (1)(5) = 6 - 5 = 1$$

**Step 2:** Since $\det(\mathbf{A}) = 1 \neq 0$, the system has a unique solution.

**Step 3:** Compute $\mathbf{A}^{-1}$:
$$\mathbf{A}^{-1} = \frac{1}{1}\begin{bmatrix} 3 & -1 \\ -5 & 2 \end{bmatrix} = \begin{bmatrix} 3 & -1 \\ -5 & 2 \end{bmatrix}$$

**Step 4:** Solve $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$:
$$\mathbf{x} = \begin{bmatrix} 3 & -1 \\ -5 & 2 \end{bmatrix}\begin{bmatrix} 5 \\ 13 \end{bmatrix} = \begin{bmatrix} 15 - 13 \\ -25 + 26 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

**Verification:** $\mathbf{A}\mathbf{x} = \begin{bmatrix}2(2)+1(1)\\5(2)+3(1)\end{bmatrix} = \begin{bmatrix}5\\13\end{bmatrix} = \mathbf{b} \checkmark$

**Bonus — show $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$** using the example matrices from Example 1:

$$(\mathbf{A}\mathbf{B})^\top = \begin{bmatrix} 19 & 43 \\ 22 & 50 \end{bmatrix}$$

$$\mathbf{B}^\top\mathbf{A}^\top = \begin{bmatrix}5 & 7\\6 & 8\end{bmatrix}\begin{bmatrix}1 & 3\\2 & 4\end{bmatrix} = \begin{bmatrix}5+14 & 15+28\\6+16 & 18+32\end{bmatrix} = \begin{bmatrix}19 & 43\\22 & 50\end{bmatrix} \checkmark$$

---

## 4. IMPLEMENTATION (Notebook Style Only)

### Importing libraries

```python
import numpy as np
import matplotlib.pyplot as plt
```

### Creating matrices

```python
A = np.array([[1, 2],
              [3, 4]], dtype=float)

B = np.array([[5, 6],
              [7, 8]], dtype=float)

print("A =\n", A)
print("B =\n", B)
```

### Basic operations

```python
print("A + B =\n", A + B)
print("2 * A =\n", 2 * A)
print("A^T  =\n", A.T)
print("Trace(A) =", np.trace(A))
```

### Matrix multiplication

```python
AB = A @ B          # preferred over np.dot for matrix products
BA = B @ A
print("AB =\n", AB)
print("BA =\n", BA)
print("AB == BA?", np.allclose(AB, BA))   # should be False
```

### Determinant and inverse

```python
det_A = np.linalg.det(A)
A_inv = np.linalg.inv(A)

print(f"det(A) = {det_A:.4f}")
print("A_inv =\n", A_inv)
print("A @ A_inv (should be I):\n", np.round(A @ A_inv, 6))
```

### Solving a linear system

```python
A_sys = np.array([[2, 1],
                  [5, 3]], dtype=float)
b_sys = np.array([5, 13], dtype=float)

# Via inverse
x_via_inv = np.linalg.inv(A_sys) @ b_sys

# Via numpy solve (numerically preferred)
x_via_solve = np.linalg.solve(A_sys, b_sys)

print("Solution via inv:  ", x_via_inv)
print("Solution via solve:", x_via_solve)
print("Verification Ax=b:", np.allclose(A_sys @ x_via_solve, b_sys))
```

### Visualising a linear transformation

```python
def plot_transform(M, title, ax):
    # Original unit square
    square = np.array([[0,1,1,0,0],
                       [0,0,1,1,0]], dtype=float)
    transformed = M @ square

    ax.plot(square[0], square[1], 'b-', lw=2, label='Original')
    ax.plot(transformed[0], transformed[1], 'r-', lw=2, label='Transformed')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(title)
    ax.grid(True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

M_scale = np.array([[2, 0], [0, 0.5]])       # scale x by 2, y by 0.5
M_rotate = np.array([[0, -1], [1, 0]])        # 90-degree rotation
M_shear = np.array([[1, 1], [0, 1]])          # shear

plot_transform(M_scale,  "Scaling", axes[0])
plot_transform(M_rotate, "Rotation 90°", axes[1])
plot_transform(M_shear,  "Shear", axes[2])

plt.tight_layout()
plt.savefig("outputs/output_1.png")   # saved to outputs/
```

### Rank computation

```python
C_full = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
C_rank2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])   # rows sum makes rank deficient

print("Rank of identity:", np.linalg.matrix_rank(C_full))
print("Rank of rank-deficient matrix:", np.linalg.matrix_rank(C_rank2))
```

### Frobenius norm

```python
frob_norm = np.linalg.norm(A, 'fro')
print(f"Frobenius norm of A: {frob_norm:.4f}")
# Frobenius = sqrt(sum of squared entries) = sqrt(trace(A^T A))
print(f"Verify: {np.sqrt(np.trace(A.T @ A)):.4f}")
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

| Area                      | How matrices appear                                                                                      |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| Neural network layers     | $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$; weight matrix $\mathbf{W}$                             |
| Attention in Transformers | $\text{Attention} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ |
| Datasets                  | $\mathbf{X} \in \mathbb{R}^{N \times D}$ — $N$ samples, $D$ features                                     |
| Covariance matrices       | $\boldsymbol{\Sigma} = \frac{1}{N}\mathbf{X}^\top\mathbf{X}$ — used in PCA                               |
| Graph adjacency           | $\mathbf{A}_{ij}=1$ if edge exists; central to GNNs                                                      |
| Backpropagation           | Jacobians are matrices of partial derivatives                                                            |

### Historical evolution:
- Arthur Cayley introduced matrix algebra in 1858
- Gaussian elimination (row operations) was known to ancient Chinese mathematicians
- Modern numerical linear algebra (LAPACK, BLAS) underpins all deep learning frameworks (PyTorch, TensorFlow use optimised BLAS routines for `@`)

### Limitations:
- Dense matrix multiplication at $O(n^3)$ is expensive for large $n$
- Numerical issues arise with near-singular matrices (use pseudo-inverse, regularisation)
- Non-commutativity of multiplication is a common source of errors in derivations

---

## 6. COMMON EXAM QUESTIONS (5 Questions + Answers)

### Q1 (Conceptual): Why is matrix multiplication not commutative in general?

**Answer:** Matrix multiplication $\mathbf{A}\mathbf{B}$ means "apply transformation $\mathbf{B}$ first, then $\mathbf{A}$." The order of transformations matters geometrically — rotating 90° then reflecting is different from reflecting then rotating 90°. Furthermore, the dimensions may not even permit $\mathbf{B}\mathbf{A}$ when $\mathbf{A}_{m \times k}$ and $\mathbf{B}_{k \times n}$ with $m \neq n$.

---

### Q2 (Conceptual): What does $\det(\mathbf{A}) = 0$ mean geometrically?

**Answer:** The transformation $\mathbf{A}$ collapses space — it maps $\mathbb{R}^n$ into a lower-dimensional subspace (a line, plane, etc.). No volume is preserved. This means columns of $\mathbf{A}$ are linearly dependent, $\mathbf{A}$ is singular (not invertible), and the system $\mathbf{A}\mathbf{x} = \mathbf{b}$ has either no solution or infinitely many.

---

### Q3 (Mathematical): If $\mathbf{A}$ is symmetric, prove $\mathbf{A}^2$ is also symmetric.

**Answer:** We need to show $(\mathbf{A}^2)^\top = \mathbf{A}^2$:

$$(\mathbf{A}^2)^\top = (\mathbf{A}\mathbf{A})^\top = \mathbf{A}^\top \mathbf{A}^\top$$

Since $\mathbf{A}$ is symmetric: $\mathbf{A}^\top = \mathbf{A}$, so:

$$(\mathbf{A}^2)^\top = \mathbf{A}\mathbf{A} = \mathbf{A}^2 \quad \checkmark$$

---

### Q4 (Mathematical): Compute the inverse of $\mathbf{A} = \begin{bmatrix} 3 & 1 \\ 2 & 1 \end{bmatrix}$ and solve $\mathbf{A}\mathbf{x} = \begin{bmatrix}7\\5\end{bmatrix}$.

**Answer:**

$\det(\mathbf{A}) = 3 - 2 = 1$

$$\mathbf{A}^{-1} = \frac{1}{1}\begin{bmatrix}1 & -1 \\ -2 & 3\end{bmatrix} = \begin{bmatrix}1 & -1 \\ -2 & 3\end{bmatrix}$$

$$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b} = \begin{bmatrix}1 & -1 \\ -2 & 3\end{bmatrix}\begin{bmatrix}7\\5\end{bmatrix} = \begin{bmatrix}2\\1\end{bmatrix}$$

---

### Q5 (Applied): In a neural network layer, $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ where $\mathbf{W} \in \mathbb{R}^{64 \times 128}$, $\mathbf{x} \in \mathbb{R}^{128}$, $\mathbf{b} \in \mathbb{R}^{64}$. How many parameters does this layer have? What is the output shape?

**Answer:** The weight matrix $\mathbf{W}$ has $64 \times 128 = 8192$ parameters, and the bias $\mathbf{b}$ has $64$ parameters, giving $8192 + 64 = \mathbf{8256}$ trainable parameters. The output $\mathbf{y} \in \mathbb{R}^{64}$ — the matrix maps the 128-dimensional input to a 64-dimensional output.

---

## 7. COMMON MISTAKES

**Mistake 1: Writing $\mathbf{A}\mathbf{B} = \mathbf{B}\mathbf{A}$**
- Matrix multiplication is not commutative. Always check dimension compatibility and ordering.

**Mistake 2: Transposing products in wrong order**
- $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top$, NOT $\mathbf{A}^\top \mathbf{B}^\top$.
- This is one of the most common errors in backpropagation derivations.

**Mistake 3: Using `*` for matrix multiplication in NumPy**
- `A * B` in NumPy is element-wise (Hadamard product), not matrix multiplication.
- Use `A @ B` or `np.matmul(A, B)` for matrix products.

**Mistake 4: Inverting a matrix when solving a linear system**
- `np.linalg.inv(A) @ b` is numerically unstable. Always prefer `np.linalg.solve(A, b)`, which uses LU factorisation internally.

**Mistake 5: Confusing rank-deficient with zero matrix**
- A rank-deficient matrix has some zero eigenvalues but is generally not a zero matrix. Its columns are just linearly dependent.

---

## 8. SUMMARY FLASHCARD

- $\mathbf{A} \in \mathbb{R}^{m \times n}$: rectangular array with $m$ rows, $n$ columns.
- Matrix multiply: $(AB)_{ij} = \sum_k a_{ik}b_{kj}$; valid when inner dims match; $O(n^3)$ for square.
- $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$; $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$ — order always reverses.
- $\det(\mathbf{A}) = 0 \Leftrightarrow$ singular $\Leftrightarrow$ non-invertible $\Leftrightarrow$ rank-deficient.
- $\text{tr}(\mathbf{A}) = \sum_i a_{ii} = \sum_i \lambda_i$; $\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$.
- Geometric view: columns of $\mathbf{A}$ show where basis vectors land under the transformation.
- Use `np.linalg.solve(A, b)` for linear systems — more stable than explicit inversion.

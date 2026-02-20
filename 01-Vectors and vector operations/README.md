# Topic 01 — Vectors and Vector Operations

---

## 0. PREREQUISITES (Precision Check)

**Required prerequisites:**
- Basic arithmetic and algebra
- Coordinate geometry (2D and 3D points)
- Notion of magnitude and direction (intuitive)

**How this builds forward:**
Everything in linear algebra — matrices, eigenvalues, SVD, neural network weights — is built on top of vectors. A vector is the fundamental object. Understanding it deeply makes every subsequent topic click naturally.

---

## 1. INTUITION FIRST (Clarity Before Formalism)

### What is a vector?

A vector is simply an **ordered list of numbers** that represents something with both **magnitude** (size) and **direction**.

**Real-world analogy:**

Imagine you're giving someone directions: "Walk 3 blocks East and 4 blocks North." That instruction captures *how far* and *which way*. That's a vector: $\mathbf{v} = (3, 4)$.

A temperature reading like $23°C$ is just a number (a **scalar**). But wind velocity — "20 km/h heading North-East" — has both magnitude and direction. That's a **vector**.

### Why do vectors exist?

- To represent **positions** in space
- To represent **transformations** (moves, rotations)
- To encode **features** of data (e.g., a person's height, weight, age as a vector)
- To measure **similarity** between objects (via dot products)

### Visual intuition:

A 2D vector $\mathbf{v} = (3, 4)$ is an arrow from the origin $(0,0)$ to the point $(3, 4)$. Its length (magnitude) is $\sqrt{3^2 + 4^2} = 5$.

---

## 2. CORE THEORY (Rigorous but Clean)

### 2.1 Formal Definition

A **vector** $\mathbf{v} \in \mathbb{R}^n$ is an ordered $n$-tuple of real numbers:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

This is a **column vector** (default convention). A row vector is denoted $\mathbf{v}^\top = [v_1, v_2, \ldots, v_n]$.

### 2.2 Vector Addition

Given $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}$$

**Geometric interpretation:** Tip-to-tail rule. Place the tail of $\mathbf{v}$ at the tip of $\mathbf{u}$; the sum points to the new tip.

**Properties:**
- Commutativity: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- Associativity: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- Identity: $\mathbf{v} + \mathbf{0} = \mathbf{v}$
- Inverse: $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$

### 2.3 Scalar Multiplication

For scalar $\alpha \in \mathbb{R}$:

$$\alpha \mathbf{v} = \begin{bmatrix} \alpha v_1 \\ \alpha v_2 \\ \vdots \\ \alpha v_n \end{bmatrix}$$

**Geometric interpretation:** Stretches (or flips) the vector. $\alpha > 1$ stretches, $0 < \alpha < 1$ shrinks, $\alpha < 0$ reverses direction.

### 2.4 Norm (Magnitude / Length)

The **Euclidean norm** (L2 norm) of $\mathbf{v}$:

$$\|\mathbf{v}\| = \|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2} = \sqrt{\mathbf{v}^\top \mathbf{v}}$$

A **unit vector** has $\|\hat{\mathbf{v}}\| = 1$. Normalise any vector:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**Other norms:**

| Norm           | Formula                                  | Meaning                |
| -------------- | ---------------------------------------- | ---------------------- |
| L1 (Manhattan) | $\|\mathbf{v}\|_1 = \sum_i               | v_i                    | $ | Sum of absolute values |
| L2 (Euclidean) | $\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$ | Straight-line distance |
| L\infty (Max)  | $\|\mathbf{v}\|_\infty = \max_i          | v_i                    | $ | Largest component      |

### 2.5 Dot Product (Inner Product)

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^\top \mathbf{v} = \sum_{i=1}^n u_i v_i$$

**Geometric form:**

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

where $\theta$ is the angle between $\mathbf{u}$ and $\mathbf{v}$.

**Key properties:**
- Commutative: $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$
- Bilinear: $(\alpha\mathbf{u}) \cdot \mathbf{v} = \alpha(\mathbf{u} \cdot \mathbf{v})$
- $\mathbf{v} \cdot \mathbf{v} = \|\mathbf{v}\|^2$

**What the dot product measures:**
- $\mathbf{u} \cdot \mathbf{v} > 0$: vectors point in the same general direction ($\theta < 90°$)
- $\mathbf{u} \cdot \mathbf{v} = 0$: vectors are **orthogonal** ($\theta = 90°$)
- $\mathbf{u} \cdot \mathbf{v} < 0$: vectors point in opposite directions ($\theta > 90°$)

**Cauchy-Schwarz Inequality:**

$$|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \|\mathbf{v}\|$$

### 2.6 Cross Product (3D only)

For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^3$:

$$\mathbf{u} \times \mathbf{v} = \begin{vmatrix} \mathbf{e}_1 & \mathbf{e}_2 & \mathbf{e}_3 \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \end{vmatrix} = \begin{bmatrix} u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1 \end{bmatrix}$$

**Properties:**
- Result is a vector **perpendicular** to both $\mathbf{u}$ and $\mathbf{v}$
- Magnitude: $\|\mathbf{u} \times \mathbf{v}\| = \|\mathbf{u}\| \|\mathbf{v}\| \sin\theta$ = area of parallelogram
- Anti-commutative: $\mathbf{u} \times \mathbf{v} = -(\mathbf{v} \times \mathbf{u})$
- $\mathbf{u} \times \mathbf{u} = \mathbf{0}$
- **Not defined for dimensions other than 3 (and 7)**

### 2.7 Projection

The scalar projection of $\mathbf{u}$ onto $\mathbf{v}$:

$$\text{proj}_{\text{scalar}} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|}$$

The vector projection of $\mathbf{u}$ onto $\mathbf{v}$:

$$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v} = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} \mathbf{v}$$

**Geometric interpretation:** The shadow of $\mathbf{u}$ cast onto the line defined by $\mathbf{v}$.

### 2.8 Linear Combination and Span

A **linear combination** of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$:

$$\mathbf{w} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k, \quad \alpha_i \in \mathbb{R}$$

The **span** is the set of all possible linear combinations:

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \Bigl\lbrace \sum_{i=1}^k \alpha_i \mathbf{v}_i \mid \alpha_i \in \mathbb{R} \Bigr\rbrace$$

### 2.9 Linear Independence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are **linearly independent** if:

$$\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0} \implies \alpha_1 = \alpha_2 = \cdots = \alpha_k = 0$$

Otherwise they are **linearly dependent** — at least one can be written as a combination of the others.

---

## 3. WORKED EXAMPLES (Exactly Two)

### Example 1: Basic Operations

Given $\mathbf{u} = (1, 2, 3)^\top$ and $\mathbf{v} = (4, 0, -1)^\top$.

**a) Vector addition:**
$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} 1+4 \\ 2+0 \\ 3+(-1) \end{bmatrix} = \begin{bmatrix} 5 \\ 2 \\ 2 \end{bmatrix}$$

**b) Scalar multiplication by $\alpha = 3$:**
$$3\mathbf{u} = \begin{bmatrix} 3 \\ 6 \\ 9 \end{bmatrix}$$

**c) Norm of $\mathbf{u}$:**
$$\|\mathbf{u}\| = \sqrt{1^2 + 2^2 + 3^2} = \sqrt{1 + 4 + 9} = \sqrt{14} \approx 3.742$$

**d) Dot product:**
$$\mathbf{u} \cdot \mathbf{v} = (1)(4) + (2)(0) + (3)(-1) = 4 + 0 - 3 = 1$$

**e) Angle between $\mathbf{u}$ and $\mathbf{v}$:**
$$\|\mathbf{v}\| = \sqrt{16 + 0 + 1} = \sqrt{17}$$
$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} = \frac{1}{\sqrt{14} \cdot \sqrt{17}} = \frac{1}{\sqrt{238}} \approx 0.0648$$
$$\theta = \arccos(0.0648) \approx 86.3°$$

**f) Cross product $\mathbf{u} \times \mathbf{v}$:**
$$\mathbf{u} \times \mathbf{v} = \begin{bmatrix} (2)(-1) - (3)(0) \\ (3)(4) - (1)(-1) \\ (1)(0) - (2)(4) \end{bmatrix} = \begin{bmatrix} -2 \\ 13 \\ -8 \end{bmatrix}$$

**Verification (orthogonality):**
$$\mathbf{u} \cdot (\mathbf{u} \times \mathbf{v}) = (1)(-2) + (2)(13) + (3)(-8) = -2 + 26 - 24 = 0 \checkmark$$

---

### Example 2: Projection and Orthogonality (Exam-level)

**Problem:** Let $\mathbf{a} = (2, 1, -1)^\top$ and $\mathbf{b} = (1, 3, 2)^\top$.

1. Find the projection of $\mathbf{a}$ onto $\mathbf{b}$.
2. Decompose $\mathbf{a}$ into components parallel and perpendicular to $\mathbf{b}$.
3. Verify orthogonality of the perpendicular component.

**Step 1: Compute $\mathbf{a} \cdot \mathbf{b}$ and $\|\mathbf{b}\|^2$:**
$$\mathbf{a} \cdot \mathbf{b} = (2)(1) + (1)(3) + (-1)(2) = 2 + 3 - 2 = 3$$
$$\|\mathbf{b}\|^2 = 1 + 9 + 4 = 14$$

**Step 2: Vector projection (parallel component):**
$$\mathbf{a}_\parallel = \text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b} = \frac{3}{14} \begin{bmatrix} 1 \\ 3 \\ 2 \end{bmatrix} = \begin{bmatrix} 3/14 \\ 9/14 \\ 6/14 \end{bmatrix} = \begin{bmatrix} 3/14 \\ 9/14 \\ 3/7 \end{bmatrix}$$

**Step 3: Perpendicular component:**
$$\mathbf{a}_\perp = \mathbf{a} - \mathbf{a}_\parallel = \begin{bmatrix} 2 \\ 1 \\ -1 \end{bmatrix} - \begin{bmatrix} 3/14 \\ 9/14 \\ 3/7 \end{bmatrix} = \begin{bmatrix} 28/14 - 3/14 \\ 14/14 - 9/14 \\ -14/14 - 6/14 \end{bmatrix} = \begin{bmatrix} 25/14 \\ 5/14 \\ -20/14 \end{bmatrix}$$

**Step 4: Verify $\mathbf{a}_\perp \perp \mathbf{b}$:**
$$\mathbf{a}_\perp \cdot \mathbf{b} = \frac{25}{14}(1) + \frac{5}{14}(3) + \frac{-20}{14}(2) = \frac{25 + 15 - 40}{14} = \frac{0}{14} = 0 \checkmark$$

**Step 5: Verify reconstruction:**
$$\mathbf{a}_\parallel + \mathbf{a}_\perp = \begin{bmatrix} 3/14 + 25/14 \\ 9/14 + 5/14 \\ 6/14 - 20/14 \end{bmatrix} = \begin{bmatrix} 28/14 \\ 14/14 \\ -14/14 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \\ -1 \end{bmatrix} = \mathbf{a} \checkmark$$

---

## 4. IMPLEMENTATION (Notebook Style Only)

### Importing libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

### Defining vectors

```python
u = np.array([1, 2, 3], dtype=float)
v = np.array([4, 0, -1], dtype=float)
a = np.array([2, 1, -1], dtype=float)
b = np.array([1, 3, 2], dtype=float)
```

### Vector addition and scalar multiplication

```python
add_result = u + v
scalar_result = 3 * u
print("u + v =", add_result)
print("3u   =", scalar_result)
```

### Norms

```python
norm_l1 = np.linalg.norm(u, ord=1)
norm_l2 = np.linalg.norm(u, ord=2)
norm_inf = np.linalg.norm(u, ord=np.inf)
print(f"L1 norm: {norm_l1:.4f}")
print(f"L2 norm: {norm_l2:.4f}")
print(f"Linf norm: {norm_inf:.4f}")
```

### Dot product and angle

```python
dot = np.dot(u, v)
cos_theta = dot / (np.linalg.norm(u) * np.linalg.norm(v))
theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
print(f"Dot product: {dot}")
print(f"Angle between u and v: {theta_deg:.2f} degrees")
```

### Cross product

```python
cross = np.cross(u, v)
print("u × v =", cross)
# Verify orthogonality
print("u · (u×v) =", np.dot(u, cross))   # should be 0
print("v · (u×v) =", np.dot(v, cross))   # should be 0
```

### Vector projection

```python
def vector_projection(a, b):
    """Project vector a onto vector b."""
    return (np.dot(a, b) / np.dot(b, b)) * b

a_parallel = vector_projection(a, b)
a_perp = a - a_parallel

print("a_parallel =", a_parallel)
print("a_perp     =", a_perp)
print("Orthogonality check:", np.isclose(np.dot(a_perp, b), 0))
print("Reconstruction check:", np.allclose(a_parallel + a_perp, a))
```

### Visualising 2D vectors

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Vector addition
ax = axes[0]
origin = np.zeros(2)
u2 = np.array([1, 2])
v2 = np.array([3, 1])
ax.quiver(*origin, *u2, color='blue', scale=1, scale_units='xy', angles='xy', label=r'$\mathbf{u}$')
ax.quiver(*origin, *v2, color='red', scale=1, scale_units='xy', angles='xy', label=r'$\mathbf{v}$')
ax.quiver(*origin, *(u2 + v2), color='green', scale=1, scale_units='xy', angles='xy', label=r'$\mathbf{u}+\mathbf{v}$')
ax.quiver(*u2, *v2, color='red', alpha=0.3, scale=1, scale_units='xy', angles='xy')
ax.set_xlim(-1, 6); ax.set_ylim(-1, 5)
ax.set_aspect('equal'); ax.grid(True)
ax.legend(); ax.set_title('Vector Addition (Tip-to-Tail)')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

# Plot 2: Projection
ax = axes[1]
b2 = np.array([3, 1])
a2 = np.array([2, 3])
proj = (np.dot(a2, b2) / np.dot(b2, b2)) * b2
perp = a2 - proj
ax.quiver(*origin, *a2, color='blue', scale=1, scale_units='xy', angles='xy', label=r'$\mathbf{a}$')
ax.quiver(*origin, *b2, color='red', scale=1, scale_units='xy', angles='xy', label=r'$\mathbf{b}$')
ax.quiver(*origin, *proj, color='orange', scale=1, scale_units='xy', angles='xy', label=r'$\mathrm{proj}_b\mathbf{a}$')
ax.plot([proj[0], a2[0]], [proj[1], a2[1]], 'g--', label=r'$\mathbf{a}_\perp$')
ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
ax.set_aspect('equal'); ax.grid(True)
ax.legend(); ax.set_title('Vector Projection')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

plt.tight_layout()
plt.savefig("outputs/output_1.png")   # saved to outputs/
```

### Visualising dot product vs angle

```python
angles = np.linspace(0, np.pi, 200)
u_fixed = np.array([1, 0])
dot_values = [np.cos(theta) for theta in angles]  # unit vectors assumed

plt.figure(figsize=(8, 4))
plt.plot(np.degrees(angles), dot_values, 'b-', lw=2)
plt.axhline(0, color='k', lw=0.5)
plt.xlabel("Angle θ (degrees)")
plt.ylabel("Dot product (cos θ)")
plt.title("Dot Product vs Angle Between Unit Vectors")
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

| Area                               | How vectors appear                                                    |
| ---------------------------------- | --------------------------------------------------------------------- |
| Feature encoding                   | Each data point is a vector in $\mathbb{R}^n$                         |
| Neural networks                    | Every layer input/output is a vector; weights are matrices            |
| Word embeddings (Word2Vec, GloVe)  | Words mapped to dense vectors; cosine similarity via dot product      |
| Attention mechanism (Transformers) | Queries, Keys, Values are all vectors; attention = scaled dot product |
| SVM                                | Decision boundary is a hyperplane defined by support vectors          |
| PCA / SVD                          | Decomposition into principal component vectors                        |
| Information retrieval              | Cosine similarity compares document vectors                           |

### Historical evolution:
- Vectors formalised in the 19th century by Grassmann and Hamilton
- Gibbs and Heaviside established modern 3D notation
- Hilbert extended to infinite-dimensional spaces (Hilbert spaces), foundational for quantum mechanics and functional analysis

### Limitations:
- Euclidean dot product is not always the right measure (use kernels for non-linear similarity)
- High-dimensional vectors suffer from the **curse of dimensionality** (distances become uninformative)

---

## 6. COMMON EXAM QUESTIONS (5 Questions + Answers)

### Q1 (Conceptual): What does a zero dot product between two non-zero vectors mean?

**Answer:** The vectors are **orthogonal** — they are perpendicular to each other ($\theta = 90°$). This means they share no component in common directions. Geometrically, neither vector has any projection along the other.

---

### Q2 (Conceptual): Why is the cross product only defined in 3D?

**Answer:** The cross product produces a vector perpendicular to both inputs. This requires exactly one "free" dimension — in 2D there is no perpendicular dimension in the plane, and in higher dimensions it's not unique (there are infinitely many perpendicular directions). There is a generalisation in 7D using octonions, but it's not standard.

---

### Q3 (Mathematical): Prove the Cauchy-Schwarz inequality $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \|\mathbf{v}\|$.

**Answer:** Consider $f(\lambda) = \|\mathbf{u} + \lambda \mathbf{v}\|^2 \geq 0$ for all $\lambda \in \mathbb{R}$:

$$f(\lambda) = \|\mathbf{u}\|^2 + 2\lambda(\mathbf{u} \cdot \mathbf{v}) + \lambda^2 \|\mathbf{v}\|^2 \geq 0$$

This is a quadratic in $\lambda$ that is always non-negative. Its discriminant must be $\leq 0$:

$$\Delta = 4(\mathbf{u} \cdot \mathbf{v})^2 - 4\|\mathbf{u}\|^2\|\mathbf{v}\|^2 \leq 0$$
$$\Rightarrow (\mathbf{u} \cdot \mathbf{v})^2 \leq \|\mathbf{u}\|^2 \|\mathbf{v}\|^2$$
$$\Rightarrow |\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \|\mathbf{v}\| \quad \checkmark$$

---

### Q4 (Mathematical): Find the unit vector in the direction of $\mathbf{v} = (3, -4)^\top$ and compute its projection onto $\mathbf{w} = (1, 1)^\top$.

**Answer:**

Step 1 — Unit vector:
$$\|\mathbf{v}\| = \sqrt{9 + 16} = 5, \quad \hat{\mathbf{v}} = \frac{1}{5}\begin{bmatrix}3 \\ -4\end{bmatrix} = \begin{bmatrix}0.6 \\ -0.8\end{bmatrix}$$

Step 2 — Projection of $\hat{\mathbf{v}}$ onto $\mathbf{w}$:
$$\text{proj}_{\mathbf{w}} \hat{\mathbf{v}} = \frac{\hat{\mathbf{v}} \cdot \mathbf{w}}{\|\mathbf{w}\|^2} \mathbf{w} = \frac{0.6 - 0.8}{2} \begin{bmatrix}1 \\ 1\end{bmatrix} = \frac{-0.2}{2}\begin{bmatrix}1 \\ 1\end{bmatrix} = \begin{bmatrix}-0.1 \\ -0.1\end{bmatrix}$$

---

### Q5 (Applied): In word embeddings, the word "king" is represented as a vector. Explain how the operation $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$ works using vector arithmetic.

**Answer:** Word2Vec learns embeddings where semantic relationships correspond to geometric directions in vector space. The direction $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}}$ captures the "royalty without gender" offset. Adding $\mathbf{v}_{\text{woman}}$ reintroduces the female gender direction. The resulting vector lies closest (by cosine similarity) to $\mathbf{v}_{\text{queen}}$ in the embedding space. This demonstrates that vector addition/subtraction can encode analogical reasoning — a direct application of linear structure in representation spaces.

---

## 7. COMMON MISTAKES

**Mistake 1: Confusing dot product (scalar) with cross product (vector)**
- Dot product $\mathbf{u} \cdot \mathbf{v}$ returns a **scalar** — it measures alignment.
- Cross product $\mathbf{u} \times \mathbf{v}$ returns a **vector** — it measures perpendicularity and area.
- They are fundamentally different operations.

**Mistake 2: Applying cross product in dimensions other than 3**
- The cross product formula only works in $\mathbb{R}^3$. For $n$-dimensional orthogonality, use projection and the null space instead.

**Mistake 3: Forgetting that $\mathbf{u} \cdot \mathbf{v} = 0$ requires both vectors to be non-zero for orthogonality**
- The zero vector is orthogonal to everything by convention, but this is a degenerate case.

**Mistake 4: Computing projection onto a non-unit vector without dividing by $\|\mathbf{v}\|^2$**
- $\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\mathbf{v}$, not $(\mathbf{u} \cdot \mathbf{v})\mathbf{v}$.
- The latter is only correct when $\mathbf{v}$ is already a unit vector.

**Mistake 5: Conflating norms — using L2 when L1 is appropriate**
- In sparse ML problems (LASSO), L1 norm is preferred as it promotes sparsity. Not all problems use L2 distance.

---

## 8. SUMMARY FLASHCARD

- A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ numbers with magnitude and direction.
- $\|\mathbf{v}\|_2 = \sqrt{\mathbf{v}^\top \mathbf{v}}$; normalise: $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$.
- Dot product: $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$; measures angular alignment.
- $\mathbf{u} \cdot \mathbf{v} = 0 \Leftrightarrow$ orthogonal; $> 0$ same direction; $< 0$ opposite.
- Projection: $\text{proj}_{\mathbf{v}}\mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\mathbf{v}$ (shadow of $\mathbf{u}$ on $\mathbf{v}$).
- Cross product (3D only): $\mathbf{u} \times \mathbf{v}$ is perpendicular to both; magnitude = area of parallelogram.
- Cauchy-Schwarz: $|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\|$; equality iff vectors are parallel.

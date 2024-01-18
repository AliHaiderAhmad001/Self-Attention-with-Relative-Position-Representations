# **Self-Attention with Relative Position Representations**

**Note:** I advise you to watch [this video](https://www.youtube.com/watch?v=DwaBQbqh5aE) if you do not have prior knowledge of this type of representation.

## Overview
The "Self-Attention with Relative Position Representations" extends the original Transformer's self-attention mechanism to efficiently consider representations of relative positions or distances between sequence elements. This enhancement proves particularly effective in tasks like machine translation, offering improved performance compared to models relying solely on absolute position representations.

### **Self-Attention**

1. **Input and Output Dimensions:**
   - Input sequence $x=\left(x_{1}, \ldots, x_{n}\right)$ of $n$ elements with dimension $d_a$.
   - Output sequence $z=\left(z_{1}, \ldots, z_{n}\right)$ with dimension $d_z$.

2. **Computation of $z_i$:**
$$z_{i}=\sum_{j=1}^{n} \alpha_{i j}\left(x_{j} W^{V}\right)$$

3. **Weight Coefficients $\alpha_{ij}\$:**
   $$
\alpha_{i j}=\frac{\exp e_{i j}}{\sum_{k=1}^{n} \exp e_{i k}}
$$

4. **Compatibility Function $e_{ij}$:**
$$
e_{i j}=\frac{\left(x_{i} W^{Q}\right)\left(x_{j} W^{K}\right)^{T}}{\sqrt{d_{z}}}
$$

### **Relation-aware Self-Attention**

1. **Extension to Self-Attention:**
   - An extension to self-attention is proposed to consider the pairwise relationships between input elements in the sense that the input is modeled as a labeled, directed, fully-connected graph. The edge between input elements $x_i$ and $x_j$ is represented by vectors: $a_{i j}^{V}, a_{i j}^{K} \in \mathbb{R}^{d_{a}}$. So, $a_{i j}^{V}, a_{i j}^{K}$ model the interaction between positions $i$ and $j$. These representations can be shared across attention heads. We use $d_a = d_z$ .
   - These representations can be shared across attention heads.
   - Edges can capture information about the relative position differences between input elements.

2. **Modification of $z_i$ with Edge Information $a_{ij}^V$:**
   $$
z_{i}=\sum_{j=1}^{n} \alpha_{i j}\left(x_{j} W^{V}+a_{i j}^{V}\right)
$$
3. **Modification of Compatibility Function with Edge Information $a_{ij}^K$:**
$$
e_{i j}=\frac{x_{i} W^{Q}\left(x_{j} W^{K}+a_{i j}^{K}\right)^{T}}{\sqrt{d_{z}}}
$$

### **Relative Position Representations**

1. **Edge Labels for Relative Positions:**
   - Edge representations $a_{ij}^K, a_{ij}^V$ capture relative position differences.
   - The maximum relative position is clipped to a maximum absolute value of $k$. It is hypothesized that precise relative position information is not useful beyond a certain distance. Clipping the maximum distance also enables the model to generalize to sequence lengths not seen during training.
   - So, $2k+1$ unique edge labels are considered.

2. **Learnable Relative Position Representations:**
   - $a_{ij}^K, a_{ij}^V$ are determined using learnable relative position representations $w^K, w^V$.
   - Clipping function:
$$
\begin{aligned}
a_{i j}^{K} & =w_{\operatorname{clip}(j-i, k)}^{K} \\
a_{i j}^{V} & =w_{\operatorname{clip}(j-i, k)}^{V} \\
\operatorname{clip}(x, k) & =\max (-k, \min (k, x))
\end{aligned}
$$

3. **Learnable Vectors \( w^K, w^V \):**
   - $w^K = (w_{-k}^K, \ldots, w_k^K)$
   - $w^V = (w_{-k}^V, \ldots, w_k^V)$

### **Analysis**

* Among the first, Shaw, Uszkoreit, and Vaswani (2018) introduced an alternative method for incorporating both absolute and relative position encodings.
* The use of relative position representations allows the model to consider pairwise relationships and capture information about the relative position differences between input elements, enhancing its ability to understand the sequence structure.
* Although it cannot directly be compared with the effect of simple addition of position embeddings, they roughly omit the position–position interaction and have only one unit–position term. In addition, they do not share the projection matrices but directly model the pairwise position interaction with the vectors $a$. In an ablation analysis they found that solely adding $a_{i j}^{K}$ might be sufficient.
*  To reduce space complexity, they share the parameters across attention heads. While it is not explicitly mentioned in their paper we understand that they add the position information in each layer but do not share the parameters. The authors find that relative position embeddings perform better in machine translation and the combination of absolute and relative embeddings does not improve the performance.

## References
[Ref.1](https://direct.mit.edu/coli/article/48/3/733/111478/Position-Information-in-Transformers-An-Overview)
[Ref.2](https://doi.org/10.18653/v1/N18-2074)
[Ref.3](https://sh-tsang.medium.com/review-self-attention-with-relative-position-representations-266ab2f78dd7)
[Ref.4](https://github.com/evelinehong/Transformer_Relative_Position_PyTorch)

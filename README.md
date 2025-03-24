# FacilityLocation.jl

To estimate serving costs, it makes more sense to iterate over customers because of memory locality in the current model.
Also, a list of customers assigned to every facility would have varying length, whereas the mapping from customer to facility is a single index.
Track move target of customer: possible without asymptotic memory increase.
Track next open facility: also possible.
 
GPU version: leverage matrix-matrix products instead of matrix-vector for memory bandwidth.
One block = one instance.
Notations:
 
- $o_{i}$: facility $i$ open in instance $k$.
- $c_{i,j}$: serving cost from facility $i$ to customer $j$ in instance $k$
 
Total serving cost $c = \sum_j \min_i [o_{i} c_{i,j} + (1-o_{i}) \infty]$
 
Store duplicate versions of $o$ for each solution in the neighborhood inside shared memory.
Bring $C$ from global memory tile-by-tile.
$O \in \{0, 1\}^{I \times V}$ (smol) and $C \in \mathbb{R}^{I \times J}$ (big).
If the neighborhood is big, split up $O$.
Works for arbitrary neighborhoods.
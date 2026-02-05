
```
Given trajectory T's length as N, M = ceil(N/S), 
Set Q1, Q2, S, 
where Q1+Q2+S = 256, and S >= Q1
\Rightarrow M = \ceil(N/S)
\Rightarrow T's head padding length = Q1_points, pad with t_1
\Rightarrow T's tail padding length = Q2_points + (N mod (M*S)), pad with t_N 
\Rightarrow Padded trajecotry T'={t'_i}^{N + (Q1_points + Q2_points + (N mod (M*S)))}_{i=1}, length = N + (Q1_points + Q2_points + (N mod (M*S)))  
\Rightarrow distance between two adjancy headbuckle's start is (S-Q1+Q1) = S

for a given chunk Cj, suppose Cj's head buckle is Hj, Payload is Pj, tail buckle is Rj
Then we have (all index start with 1):
chunk Cj: 
    - head buckle Hj = {t'_i}^{(j-1)*S+Q1}_{i=(j-1)*S+1}
    - payload     Pj = {t'_i}^{j*S+Q1}_{i=(j-1)*S+Q1+1}, where (j-1)*S+Q1 = end of j-1 th head buckle 
    - tail buckle Rj = {t'_i}^{j*S+Q1+Q2}_{i=j*S+Q1+1}

case 1: 
N = 257
Q1 = 1 byte = 8 point, 
Q2 = 1 byte = 8 point, 
S = 256- (Q1+Q2)_point = 240
\Rightarrow M = \ceil(N/S) = 2
\Rightarrow T's head padding length = Q1 = 8 points
\Rightarrow T's tail padding length = Q2+(N mod M*S) = 8 + 223 = 231 points

Trajectory:  T  = {t_i}^{N=257}_i=1
Padded Traj: T' = {t_1}^{8} union {t_i}^{N=257}_{i=1} union {t_N}^{25= Q2_point + (N mod (M*S))}
             T' = {t'_i}^{257+(Q1_point + Q2_point + (N mod S))}_{i=1} = {t'_i}^{257+239=496}
chunk 1: 
    - head buckle = {t_1}^{8}_i=1
    - payload     = {t_i}^{240=S1_end}_{i=1=S1_start}
    - tail buckle = {t_i}^{256=C1_end}_{i=241}
chunk 2:
    - head buckle = {t_1}^{240=S1_end}_{i=232=S1_start-Q2_point}
    - payload     = {t_i}^{481}_{i=241=S1_end+1=S2_start}
    - tail buckle = {t_i}^{256}_{i=241}
```

## Corrected Trajectory Chunking & Padding Spec

Given a trajectory  
$T = \{t_i\}_{i=1}^{N}$

Choose parameters $Q_1, Q_2, S$ such that:

- $Q_1 + S + Q_2 = 256$
- $S \ge Q_1$

---

### Number of chunks
$$
M = \left\lceil \frac{N}{S} \right\rceil
$$

---

### Padding rules

- **Head padding**
  - Length: $Q_1$
  - Padding value: $t_1$

- **Payload-completion padding**
  - Length:
    $$
    M \cdot S - N \in [0, S-1]
    $$
  - Padding value: $t_N$

- **Tail buckle padding**
  - Length: $Q_2$
  - Padding value: $t_N$

Total tail-side padding:
$$
Q_2 + (M \cdot S - N)
$$

---

### Padded trajectory

Define the padded trajectory:
$$
T' = \{t'_i\}_{i=1}^{Q_1 + M\cdot S + Q_2}
$$

Total length:
$$
|T'| = Q_1 + M\cdot S + Q_2
$$

---

### Chunk indexing (1-based)

For chunk $C_j$, $j = 1,\dots,M$:

- **Head buckle** $H_j$ (length $Q_1$):
  $$
  H_j = \{t'_i\}_{i=(j-1)S+1}^{(j-1)S+Q_1}
  $$

- **Payload** $P_j$ (length $S$):
  $$
  P_j = \{t'_i\}_{i=(j-1)S+Q_1+1}^{jS+Q_1}
  $$

- **Tail buckle** $R_j$ (length $Q_2$):
  $$
  R_j = \{t'_i\}_{i=jS+Q_1+1}^{jS+Q_1+Q_2}
  $$

Each chunk length is $Q_1 + S + Q_2 = 256$.

The distance between start indices of adjacent head buckles is:
$$
((j)S+1) - ((j-1)S+1) = S
$$


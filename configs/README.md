## Guidelines

To achieve terminal phase of training, exact recovery should be possible. In case of SBM graphs, the regime of $p = \frac{an}{\log n}, q = \frac{bn}{\log n}$ is the most suited for our experiments [Abbe et.al'17](https://arxiv.org/pdf/1703.10146.pdf). The community detection paper by [Zhengdao et.al '19](https://arxiv.org/pdf/1705.08415.pdf) focuses of detection but not on recovery, which is the primary motivation in our work. For exact recovery, a sharp threshold is given by:

$$
\frac{(a+b)}{2} - \sqrt{ab} > 1
$$

While experimenting, one can explore regimes where LHS $>1, = 1, < 1$ to better understand the collapse properties. As a guideline, when $n = 1000, k = 2, P=[0.5, 0.5]$, and assuming $a+b = 4$ (a heuristic constant), the regimes of:

| a   | b | $\frac{(a+b)}{2} - \sqrt{ab}$ | p     | q |
| --- | - | ----------------------------- | ----- | - |
| 4   | 0 | 2                             | 0.012 | 0 |
| 3.8 | 0.2 | 1.13                        | 0.0114 | 0.0006 |
| 3.7325 | 0.2675 | $\sim 1$              | 0.0112 | 0.0008 |
| 3.6 | 0.4 | 0.8                         | 0.0108 | 0.0012 |
| 3   | 1 | 0.27                          | 0.009  | 0.003  |


demonstrates the relation between recovery and the extent of variability collapse. When $\frac{(a+b)}{2} - \sqrt{ab} < 1$, it is impossible to recover the communities and for $\frac{(a+b)}{2} - \sqrt{ab} > 1$, the recovery is information theoretically possible. At the threshold of $\frac{(a+b)}{2} - \sqrt{ab} = 1$, exact recovery is possible if $a, b > 0$.
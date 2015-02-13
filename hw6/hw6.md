
<head><meta charset="UTF-8"></head>
<script type="text/javascript"
  src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

1.

$\triangledown F(A, B) = $

$(\frac{\partial \frac{1}{N}\sum^N_{n=1}ln(1+exp(-y_n(A(w \phi(x_n)+b)+B)))}{\partial A},
\frac{\partial \frac{1}{N}\sum^N_{n=1}ln(1+exp(-y_n(A(w \phi(x_n)+b)+B)))}{\partial B})
=$

($K=-y_n(A(w \phi(x_n)+b)+B)$)

$\frac{1}{N} \sum^N_{n=1}
(\frac{\partial ln(1+e^K)}{\partial A},
\frac{\partial ln(1+e^K)}{\partial B}) = $

$\frac{1}{N} \sum^N_{n=1} \frac{e^K}{1+e^K}
(\frac{\partial K}{\partial A}, \frac{\partial K}{\partial B}) = $

$\frac{1}{N} \sum^N_{n=1} \frac{e^K}{1+e^K}
(-y_n(w\phi(x_n)+b), -y_n) = $

$\frac{1}{N} \sum^N_{n=1} \theta(-y_n(Az_n+B))
(-y_n(w\phi(x_n)+b), -y_n) = $

$\frac{1}{N} \sum^N_{n=1} (-y_np_nz_n, -y_np_n) = $

[a]

---

2.

$\frac{\partial -y_np_n(w\phi(x_n)+b)}{\partial A} = 
-y_nz_n\frac{p_n}{\partial A} =
-y_nz_np_n(1-p_n)\frac{\partial Az_n+B}{\partial A}
= -y_nz^2_np_n(1-p_n)$

$\frac{\partial -y_np_n(w\phi(x_n)+b)}{\partial B} = 
-y_nz_n\frac{p_n}{\partial B} =
-y_nz_np_n(1-p_n)\frac{\partial Az_n+B}{\partial B}
= -y_nz_np_n(1-p_n)$

$\frac{\partial -y_np_n}{\partial A} = 
-y_n\frac{p_n}{\partial A} =
-y_np_n(1-p_n)\frac{\partial Az_n+B}{\partial A}
= -y_nz_np_n(1-p_n)$

$\frac{\partial -y_np_n}{\partial B} = 
-y_n\frac{p_n}{\partial B} =
-y_np_n(1-p_n)\frac{\partial Az_n+B}{\partial B}
= -y_np_n(1-p_n)$

[a]

---

3.

ridge regression: $w = (Z^TZ + \lambda I)^{-1} Z^Ty$

kernel ridge regression: $y = (K + \lambda I)^{-1} k(x)$

Kernel function $K$ is N x N

[b]

---

4.

$-\epsilon - \xi^{\vee}_n \leq y_n - w^T\phi(x_n) - b \geq \epsilon + \xi^{\wedge}_n $
$\Rightarrow |y_n - w^T\phi(x_n) - b| \leq \epsilon + max(\xi^{\vee}_n, \xi^{\wedge}_n$

if there is no violation on the hyperplane, $|y_n - w^T\phi(x_n) - b| \leq 0$
$\xi^{\vee}_n = \xi^{\wedge}_n = 0$, there will be no penalty.

otherwise,
since we are minimizing \frac{1}{2}w^Tw + C\sum^N_{n=1}(\xi^{\vee^2}_n + \xi^{\wedge^2}_n)

one of $\xi^{\vee^2}_n$ or $\xi^{\wedge^2}_n$ = 0 (only violating one side of
constraint a time), $|y_n - w^T\phi(x_n) - b| - \epsilon = \xi^{\vee}_n$ or $\xi^{\wedge}_n$

$\Rightarrow (|y_n - w^T\phi(x_n) - b| - \epsilon)^2 = \xi^{\vee^2}_n + \xi^{\wedge^2}_n$

unconstrained form of (P2) will be
$\frac{1}{2}w^Tw + C\sum^N_{n=1}(max(0, |y_n - w^T\phi(x_n) - b| - \epsilon))^2$

[d]

---

5.

$\frac{1}{2}w^Tw + C\sum^N_{n=1}(max(0, |y_n - w^T\phi(x_n) - b| - \epsilon))^2
=$

$\frac{1}{2}\sum^N_{m=1}\sum^N_{n=1}\beta_n\beta_mK(x_n, x_m) +
C\sum^N_{n=1}(max(0, |y_n - \sum^N_{m=1}\beta_mK(x_n, x_m) - b| - \epsilon))^2=$

gradient

$\frac{\partial \sum^N_{m=1}\sum^N_{n=1}\beta_n\beta_mK(x_n, x_m)}{\partial
\beta_m} = 2\sum^N_{n=1}\beta_nK(x_n, x_m)$

if $|y_n - s_n| \geq \epsilon$

&nbsp;&nbsp;
if $y_n - \sum^N_{m=1}\beta_mK(x_n, x_m) - b = y_n-s_n \geq 0$

&nbsp;&nbsp;&nbsp;&nbsp;
$((y_n - \sum^N_{m=1}\beta_mK(x_n, x_m) - b) - \epsilon)(-K(x_n, x_m))=$

&nbsp;&nbsp;&nbsp;&nbsp;
$(|y_n - s_n| - \epsilon)(-K(x_n, x_m))$

&nbsp;&nbsp;
if $y_n - \sum^N_{m=1}\beta_mK(x_n, x_m) - b = y_n-s_n \leq 0$

&nbsp;&nbsp;&nbsp;&nbsp;
$(-(y_n - \sum^N_{m=1}\beta_mK(x_n, x_m) - b) - \epsilon)(K(x_n, x_m)))=$

&nbsp;&nbsp;&nbsp;&nbsp;
$(|y_n - s_n| - \epsilon)(K(x_n, x_m)))$

if $|y_n - s_n| \leq \epsilon$

&nbsp;&nbsp;
gradient = 0

Combining the if statments and remove the max function

$\Rightarrow \sum^N_{n=1}\beta_nK(x_n, x_m) -
2C\sum^N_{n=1} [|y_n - s_n| \geq \epsilon]
sign(y_n - s_n)(|y_n - s_n| - \epsilon) K(x_n, x_m)$

[a]

---

6.

$e_0 = \sum^M_{m=1}\tilde{y_m}^2$

$\frac{1}{M}\sum^M_{m=1}(g_t(\tilde{x_m}) - \tilde{y_m})^2 =$

$\frac{1}{M}\sum^M_{m=1}(g_t(\tilde{x_m})^2 - 2g_t(\tilde{x_m})\tilde{y_m} + \tilde{y_m}^2) =$

$\frac{1}{M}(s_t + e_0 - 2\sum^M_{m=1}g_t(\tilde{x_m})\tilde{y_m}) = e_t$

$\Rightarrow \sum^M_{m=1}g_t(\tilde{x_m})\tilde{y_m} = \frac{M}{2}(e_0 + s_t - e_t)$

[a]

---

7.

probability for a data to be out of sampled is $(1-\frac{1}{N})^{pN}$

Approximately
$(1-\frac{1}{N})^{pN}N = ((1-\frac{1}{N})^{N})^p * N =
\frac{1}{e}^p * N$ examples will not be sampled as N close to $\inf$.

[b]

---

8.

[a] false, the size of $X \leq (R-L+1)^{d}$. the dimension of an example
can't be infinity before transform.

[b] true, both $g_{+1, 1, L-1}, g_{-1, 3, R+1}$ = 1 for comtains all examples.

[c] true, it would obviously be right for s = 1 since it is positive ray.

For s = -1, it would only affect $x_i = ceiling(\theta)$,
$sign(x_i - ceiling(\theta)) = -1$ and $sign(x_i - \theta) = 1$

but s*sign(x_i - ceiling(\theta)) will make them equal.

Therefore, $g_{-1, i, \theta}$ and $g_{-1, i, ceiling(\theta)}$ are the same.

[d] false, the number of different decision stumps are $2(R-L) * d + 2$

[e] true, assume all data are different

$2*2*(6-1)+2 = 22$ separtaing in the middle (2*2*(6-1)), all +1 and all -1 (2)

[b][c][e]

---

9.

for each $x_i$ and $x'_i$, there will be $2\|x_i-x'_i\|_1$ different
$g_{s, i, \theta}(x_i) * g_{s, i, \theta}(x'_i) = -1$ for each different s and
$\theta$ (same i), other $2(R-L)-2\|x_i-x'_i\|_1 + 2$ will be
$g_{s, i, \theta}(x_i) * g_{s, i, \theta}(x'_i) = 1$

$g_{+1, i, L-1} == g_{-1, i', R+1}$ for all different i and i'.
same as $g_{-1, i, L-1} == g_{+1, i', R+1}$ for all different i and i'.

so there is a 2 that didn't multiply by d
$K_{ds}(x, x') = d(2(R-L) - 2\|x_i-x'_i\|_1 - 2\|x_i-x'_i\|_1) + 2=$
$2d(R-L) - 4\|x-x'\|_1 + 2$

[c]

---

10.

$u_n^{T+1} = \frac{1}{N} \pi^{T}_{t=1} (
[y_n \neq g_t(x_n)]\sqrt{\frac{1-\epsilon_t}{\epsilon_t}} +
[y_n = g_t(x_n)]\sqrt{\frac{\epsilon_t}{1-\epsilon_t}})$

$ln(u_n^{T+1}) \propto \sum^{T}_{t=1}(
(-1*g_t(x_n)*y_n)*ln(\sqrt{\frac{1-\epsilon_t}{\epsilon_t}}))=
-y_n\sum^{T}_{t=1}(\alpha_t g_t(x_n))$

[d]

---

11.

[a] true, $U^{(1)} = \sum^N_{n=1}\frac{1}{N}$

[b] false, There is counter example in problem 12-15 when t is not large enough.

[c] true, from problem 10, 
$ln(N) + ln(u_n^{T+1}) = -y_n\sum^{T}_{t=1}(\alpha_t g_t(x_n))$

$ln(u_n^{T+1}) \geq -y_n\sum^{T}_{t=1}(\alpha_t g_t(x_n))$

since $ln(u_n^{T+1}) \geq 0$

$U^{T+1} \geq \sum^{N}_{n=1} exp(-y_n\sum^{T}_{t=1}(\alpha_t g_t(x_n)))$

$= \sum^N_{n=1} exp(err) \geq$

$\sum^N_{n=1} [sign(-y_n\sum^{T}_{t=1}(\alpha_t g_t(x_n)))==1 =$

$E_in(G_t) $

<!--$\frac{1}{2} = 
\frac{\sum^N_{n=1}u_{n}^{(t+1)}[y_n \neq
g_t(x_n)]}{\sum^N_{n=1}u_{n}^{(t+1)}}$-->

[d] true, 
$u_n^{T+1} = \frac{1}{N} \pi^{T}_{t=1} (
[y_n \neq g_t(x_n)]\sqrt{\frac{1-\epsilon_t}{\epsilon_t}} +
[y_n = g_t(x_n)]\sqrt{\frac{\epsilon_t}{1-\epsilon_t}})$

$U^{T+1} = \sum^N_{n=1} u_n^{T} (
[y_n \neq g_t(x_n)]\sqrt{\frac{1-\epsilon_T}{\epsilon_T}} +
[y_n = g_t(x_n)]\sqrt{\frac{\epsilon_T}{1-\epsilon_T}})$

$U^{T+1} = \sqrt{\frac{1-\epsilon_T}{\epsilon_T}}
\sum_{n, y_n g_t(x_n)<0} u_n^{T} + \sqrt{\frac{\epsilon_T}{1-\epsilon_T}}
\sum_{n, y_n g_t(x_n)>0} u_n^{T}$

$U^{T+1} = \sum^N_n=1 u_n^{T} \sqrt{\frac{1-\epsilon_T}{\epsilon_T}}
\sum_{n, y_n g_t(x_n)<0} \frac{u_n^{T}}{\sum^N_n=1 u_n^{T}} +
\sqrt{\frac{\epsilon_T}{1-\epsilon_T}}
\sum_{n, y_n g_t(x_n)>0} \frac{u_n^{T}}{\sum^N_n=1 u_n^{T}}$

$U^{T+1} = U^{T} \sqrt{\frac{1-\epsilon_T}{\epsilon_T}} \epsilon_T+
\sqrt{\frac{\epsilon_T}{1-\epsilon_T}} (1-\epsilon_T)$

$\leq U^{T+1} = U^{T} ((1-\epsilon_T+\epsilon_T)/2 + (1-\epsilon_T+\epsilon_T)/2)$
(Arithmetic and Geometric Mean Inequality)

$= U^{T}$

Note T is equal to the notation of t in choice.


[e] false, $\epsilon_t = 1$ scaling factor = 0, $U^{t+1} = 0$

[a][c][d]

---

12.

Ein=0.000000

[a]

---

13.

Eout=0.132000

[c]

---

14.

$U^T=0.005401$

[b]

---

15.

[d]

---

16.

num of branch functions = 10

[c]

---

17.

Ein = 0.000000

[a]

---

18.

Eout = 0.126000

[c]

---

19.

average Eout = 0.075680

[c]

---

20.

[a]

---

21.

False.

If $\epsilon_t == 0.5$, $g_t == g_{t+1}$

---

22.

True.



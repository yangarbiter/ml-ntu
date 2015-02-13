
<head><meta charset="UTF-8"></head>
<script type="text/javascript"
  src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

1.

deterministic noise --> |f(x) - H'(x)|, the part of f that H is unable to
capture. As hypothesis set getting smaller, deterministic noise will increase in
general.

[b]

-----

2.

$H_2 = H(10, 0, 3)$ and $H(10, 0, 3) \subset H(10, 0, 4)$

Therefore $H(10, 0, 3)\cap H(10, 0, 4) = H_2$

[c]

-----

3.

$\bigtriangledown E_{aug}(w) = \bigtriangledown E_{in}(w) +
                                \frac{2\lambda}{N}*w$

gradient decent --> $w(t+1) = w(t) - \eta\bigtriangledown (E_{aug}(w(t)))$

$= w(t) - \eta\bigtriangledown (E_in(w(t))) - \eta\frac{2\lambda}{N}*w(t)$

$= (1-\frac{2\lambda\eta}{N})w(t) - \eta\bigtriangledown (E_{in}(w(t)))$

[a][d]

-----

4.

[b]

For $\lambda \gt 0$

if $\|w_{reg}(\lambda)\| \gt \|w_{lin}\|,$

since $w_{lin}$ is the optimal solution for $E_{in}(w)$

$E_{aug}(w_{lin}) = E_{in}(w_{lin}) + \frac{\lambda}{N}w_{lin}^Tw_{lin}
\lt E{in}(w_{reg}(\lambda)) + \frac{\lambda}{N}w_{reg}(\lambda)^Tw_{reg}(\lambda) =
E_{aug}(w_{reg}(\lambda)$

This contradicts the fact that $w_{reg}(\lambda)$ is the optimal solution for
$E_{aug}(w)$. Therefore $\|w_{reg}(\lambda)\| \leq \|w_{lin}\|$

Since $w^Tw \geq 0, E_{aug}(w) \geq E_{in}$ for $\lambda \gt 0$

[a] Proven in [b] that $\|w_{reg}(\lambda)\| \gt \|w_{lin}\|$ can't be true.

[c]

For $\lambda \gt 0$

Assume $\|w_{reg}(\lambda)\|$ is a increasing function of $\lambda$. That is,
$\|w_{reg}(\lambda)\| \lt \|w_{reg}(\lambda+1)\|$


[d] Proven not true in [c]

[e] $\|w_{reg}(\lambda)\|$ won't always = $\|w_{in}\|$

[b][c]

-----

5.

$A(-1, 0), B(\rho, 1), C(1, 0)$

|train|$b_0$|$a_1$             |$b_1$              |ValErr $h_0$|ValErr $h_1$|
|-----|-----|------------------|-------------------|------------|------------|
|AB   |0.5  |$\frac{1}{\rho+1}$|$\frac{1}{\rho+1}$ |0.25        |$(\frac{2}{\rho+1})^2$|
|BC   |0.5  |$\frac{1}{\rho-1}$|$\frac{-1}{\rho-1}$|0.25        |$(\frac{-2}{\rho-1})^2$|
|AC   |0    |0                 |0                  |1           |1|

$(\frac{-2}{\rho-1})^2 + (\frac{2}{\rho+1})^2 = 0.5$

$\rho^4 - 18\rho^2 - 15 = 0$

$\rho^2 = 9 \pm 4\sqrt{6}$

$\rho = \sqrt{9+4\sqrt{6}}$

[c]

-----

6.

[a] True, $2^5 = 32$

[b] False, at least 32 people.

[c] True, 32 / 2 = 16

[d] False, 32 + 16 + 8 + 4 = 60 are sent.

[e] False.

[a][c]

-----

7.

32 + 16 + 8 + 4 + 2 = 62

6th game --> 1 mail

He earned 1000 - 63 * 10 = 370

[b]

-----

8.

M=1 since g is derived mathematiclly without looking at the data.

[a]

-----

9.

hoeffding bound --> $P[|E_{in}(g)-E_{out}(g)| \gt \epsilon] \leq
                        2 * M * exp(-2 \epsilon^2N)$

N=10000, M=1, $\epsilon=0.01$

$2*M*exp(-2\epsilon^2N) \approx 0.27$

[c]

-----

10.

[a] If a(x) = +1, in sample data would than came from the same distribution as
the out of sample data. So this shouldn't happen when a(x)=+1.

[c] Because E_in is test on those which a(x) approved already, a(x) AND g(x) should
be applied to approve credit for new customers.

[a][c]

-----

11.

$\sum^{N}_{n-1}((y_n - w^Tx_n)^2 \rightarrow (y-Xw)^2$

$\bigtriangledown (y-Xw)^2 = -X^T(y-Xw)$

gradient of error function: $-X^T(y-Xw) - \tilde{X}^T(\tilde{y}-\tilde{X}w) =
-X^Ty - \tilde{X}^T\tilde{y} + w(X^TX + \tilde{X}^T\tilde{X}) = 0$

$w = (X^TX + \tilde{X}^T\tilde{X})^{-1}(X^Ty + \tilde{X}^T\tilde{y})$

[d]

-----

12.

when $\tilde{X}=\sqrt{\lambda}I, \tilde{y}=0$ minimizing,

$\frac{1}{N+K}
(\sum^{N}_{n=1}(y_n - w^Tx_n)^2 + \sum^{K}_{k=1}(\tilde{y_k} - w^T\tilde{x_k})^2) =
\frac{1}{N+K}
(\sum^{N}_{n=1}(y_n - w^Tx_n)^2 + \sum^{K}_{k=1}(\sqrt{\lambda}w_k^2) =
\frac{1}{N+K}
((y-Xw)^2 + \lambda\|w\|^2)$

will be equivalent to minimizing $\frac{\lambda}{N}\|w\|^2 + \frac{1}{N}(Xw-y)^2$

[b]

-----

13.

Ein: 0.05, Eout: 0.045

[d]

-----

14.

log Lambda: 2.0 Ein: 0.24 Eout: 0.261 <br>
log Lambda: 1.0 Ein: 0.05 Eout: 0.045 <br>
log Lambda: 0.0 Ein: 0.035 Eout: 0.02 <br>
log Lambda: -1.0 Ein: 0.035 Eout: 0.016 <br>
log Lambda: -2.0 Ein: 0.03 Eout: 0.016 <br>
log Lambda: -3.0 Ein: 0.03 Eout: 0.016 <br>
log Lambda: -4.0 Ein: 0.03 Eout: 0.016 <br>
log Lambda: -5.0 Ein: 0.03 Eout: 0.016 <br>
log Lambda: -6.0 Ein: 0.035 Eout: 0.016 <br>
<b>log Lambda: -7.0 Ein: 0.03 Eout: 0.015</b> <br>
<b>log Lambda: -8.0 Ein: 0.015 Eout: 0.02</b> <br>
log Lambda: -9.0 Ein: 0.015 Eout: 0.02 <br>
log Lambda: -10.0 Ein: 0.015 Eout: 0.02 <br>

[d]

-----

15.

[d]

-----

16.

log Lambda: -8.0 Etrain: 0.0 Eval: 0.05 Eout: 0.025

[e]

-----

17.

log Lambda: 0.0 Etrain: 0.0333333333333 Eval: 0.0375 Eout: 0.028

[a]

-----

18.

log Lambda: 0.0 Ein: 0.035 Eout: 0.02

[c]

-----

19.

log Lambda: -8.0 Ecv: 0.03

[e]

-----

20.

Ein: 0.015 Eout: 0.02

[b]

-----

21.

$\Gamma$ is a multiple of identity matrix ($\alpha I$).

Usual cost function for Tikhonov regularization:
$\frac{\lambda}{N}\|\Gamma w\|^2 + \frac{1}{N}(Xw-y)^2 =
\frac{\lambda}{N}\|\alpha w\|^2 + \frac{1}{N}(Xw-y)^2$

Ans: $\tilde{X}=\sqrt{\lambda}\Gamma, \tilde{y}=0$ minimizing, $\Gamma=\alpha I$

$\frac{1}{N+K}
(\sum^{N}_{n=1}(y_n - w^Tx_n)^2 + \sum^{K}_{k=1}(\tilde{y_k} - w^T\tilde{x_k})^2)=
\frac{1}{N+K}
(\sum^{N}_{n=1}(y_n - w^Tx_n)^2 + \sum^{K}_{k=1}(\alpha\sqrt{\lambda}w_k)^2)$

$=\frac{1}{N+K}
((y-Xw)^2 + \lambda\|\Gamma w\|^2)$

will be equivalent to minimizing
$\frac{\lambda}{N}\|\Gamma w\|^2 + \frac{1}{N}(Xw-y)^2$

-----

22.

Ans: $\tilde{X}=\sqrt{\lambda}I, \tilde{y}=\sqrt{\lambda}h_{hint}$

Minimizing,

$\frac{1}{N+K}
(\sum^{N}_{n=1}(y_n - w^Tx_n)^2 + \sum^{K}_{k=1}(\tilde{y_k} - w^T\tilde{x_k})^2)$

$=\frac{1}{N+K}
(\sum^{N}_{n=1}(y_n - w^Tx_n)^2 +
 \sum^{K}_{k=1}(\sqrt{\lambda}w_{hint_k} - w^T\sqrt{\lambda}I_k)^2)$

$=\frac{1}{N+K}
(\sum^{N}_{n=1}(y_n - w^Tx_n)^2 + \lambda\|w-w_{hint}\|^2)$

will be equivalent to minimizing
$\frac{\lambda}{N}\|w-w_{hint}\|^2 + \frac{1}{N}(Xw-y)^2$

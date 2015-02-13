
<head><meta charset="UTF-8"></head>
<script type="text/javascript"
  src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

1. 

$\sigma^2(1-\frac{d+1}{N}) \lt 0.008$

$N \gt 45$

[c]

-----

2. 

$H = X(X^TX)^{-1}X^T$

i. $Rank(AB) \leq min(Rank(A), Rank(B))$, therefore H's rank is d+1.
(Rank(X)=d+1 because $X^TX$ is invertible)

ii. $HX = X*(X^TX)^{-1}*(X^TX) = X = IX$


[a]

H is a square matrix, it can be factorized into $Q\Lambda Q^{-1}$
(eigendecomposition), becauese its eigenvalues are all 1s, $\bar{H^T} = H$
Therefore it is a positive semi-definite matrix.

[b]

When N != d+1, --> rank = d+1 != N, H is invertable.

[c]

eigenvalues are all 1s.

[d]

Due to i. and ii. H has d+1 eigenvalues=1.

[e]

eigenvalues are all 1s, $H^{1126}=Q\Lambda^{1126}Q^{-1}=Q\Lambda Q^{-1}=H$

[a][d][e]

-----

3. 
    a. 
        if $y = sign(w^Tx), max(0, 1-yw^Tx) \geq [sign(w^Tx) \neq y] = 0$

        if $y \neq sign(w^Tx), yw^Tx \leq 0$

        $\therefore max(0, 1-yw^Tx) \geq [sign(w^Tx) \neq y] = 1$

    b. 
        if $y = sign(w^Tx), max(0, 1-yw^Tx)^2 \geq [sign(w^Tx) \neq y] = 0$

        if $y \neq sign(w^Tx), \because yw^Tx \leq 0$

        $\therefore max(0, 1-yw^Tx)^2 \geq [sign(w^Tx) \neq y] = 1$

    c. 
        if $y = sign(w^Tx), max(0, -yw^Tx) = 0$ and $[sign(w^Tx) \neq y] = 0$ 

        if $y \neq sign(w^Tx), max(0, -yw^Tx) = -yw^Tx$ not a upper bound of 1

    d. 
        if $y = sign(w^Tx), \theta(-yw^Tx) \geq [sign(w^Tx) \neq y] = 0$

        if $y \neq sign(w^Tx), \theta(-yw^Tx) \leq [sign(w^Tx) \neq y] = 1$

    e. 
        if $y = sign(w^Tx), exp(-yw^Tx) \geq [sign(w^Tx) \neq y] = 0$

        if $y \neq sign(w^Tx), exp(-yw^Tx) \geq [sign(w^Tx) \neq y] = 1$

[a][b][e]

-----

4. 

    [a][c] not differentiable when $yw^Tx = 0$.

    [b] $err(w) = (max(0, -yw^Tx))^2$

    for y = -1, limit from the left to $w^Tx = 0$ $err(w) = 0$, 
            limit from the right to $w^Tx = 0$ $err(w) = (w^Tx)^2 = 0$

    for y = 1, limit from the right to $w^Tx = 0$ $err(w) = 0$, 
            limit from the left to $w^Tx = 0$ $err(w) = (-w^Tx)^2 = 0$

    err(w) is differentiable anywhere for w in this case.

    [d][e] $\Theta$ and exp are continous functions of w.

[b][d][e]
        
-----

5. 

for $err(w) = max(0, -yw^Tx)$

if $-yw^Tx \geq 0$ (sign(w^Tx) != sign(y), wrong label),
 $\frac{\partial err(w)}{\partial w} = -yx$

else $\frac{\partial err(w)}{\partial w} = 0$ (ignores the points that are not
differentiable.

for perceptron: if $sign(w_t^Tx_n) \neq y_n, w_{t+1} = w_t + y_nx_n
= w_t - \frac{\partial err(w)}{\partial w}$

and when the label is right, $w_t$ won't change.

[c]

-----

6. 

gradient $e^u+e^{2v}+e^{uv}+u^2-2uv+2v^2-3u-2v$, $(u, v) = (0, 0)$

$\bigtriangledown E(u, v) = (e^u+ve^{uv}+2u-2v-3, 2e^{2v}+ue^{uv}-2u+4v-2) = (-2
, 0)$

[d]

-----

7. 

```python 
import math
eta = 0.01
gradient = lambda u, v: (math.exp(u)+v*math.exp(u*v)+2*u-2*v-3, 2*math.exp(2*v)+u*math.exp(u*v)-2*u+4*v-2)
E = lambda u, v: math.exp(u)+math.exp(2*v)+math.exp(u*v)+u**2-2*u*v+2*v**2-3*u-2*v
a=b=0
for i in xrange(5):
    g = gradient(a, b)
    a -= eta*g[0]
    b -= eta*g[1]
    print(E(a, b))

print(E(a, b))
```

$(u_5, v_5) = (0.09413996302028127, 0.0017891105951028273)$

$E(u_5, v_5) = 2.8250003566832635$

[c]

-----

8. 

$b=E(0,0)=3$

$b_v=\frac{\partial E}{\partial v}(0, 0)=-2$

$b_u=\frac{\partial E}{\partial u}(0, 0)=0$

$b_{uv}=\frac{1}{2!}(\frac{\partial E}{\partial u \partial v}(0, 0) +
                     \frac{\partial E}{\partial v \partial u}(0, 0))=-1$

$b_{vv}=\frac{1}{2!}\frac{\partial E}{\partial v \partial v}(0, 0) = 4$

$b_{uu}=\frac{1}{2!}\frac{\partial E}{\partial u \partial u}(0, 0) = 1.5$

[b]

-----

9. 

$f_{xy} = \frac{\partial E}{\partial x \partial y}$

gradient of quadratic Taylor expansion = 
$((\Delta u) f_{uu}(u, v) + (\Delta v) f_{uv}(u, v) + f_{u}(u, v) +
 (\Delta v) f_{vv}(u, v) + (\Delta u) f_{uv}(u, v) + f_{v}(u, v)) = 0$

$(\Delta u, \Delta v)*\nabla^2E(u, v) + \nabla E(u, v)= 0$

$(\Delta u, \Delta v) = - (\nabla^2E(u, v))^{-1} * \nabla E(u, v)$

[a]

-----

10. 

$\hat{E_2}(\Delta u, (\Delta v)) = 1.5*(\Delta u)^2 + 4*(\Delta v)^2 - 
(\Delta u)(\Delta v) -2 (\Delta u) + 3$

```python
from sympy import *
import math

e = math.e
u, v = symbols('u v')
E = e**u + e**(2*v) + e**(u*v) + u**2 - 2*u*v + 2*v**2 - 3*u - 2*v
para = [0, 0]
for i in xrange(5):
    subs = {u:para[0], v:para[1]}
    update = -1 * Matrix([[diff(E,u), diff(E, v)]]) * hessian(E, [u, v]).inv() 
    update = update.evalf(subs=subs)
    para[0] += update[0]
    para[1] += update[1]
print(E.evalf(subs={u:para[0], v:para[1]}))
```
$E(u_5, v_5) = 2.36082334564314$

[c]

minimize $\hat{E_2}$ --> $\nabla$ of $\hat{E_2}$ = 0

$2*b_{vv}(\Delta v) + b_{uv}(\Delta u) + b_u = 4(\Delta v) + -(\Delta u) - 2 = 0$

$2*b_{uu}(\Delta u) + b_{uv}(\Delta v) + b_v = 4(\Delta u) + -(\Delta v) = 0$

$(\Delta u) = 0.66$

$(\Delta v) = 0.66$

-----

11.

Because we could use the union of quadratic, linear, or constant hypotheses.
Project all points to $(1, x_1, x_2, x_1^2, x_1x_2, x_2^2)$

VC dimension of PLA on 6 dimensional space is = 7, which we could shatter any
6 point input with PLA.

[e]

-----

12.

In the $n^{th}$ dimension of $\Phi$ it represents whether it is the $n^{th}$ x
or not.

In N dimensional space, linear classification can shatter any N+1 points.
Therfore $d_{vc}(H_{\phi}) = \inf$

[e]

-----

13.

Average in sample 01Error: 0.503979

[c]

-----

14.
```
error rate for each of the choices:
a 0.113
b 0.234
c 0.221
d 0.477
e 0.316
```

[a]

-----

15.

Average out sample 01Error: 0.126316

[a]

-----

16.

$p(y|x) = h_y(x) = \frac{exp(w_y^Tx)}{\sum^K_{i=1}exp(w_i^Tx)}$

$ln(p(y|x) = w_y^Tx - ln(\sum^K_{i=1} exp(w_i^Tx))$

likelihood(h) = $ p(y_1)p(y_1|x_1) * p(y_2)p(y_2|x_2) ... * p(y_N)p(y_N|x_N)$
              $\propto  p(y_1|x_1) * p(y_2|x_2) ... * p(y_N|x_N)$

negative log likelihood $\propto -(ln(p(y_1|x_1))-ln(p(y_2|x_2)) ... +ln(p(y_N|x_N)))$
                        $= -\sum^N_{i=1}(w_{y_i}^Tx_i) -
                             \sum^N_{i=1}(ln(\sum^K_{j=1}exp(w_j^Tx_i)))$
                        $\propto \frac{1}{N} * -\sum^N_{i=1}(w_{y_i}^Tx_i -
                                 ln(\sum^K_{j=1}exp(w_j^Tx)))$

[d]

-----

17.

$\frac{\partial E_{in}}{\partial w_i} =
    \sum^N_{n=1}( x_n*\frac{e^{w_i^Tx_n}}{\sum^K_{j=1} e^{w^T_jx_n}}
                     - [y_n==i]x_n)$

[c]

-----

18.

zero-one error on test set: 0.475000(eta=0.001)

[a]

-----

19.

zero-one error on test set: 0.220000(eta=0.01)

[d]

-----

20.

zero-one error on test set: 0.477000(pick example in cyclic order)

[a]

-----

21.

It needs N+1 queries.

Define $h_0(x) = 0$ and $h_i(x) = [x == x_i]$, i=1 to N

we can get N+1 equations in N+1 queries:

$\sqrt{\frac{1}{N}((y_1)^2 + (y_2)^2 ... (y_N)^2)} = RMSE(h_1)$

$\sqrt{\frac{1}{N}((y_1 - 1)^2 + (y_2)^2 ... (y_N)^2)} = RMSE(h_1)$

$\sqrt{\frac{1}{N}((y_1)^2 + (y_2 - 1)^2 ... (y_N)^2)} = RMSE(h_2)$

...

$\sqrt{\frac{1}{N}((y_1)^2 + (y_2)^2 ... (y_N - 1)^2)} = RMSE(h_N)$

we can get a solutions of y from these equations, and generating g by letting 
$g(x_1)=y_1$, $g(x_2)=y_2$, ... $g(x_N)=y_N$.

-----

22.

It needs 2 queries.

By querying QMSE($h_0$), where $h_0$ is a constant function always output 0. We
can get $\sqrt{\frac{1}{N}\sum^N_{n=1}y_n^2} = QMSE(h_0)$

$\sqrt{\frac{1}{N}\sum^N_{n=1}(y_n-h(x_n))^2} =
 \sqrt{\frac{1}{N}\sum^N_{n=1}(y_n^2-2*y_n*h(x_n)+h(x_n)^2)} = QMSE(h)$, h is
 the given hypothesis.

$N*(QMSE(h)^2 - QMSE(h_0)^2) = \sum^N_{n=1}(2*y_n*h(x_n)+h(x_n)^2)
                            = 2*\sum^N_{n=1}(y_n*h(x_n)) +
                                 \sum^N_{n=1}(h(x_n)^2)$

$\sum^N_{n=1}(h(x_n)^2)$ can be calculated without query, therefore we can
obtain $\sum^N_{n=1}(y_n*h(x_n)) = h^Ty$ with two queries.

-----

23.

It needs K+1 queries.

To minimise, 
$\frac{\partial RMSE(H)}{\partial w_i} =
 \frac{1}{RMSE(H)} * \frac{1}{N} * \sum^N_{n=1}(2*(y_n - H(x_n))(-h_i(x_n))) =
 0$

$\sum^N_{n=1}(y_n - H(x_n)) \propto h_i*\sum^N_{n=1}y_n - \sum^N_{n=1}H(x_n)$
$h_i*\sum^N_{n=1}y_n$ can be obtain in 2 queries.

We will need total K+1 queries(QMSE($h_1$) to QMSE($h_K$) and QMSE($h_0$) where
$h_0$ is a constant function = 0).

Then there will be K queations to solve $w_1, w_2, ..., w_n$.


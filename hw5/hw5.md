
<head><meta charset="UTF-8"></head>
<script type="text/javascript"
  src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

1.

minimizing $\frac{1}{2}\sum^N_{n=1}\sum^N_{m=1}\alpha_n\alpha_m y_n y_m z^T_n
z^T_m - \sum^N_{n=1}\alpha_n$ with some constraints.

We want to find optimal [b w $\xi$] which has 1+d+N variables

[b]

-----

2.

$

x_1 = (1, 0),  z_1 = (1, -2)

x_2 = (0, 1),  z_2 = (4, -5)

x_3 = (0, -1), z_3 = (4, -1)

x_4 = (-1, 0), z_4 = (5, -2)

x_5 = (0, 2),  z_5 = (7, -7)

x_6 = (0, -2), z_6 = (7, 1)

x_7 = (-2, 0), z_7 = (7, 1)

z_1 = 4.5 is the optimal separting "hyperplane" in Z space

[c]

-----

3.

polynomial kernel with gamma = 1, coef0 = 1 and degree = 2 and a very large C

```python
from sklearn import svm
X = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
y = [-1, -1, -1, 1, 1, 1, 1]

clf = svm.SVC(C=999999.0, kernel='poly', coef0=1, degree=2, gamma=1)
clf.fit(X, y)
print clf.support_vectors_
print clf.dual_coef_
```

support vectors: [(0, 1), (0, -1), (-1, 0), (0, 2), (0, -2)]

nonzero $\alpha$: [0.59647182, 0.81065085, 0.8887034, 0.20566488, 0.31275439]

5 support vectors --> 5 nonzero $\alpha$

$\sum^7_{n=1} \alpha_n \approx 2.8148$

$max_{1\leq n \leq 7} \alpha_n = \alpha_4$

$min_{1\leq n \leq 7} \alpha_n = \alpha_7$ or $\alpha_1 = 0$
(non-support vector)

[b][d]

-----

4.

$\sum^5_{i=1} \alpha_n y_n K(x_n, x)$

```python
m = svmutil.svm_train(y, X, '-s 0 -t 1 -d 2 -g 1 -r 1 -c 99999'
m.rho[0]
```
$b = -1.66633087511 \approx \frac{-15}{9}$

```python
yk = []
for x, ya in zip(clf.support_vectors_, clf.dual_coef_[0]):
    ayk.append([ya * x[1]**2, ya * x[0]**2, ya * 2 * x[1], ya * 2 * x[0], ya * 1])
#x2^2, x1^2, x2, x1, 1
print 9 * np.sum(np.array(ayk), axis=0)
[ -5.99898975e+00  -7.99833061e+00  -1.99840144e-15   1.59966612e+01
   4.99600361e-16]
```
$\frac{1}{9}(-6 * x^2_2 + -8 * x^2_1 + 16 * x_2 + (0 + -15)) = 0$

[b]

-----

5.

$z_1 = 4.5 & $\frac{1}{9}(-6 * x^2_2 + -8 * x^2_1 + 16 * x_2 + (0 + -15)) = 0$
are different because they are learned with respect to different Z space. (They
have the same raw data)

[c]

-----

6.

$R^2 + \sum^N_{n=1} \lambda_n(\|x_n - c\|^2 - R^2)$

because $\|x_n - c\|^2 \leq R^2$, $\|x_n - c\|^2 - R^2 \leq 0$

Therefore if any R break the constraint $\|x_n - c\|^2 \leq R^2$, $\lambda \geq
0, L(R, c, \lambda) \to \infty$. This way it will not be able to minimize
$\infty$.

And $\|x_n - c\|^2 - R^2 \leq 0$ if $\|x_n - c\|^2 - R^2 \neq 0$, \lambda_n = 0,
so max L(R, c, \lambda) = 0, will make
$R^2 + \sum^N_{n=1} \lambda_n(\|x_n - c\|^2 - R^2)$ equuivalent to original
problem.

[d]

-----

7.

KKT condition

Dual feasible: $\lambda_n \geq 0$

Primal feasible: $\|x_n - c\|^2 \leq R^2$

Complementary slackness: $\lambda_n(\|x_n - c\|^2 - R^2) = 0$

Stationarity:

$\frac{\partial\Lambda(R, c, \lambda)}{\partial R} = 0 = 2R -
2R\sum^N_{n=1}\lambda_n \rightarrow \sum^N_{n=1}\lambda_n = 1$

$\frac{\partial\Lambda(R, c, \lambda)}{\partial c} = 0 =
\sum^N_{n=1}\lambda_n(2c - 2x_n) \rightarrow c\sum^N_{n=1}\lambda_n =
\sum^N_{n=1}\lambda_n x_n$

[a][c][d]

-----

8.

$R^2 + \sum^N_{n=1} \lambda_n(\|x_n - c\|^2 - R^2) = $

$R^2 + \sum^N_{n=1} \lambda_n\|x_n - c\|^2 - \sum^N_{n=1} \lambda_n R^2 = $
(because $\sum^N_{n=1}\lambda_n=1$)

$R^2 + \sum^N_{n=1} \lambda_n\|x_n - c\|^2 - R^2 = $

$\sum^N_{n=1} \lambda_n (x_n^2 - 2cx_n + c^2) = $
(because $c = \frac{\sum^N_{n=1}\lambda_n x_n}{\sum^N_{n=1}\lambda_n}$)

$\sum^N_{n=1} \lambda_n (x_n^2 -
2x_n\frac{(\sum^N_{m=1}\lambda_m x_m)}{(\sum^N_{m=1}\lambda_m)}
+ \frac{(\sum^N_{m=1}\lambda_m x_m)^2}{(\sum^N_{m=1}\lambda_m)^2}) = $
(because $\sum^N_{n=1}\lambda_n=1$)

$\sum^N_{n=1} \lambda_n (x_n^2 -
2x_n(\sum^N_{m=1}\lambda_m x_m) + (\sum^N_{m=1}\lambda_m x_m)^2) = $

$\sum^N_{n=1} \lambda_n \|x_n - \sum^N_{m=1}\lambda_m x_m\|^2$

(The dual problem defined in the problem seems to mistype max as min)

[a]

-----

9.

$\sum^N_{n=1} \lambda_n \|x_n^2 - \sum^N_{m=1}\lambda_m x_m\|^2 = $

$\sum^N_{n=1} \lambda_n (x_n^2 -
2x_n(\sum^N_{m=1}\lambda_m x_m) + (\sum^N_{m=1}\lambda_m x_m)^2) = $

$\sum^n_{n=1} \lambda_n k(x_n, x_n) -
2\sum^n_{n=1} \sum^n_{m=1} \lambda_n \lambda_m k(x_n, x_m) +
\sum^n_{n=1} \sum^n_{m=1} \lambda_n \lambda_m k(x_n, x_m) = $

$\sum^N_{n=1} \lambda_n K(x_n, x_n) -
\sum^N_{n=1} \sum^N_{m=1} \lambda_n \lambda_m K(x_n, x_m)$

[c]

-----

10.

if $\lambda_i \gt 0$, by complementary slackness of KKT condition, $(\|x_i - c\|^2
    - R^2) = 0$

$R = \|x_i - c\| = \|x_i -
\frac{\sum^N_{n=1}\lambda_n x_n}{\sum^N_{n=1}\lambda_n}\| = $
(because $c = \frac{\sum^N_{n=1}\lambda_n x_n}{\sum^N_{n=1}\lambda_n}$)

$R = \|x_i - c\| = \sqrt{\|x_i - \sum^N_{n=1}\lambda_n x_n\|^2} = $

$\sqrt{x_i^2 - 2x_i\sum^N_{n=1}\lambda_n x_n + (\sum^N_{n=1}\lambda_n x_n)^2} = $

$\sqrt{K(x_i, x_i) - 2\sum^N_{n=1}\lambda_n K(x_i, x_n) +
 (\sum^N_{n=1}\sum^N_{m=1}\lambda_n\lambda_m K(x_n, x_m)}$

[a]

-----

11.

let $\tilde{w} = [w, y_n * \sqrt{2C} * \xi_n]$, $y_n \in {1, -1}$

$\tilde{x_n} = [x_n, v_1, v_2, ..., v_N]$, where $v_i =
\frac{1}{\sqrt{2C}}[i=n]$

objective function:

$\frac{1}{2}\tilde{w}^T\tilde{w} = $
$\frac{1}{2}w^Tw + \frac{1}{2} 2C y_n^2\sum^N_{n=1}\xi_n^2$

constraint:

$y_n(\tilde{w}^T\tilde{x_n} + b) \geq 1 \Rightarrow
 y_n(w^Tx_n + y_n * \xi_n + b) \geq 1 \Rightarrow
 y_n(w^Tx_n + b) \geq 1 - \xi_n$

[a]

-----

12.

$K_1$ and $K_2$ are positive semi-definite.

That is, $x^TK_1x \geq 0$ and $x^TK_2x \geq 0 \forall x \in R^d$

[a] $x^T(K_1+K_2)x = x^TK_1x + x^TK_2x \geq 0 \forall x \in R^d$

K(x, x') is positive semi-definite, it is a valid kernel.

[b] Counter example: $K_1$ = ones(5)+100*eye(5) and $K_2$ = ones(5)+1000*eye(5)

$K_1$ and $K_2$ is valid kernel.

```matlab
>>  eig((ones(5)+100*eye(5))-(ones(5)+1000*eye(5)))

ans =

  -900
  -900
  -900
  -900
  -900
```

$K_1 - K_2$ is not positive semi-definite matrix, thus it is not a valid kernel.

[c] $K_1 \circ K_2$

$K_i$ --> ith column of matrix K.

$K_1_i$ --> ith column of matrix $K_1$.

$x_i$ --> ith element of column vector $x$.

$x^T(K_1 \circ K_2)x = x^TKx$

$\sum^N_{i=1}x^T * K_i * x_i = $

$\sum^N_{i=1}x^T * (K_1_i \circ K_2_i) * x_i = $

$\sum^N_{i=1}x^T * K_1_i * x_i + \sum^N_{i=1}x^T * K_2_i * x_i = $

$x^TK_1x + x^TK_2x \geq 0 $ (because $K_1$, $K_2$ is valid kernel.)

Therefore positive semi-definite is close under Hadamard product.

Schur product theorem

[d]

counter example: $K_1$ = ones(5)+1000*eye(5) and $K_2$ = ones(5)+10000*eye(5)

$K_1$ and $K_2$ are positive semi-definite because their eigenvalue >= 0.

```matlab
>> eig((ones(5)+1000*eye(5))./(ones(5)+10000*eye(5)))

ans =

   -0.8999
   -0.8999
   -0.8999
   -0.8999
    4.1001
```

[a][c]

-----

13.

[a]

counter example: $K_1 = I_2$

```matlab
>> eig(eye(2))

ans =

    1
    1
```

```matlab
>> eig((1-eye(2)).^2)

ans =

    -1
    1
```

this way $(1-K_1).^2$ is not positive semi-definite matrix.

[b]

$K_1$ is positive semi-definite.

That is, $x^TK_1x \geq 0 \forall x \in R^d$

$x^T1126K_1x \geq 0$

1126K(x, x') is positive semi-definite, it is a valid kernel.

[c]

counter example: $K_1 = I_2$

eig(1*eye(2)) = [1 1]

eig(exp(-1*eye(2))) = [-0.6321 1.3679]

[d]

$(1-K_1)^{-1} = 1 + K_1 + K_1^2 + k_1^3 ...$

form problem 12 positive semi-definite matrix is close under addition and
multiplication. $(1-K_1).^{-1}$ is positive semi-definite matrix, so it is valid
kernel.

[b][d]

-----

14.

The curve should be same: $wz + b = \tilde{w}z + \tilde{b}$

N = number of SV, $z_s$ a unbounded SV.

$\tilde{w}z + \tilde{b} = $

$\sum^N_{n=1}\tilde{\alpha_n} y_n \tilde{K}(z_n, z) + y_s -
\sum^N_{n=1}\tilde{\alpha_n} y_n \tilde{K}(z_n, z_s) = $

$\sum^N_{n=1}\tilde{\alpha_n} y_n (pK(z_n, z) + q) + y_s -
\sum^N_{n=1}\tilde{\alpha_n} y_n (pK(z_n, z_s) + q) = $

$p\sum^N_{n=1}\tilde{\alpha_n} y_n K(z_n, z) +
 p\sum^N_{n=1}\tilde{\alpha_n} y_n K(z_n, z_s) +
 2q\sum^N_{n=1}\tilde{\alpha_n} y_n + y_s = $
(KKT condition --> $\sum^N_{n=1}\tilde{\alpha_n} y_n = 0$)

$\sum^N_{n=1}\alpha_n y_n K(z_n, z) +
 \sum^N_{n=1}\alpha_n y_n K(z_n, z_s) + y_s = $

$wz + b$

$\tilde{\alpha} = \frac{1}{p}\alpha$

Therefore, $\tilde{C} = \frac{1}{p}C$, ($\alpha$ will be scaled by C).

[c]

<!--$w^Tw = \sum^N_{n=1}\sum^N_{m=1}\alpha_n\alpha_m y_n y_m \tilde{K}(x_n, x_m) =$

$p\sum^N_{n=1}\sum^N_{m=1}\alpha_n\alpha_m y_n y_m K(x_n, x_m) +
 q\sum^N_{n=1}\sum^N_{m=1}\alpha_n\alpha_m y_n y_m =$
(with KTT condition $\sum^N_{i=1}\alpha_i y_i = 0$)

$p\sum^N_{n=1}\sum^N_{m=1}\alpha_n\alpha_m y_n y_m K(x_n, x_m) =$

$\xi_n = 1 - y_n\sum^N_{i=1}\alpha_i y_i \tilde{K}(x_i, x_n)$

$C\sum^N_{n=1}\xi_n = $

$C\sum^N_{n=1}(1 - y_n(\sum^N_{i=1}\alpha_i y_i \tilde{K}(x_i, x_n) + b)) = $

$C\sum^N_{n=1}(1 - y_n\sum^N_{i=1}\alpha_i y_i (pK(x_i, x_n)+q+b)) = $

$C\sum^N_{n=1}(1 - py_n\sum^N_{i=1}\alpha_i y_iK(x_i, x_n) -
 (q+b)\sum^N_{i=1}\alpha_i y_i)) = $

$pC\sum^N_{n=1}(1 - y_n\sum^N_{i=1}\alpha_i y_iK(x_i, x_n)) +
 C\sum^N_{n=1}(1 - p - (q+b)\sum^N_{i=1}\alpha_i y_i) = $
(with KTT condition $\sum^N_{i=1}\alpha_i y_i = 0$)

$pC\sum^N_{n=1}(1 - y_n\sum^N_{i=1}\alpha_i y_iK(x_i, x_n)) + Constant$

$\tilde{C}$ has to equal to $C$ to get the same optimization result.-->

-----

15.

|w| = 0.571317

[b]

-----

16.

0 v.s. not 0, E_in = 11.160937%

2 v.s. not 2, E_in = 9.865471%

4 v.s. not 4, E_in = 9.965122%

6 v.s. not 6, E_in = 8.470354%

8 v.s. not 8, E_in = 8.271051%

[e]

-----

17.

0 v.s. not 0, sigma_alpha = 21.780000

2 v.s. not 2, sigma_alpha = 14.620000

4 v.s. not 4, sigma_alpha = 13.040000

6 v.s. not 6, sigma_alpha = 13.280000

8 v.s. not 8, sigma_alpha = 10.840000

[d]

-----

18.

Margin = [8.237752205485867, 0.82468820878303284, 0.12880771012515943,
0.084148113805552607, 0.041091229302772682]

Sigma_xi = [2353.4103603096287, 2044.4950499456361, 1361.7852805909249,
1274.5037745010231, 1231.2604538018772]

number of support vectors = [2398, 2520, 2285, 1773, 1676]

Eout =  [17.88739412057798, 17.88739412057798, 10.513203786746388,
10.36372695565521, 10.463378176382662]

objective value = [-2.380630, -23.145040, -178.198694, -1401.259369,
 -13027.300913]

 [a][b][e]

-----

19.

gamma=1 Eout=10.712506%

gamma=10 Eout=9.915296%

gamma=100 Eout=10.513204%

gamma=1000 Eout=17.887394%

gamma=10000 Eout=17.887394%

[b]

-----

20.

gamma=1000:0,  gamma=1:11,  gamma=10:56,  gamma=100:33,  gamma=10000:0

[b]

-----

21.

False.

All these support vectors could violate the margin a little but didn't cross the
decision boundary.

-----

22.

False.

If the SVM have a small C, this could happen.




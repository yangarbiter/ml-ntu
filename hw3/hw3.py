
import sympy as sym
import math

e = sym.E
u, v = sym.symbols('u v')
E = e**u + e**(2*v) + e**(u*v) + u**2 - 2*u*v + 2*v**2 - 3*u - 2*v
para = [0, 0]
for i in xrange(5):
    print para
    subs = {u:para[0], v:para[1]}
    update = -1 * sym.Matrix([[sym.diff(E,u), sym.diff(E, v)]]) * sym.hessian(E, [u, v]).inv() 
    print sym.simplify(update)
    update = sym.simplify(update)
    update = update.evalf(subs=subs)
    para[0] += update[0]
    para[1] += update[1]
print(E.evalf(subs={u:para[0], v:para[1]}))

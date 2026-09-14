import numpy as np
from audit import solve,loss,derivative,ap
p=np.array([.9,.1]);l=np.array([.1,.9]);y=np.array([1.,0.])
assert solve(p,l,y)[0]==0
assert solve(l,p,y)[0]==1
p=np.array([.9,.9]);l=np.array([.1,.1]);y=np.array([1.,0.])
g,*_=solve(p,l,y);assert abs(g-.5)<1e-8
h=1e-6;fd=(loss(p,l,y,.3+h)['bce']-loss(p,l,y,.3-h)['bce'])/(2*h)
assert abs(fd-derivative(p,l,y,.3))<1e-7
assert ap(np.array([0,1]),np.array([.1,.9]))==100
assert ap(np.zeros(2),np.ones(2))==0
print('PASS: endpoints, interior convex optimum, finite-difference gradient, AP support')

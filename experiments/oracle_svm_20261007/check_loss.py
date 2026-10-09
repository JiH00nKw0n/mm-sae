import torch
from svm_pursuit import ProbeLoss, fit_pursuit
torch.manual_seed(7)
x=torch.randn(70,9,dtype=torch.float64); x-=x.mean(0)
y=torch.randint(0,2,(70,3)).double();q=torch.ones(70,dtype=torch.float64)
w=torch.randn(9,3,dtype=torch.float64,requires_grad=True);b=torch.randn(3,dtype=torch.float64,requires_grad=True)
p=ProbeLoss(x,y,q,'squared_hinge',.01)
v,g,h=p.full(w.detach(),b.detach())
ref=(1-(2*y-1)*(x@w+b)).clamp_min(0).square().sum(1).mean()+.005*w.square().sum();ref.backward()
assert abs(v-float(ref.detach()))<1e-10
torch.testing.assert_close(g,w.grad);torch.testing.assert_close(h,b.grad)
a,bb,audit=fit_pursuit(x,y,q,kind='squared_hinge',penalty=.01,budget=3,max_outer=5,inner_iter=80)
assert (a!=0).sum(0).max()<=3
assert audit['objective']<=audit['initial_objective']
print('Squared-hinge objective, analytical gradient, support budget, descent checks passed')

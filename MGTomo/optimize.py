import torch

def armijo_linesearch(f, x: torch.tensor, d: torch.tensor, a=1., r=0.5, c=1e-4):
    fx = f(x)
    fx.backward()
    dgk = torch.sum(x.grad * d)
    
    assert dgk <= 0, 'd needs to be a descent direction (dgk = %.5e)' % dgk
    
    if dgk == 0.:
        return x, 0.
    
    while True:
        x_new = x + a * d
        f_new = f(x_new)
        
        if f_new <= fx + a * c * dgk and x_new.min() >= 0:
            break
        
        a *= r
        if a <= 1e-7:
            return x, 0.
    
    return x_new, a
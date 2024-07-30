import torch

def armijo_linesearch_box(f, x: torch.tensor, d: torch.tensor, a=1., r=0.5, c=1e-3, verbose = True):
    fx = f(x)
    fx.backward()
    dgk = torch.sum(x.grad * d)
    
    assert dgk <= 0, 'd needs to be a descent direction (dgk = %.5e)' % dgk
    
    if dgk == 0.:
        return x, 0.
    
    while True:
        x_new = x + a * d
        
        mask0 = (torch.abs(x_new) <= 1e-5)
        x_new[mask0] = 0
        
        mask1 = torch.logical_and(x_new > 1, x_new <= 1+1e-5)
        x_new[mask1] = 1
        
        
        f_new = f(x_new)
        
        if f_new <= fx + a * c * dgk:
            if verbose:
                print('at a = ', a, 'f_new is <= and x_new.min() = ', x_new.min(), 'with #<0: ', sum(sum(i < 0 for i in x_new)), 'and x_new.max() = ', x_new.max(), 'with #>1: ', sum(sum(i > 1 for i in x_new)))
            if x_new.min() >= 0 and x_new.max() <= 1:
                break
        
        a *= r
        if a <= 1e-7:
            if verbose:
                print('Armijo step too small, a = 0', 'x_new.min() = ', x_new.min(), ' x_new.argmin()' , (x_new==torch.min(x_new)).nonzero() ,sum(sum(i < 0 for i in x_new)), 'indices < 0')
                print('Armijo step too small, a = 0', 'x_new.max() = ', x_new.max(), ' x_new.argmax()' , (x_new==torch.max(x_new)).nonzero() ,sum(sum(i > 0 for i in x_new)), 'indices > 1')
            return x, 0.
    
    return x_new, a


def armijo_linesearch(f, x: torch.tensor, d: torch.tensor, a=1., r = 0.5, c = 1e-3):
    fx = f(x)
    fx.backward()
    dgk = torch.sum(x.grad * d)
    
    assert dgk <= 0, 'd needs to be a descent direction (dgk = %.5e)' % dgk
    
    if dgk == 0.:
        return x, 0.
    
    while True:
        x_new = x + a * d
        
        f_new = f(x_new)
        
        if f_new <= fx + a * c * dgk:
            break
        
        a *= r
        if a <= 1e-7:
            print('Armijo step too small, a = 0')
            return x, 0.
    
    return x_new, a

def box_bounds(xh, xH, P_inf, lh, uh):
    lmax = torch.max(lh - xh)
    umin = torch.min(uh-xh)

    lH = xH + 1/P_inf * lmax
    uH = xH + 1/P_inf * umin

    return lH, uH
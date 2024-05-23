def kl_distance_pts(p: torch.tensor, q: torch.tensor):
    #might need a q>0 mask
    if assert(len(p) == len(q)):
        return torch.inner(p, mylog(p)-mylog(q)) - torch.sum(p-q)

def kl_distance(x: torch.tensor, proj: TomoParallel, b: torch.tensor):
    ax = proj.matvec(x) #type?
    fx = kl_distance_pts(ax, b)
    
    return fx
# ARCHIVED -- kept for the historical record, not part of the live pipeline.
#
# Status: BROKEN. riemann_up() calls christoffel(u) and gaussian_K() calls
# metric(u), but only s2_christoffel(u)/s2_metric(u) are defined in this file
# -- running __main__ raises NameError immediately. Left uncorrected on
# purpose: fixing it here would misrepresent what was actually run.
#
# This was the first attempt at the "pullback metric -> curvature -> verify
# Chern-Gauss-Bonnet" pipeline, hardcoded to the analytic S^2 embedding. The
# working, corrected version -- parametrized over an arbitrary embedding
# (so it also works when the embedding is a decoder network) -- is
# legacy/simpleembedding.py. The maintained, general implementation used by
# the current codebase lives in vae_topology/geometry.py.

import torch
from torch.func import jacrev, vmap

torch.set_default_dtype(torch.float64)
 
 
def s2_embedding(u):
    theta, phi = u[0], u[1]
    return torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ])
 
 

def s2_metric(u):
    J = jacrev(s2_embedding)(u)          
    return J.T @ J                    
 
 
def s2_christoffel(u):
    g = s2_metric(u)
    g_inv = torch.linalg.inv(g)
    dg = jacrev(s2_metric)(u)           
    G  = 0.5 * torch.einsum('kl,jli->kij', g_inv, dg)   
    G += 0.5 * torch.einsum('kl,ilj->kij', g_inv, dg)   
    G -= 0.5 * torch.einsum('kl,ijl->kij', g_inv, dg) 
    return G                         
 
 

def riemann_up(u):
    G  = christoffel(u)
    dG = jacrev(christoffel)(u)      
    t1 = torch.einsum('rnsm->rsmn', dG)
    t2 = torch.einsum('rmsn->rsmn', dG)
    t3 = torch.einsum('rml,lns->rsmn', G, G)
    t4 = torch.einsum('rnl,lms->rsmn', G, G)
    return t1 - t2 + t3 - t4         
 

def gaussian_K(u):
    g = metric(u)
    Ru = riemann_up(u)
    Rl = torch.einsum('pl,lsmn->psmn', g, Ru)  
    return Rl[0, 1, 0, 1] / torch.linalg.det(g)
 
 

def integrand(u):
    return gaussian_K(u) * torch.sqrt(torch.linalg.det(metric(u)))
 
 
if __name__ == "__main__":
    print("point checks:")
    for th, ph in [(0.7, 0.3), (1.2, 2.0), (2.5, 5.0)]:
        u = torch.tensor([th, ph])
        g = metric(u)
        K = gaussian_K(u).item()
        print(f"  theta={th:.2f}  g=diag({g[0,0]:.4f},{g[1,1]:.4f})  "
              f"sin^2={torch.sin(u[0])**2:.4f}   K={K:.6f}")
 

    batch_integrand = vmap(integrand)
 

    N = 200
    th = (torch.arange(N) + 0.5) * torch.pi / N
    ph = (torch.arange(N) + 0.5) * 2 * torch.pi / N
    TH, PH = torch.meshgrid(th, ph, indexing="ij")
    pts = torch.stack([TH.reshape(-1), PH.reshape(-1)], dim=1)
    dA = (torch.pi / N) * (2 * torch.pi / N)
    grid = (batch_integrand(pts).sum() * dA).item()
 

    torch.manual_seed(0)
    M = 4000
    ts = torch.empty(M).uniform_(1e-3, torch.pi - 1e-3)
    ps = torch.empty(M).uniform_(0, 2 * torch.pi)
    samp = torch.stack([ts, ps], dim=1)
    area = torch.pi * 2 * torch.pi
    vals = batch_integrand(samp)
    mc = (area * vals.mean()).item()
    err = (area * vals.std() / M**0.5).item()
 
    print(f"\nGrid integral = {grid:.5f}")
    print(f"MC   integral = {mc:.5f} +/- {err:.5f}")
    print(f"target  4*pi  = {4*torch.pi:.5f}")


    #TASKS:
    #Implement the integration and the embedding as functionals (meaning functions in python taking embedding functions)
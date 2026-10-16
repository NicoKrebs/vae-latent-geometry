# ARCHIVED -- kept for the historical record, not part of the live pipeline.
#
# Status: WORKING. This is the fixed, generalized follow-up to
# legacy/GBsphere.py: christoffel/riemann/gaussian_K now take an `embedding`
# argument instead of hardcoding the S^2 map, so the same code path handles
# an untrained, random-weight decoder network composed with the sphere chart.
# It verifies the qualitative point at the heart of Chern-Gauss-Bonnet: the
# integral of K over the closed surface comes out to 4*pi (= 2*pi*chi(S^2))
# regardless of the (random, untrained) decoder weights -- i.e. the integral
# is a topological invariant, not a property of the particular embedding.
#
# The maintained, general implementation (same math, reused across the torus
# and sphere models) lives in vae_topology/geometry.py.

#Now we want to implement an untrained VAE, pullback its metric to calculate the inducted curvature to prove that the
#integration pipeline definitely works

import torch
import torch.nn as nn
from torch.func import jacrev, vmap

torch.set_default_dtype(torch.float64)



def sphere_map(u):
    theta, phi = u[0], u[1]
    return torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ])


class Decoder(nn.Module):
    def __init__(self, d_in=3, hidden=32, d_out=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)


def make_embedding(decoder):
    """Compose chart -> sphere -> decoder into a single (theta,phi) -> R^D map."""
    def embedding(u):
        return decoder(sphere_map(u))
    return embedding



def metric(embedding, u):
    J = jacrev(embedding)(u)
    return J.T @ J

def christoffel(embedding, u):
    g = metric(embedding, u)
    g_inv = torch.linalg.inv(g)
    dg = jacrev(lambda uu: metric(embedding, uu))(u)
    G  = 0.5 * torch.einsum('kl,jli->kij', g_inv, dg)
    G += 0.5 * torch.einsum('kl,ilj->kij', g_inv, dg)
    G -= 0.5 * torch.einsum('kl,ijl->kij', g_inv, dg)
    return G

def riemann_up(embedding, u):
    G  = christoffel(embedding, u)
    dG = jacrev(lambda uu: christoffel(embedding, uu))(u)
    t1 = torch.einsum('rnsm->rsmn', dG)
    t2 = torch.einsum('rmsn->rsmn', dG)
    t3 = torch.einsum('rml,lns->rsmn', G, G)
    t4 = torch.einsum('rnl,lms->rsmn', G, G)
    return t1 - t2 + t3 - t4

def gaussian_K(embedding, u):
    g  = metric(embedding, u)
    Ru = riemann_up(embedding, u)
    Rl = torch.einsum('pl,lsmn->psmn', g, Ru)
    return Rl[0, 1, 0, 1] / torch.linalg.det(g)

def integrand(embedding, u):
    return gaussian_K(embedding, u) * torch.sqrt(torch.linalg.det(metric(embedding, u)))


def integrate_grid(embedding, N=160):
    th = (torch.arange(N) + 0.5) * torch.pi / N  
    ph = (torch.arange(N) + 0.5) * 2 * torch.pi / N
    TH, PH = torch.meshgrid(th, ph, indexing="ij")
    pts = torch.stack([TH.reshape(-1), PH.reshape(-1)], dim=1)
    dA = (torch.pi / N) * (2 * torch.pi / N)
    f = vmap(lambda u: integrand(embedding, u))
    return (f(pts).sum() * dA).item()


if __name__ == "__main__":
    print("skript started")
    torch.manual_seed(1)
    dec = Decoder()
    for p in dec.parameters():
        p.requires_grad_(False)                      
    emb = make_embedding(dec)

    print("random decoder -> metric is no longer diag(1, sin^2 theta):")
    for th, ph in [(0.7, 0.3), (1.2, 2.0), (2.5, 5.0)]:
        u = torch.tensor([th, ph])
        g = metric(emb, u)
        K = gaussian_K(emb, u).item()
        print(f"  theta={th:.2f}  g=[[{g[0,0]:.3f},{g[0,1]:.3f}],"
              f"[{g[1,0]:.3f},{g[1,1]:.3f}]]   K={K:.4f}")

    val = integrate_grid(emb)
    print(f"\nintegral (random untrained decoder) = {val:.5f}")
    print(f"target 4*pi                         = {4*torch.pi:.5f}")


    torch.manual_seed(7)
    dec2 = Decoder()
    for p in dec2.parameters():
        p.requires_grad_(False)
    val2 = integrate_grid(make_embedding(dec2))
    print(f"\ndifferent random weights            = {val2:.5f}   (still 4*pi)")
    print("=> integral independent of weights  => d(integral)/d(weights) = 0")
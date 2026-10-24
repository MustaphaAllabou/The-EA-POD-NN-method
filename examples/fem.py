"""P1 finite elements + SUPG for the parametrized steady advection-diffusion problem

    -div(lambda(mu) grad u) + w . grad u = f   in (0,1)^2
    u = 1 on Gamma_D = {0} x (0,1),   -lambda grad u . n = 0 on Gamma_N.

Affine structure (P1, SUPG with tau_K = h_K / (2|w|_K)):
    A(mu) = lambda(mu) * K + N,   N = B (advection) + S (SUPG),   F = f + f_supg
The lifting u_g = 1 has zero gradient, hence F does not depend on mu.
"""
import numpy as np
import scipy.sparse as sp

MU0 = 0.7
MU_MIN, MU_MAX = 1.0, 10.0


def lam(mu):
    return np.exp(MU0 * (np.asarray(mu) + 1.0))


def w0(x, y):
    return np.stack([1.0 + np.abs(np.cos(2 * np.pi * x)),
                     np.sin(2 * np.pi * x / 0.2)], axis=-1)


def f_src(x, y):
    return 1e3 * np.exp(-(np.abs(x - 0.5) + np.abs(y - 0.5)) / 0.1)


class Problem:
    def __init__(self, n, alpha):
        self.n, self.alpha = n, alpha
        h = 1.0 / n
        xs = np.linspace(0, 1, n + 1)
        X, Y = np.meshgrid(xs, xs)              # node (i,j) -> i + j(n+1)
        self.x, self.y = X.ravel(), Y.ravel()
        idx = lambda i, j: i + j * (n + 1)
        tris = []
        for j in range(n):
            for i in range(n):
                a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
                tris += [(a, b, c), (a, c, d)]
        T = np.array(tris)
        self.tris = T
        Nn = len(self.x)
        P = np.stack([self.x[T], self.y[T]], -1)            # (nt,3,2)
        e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        area = 0.5 * np.abs(det)
        # gradients of barycentric functions
        G = np.zeros((len(T), 3, 2))
        G[:, 1] = np.stack([e2[:, 1], -e2[:, 0]], -1) / det[:, None]
        G[:, 2] = np.stack([-e1[:, 1], e1[:, 0]], -1) / det[:, None]
        G[:, 0] = -G[:, 1] - G[:, 2]
        # edge-midpoint quadrature (exact for P2)
        qb = np.array([[.5, .5, 0], [0, .5, .5], [.5, 0, .5]])  # barycentric coords
        Q = np.einsum('qk,tkd->tqd', qb, P)                     # (nt,3q,2)
        W = alpha * w0(Q[..., 0], Q[..., 1])                    # (nt,q,2)
        Fq = f_src(Q[..., 0], Q[..., 1])                        # (nt,q)
        cen = P.mean(1)
        wc = np.linalg.norm(alpha * w0(cen[:, 0], cen[:, 1]), axis=-1)
        hK = np.sqrt(2.0) * h
        tau = hK / (2.0 * wc)
        self.h = h
        # local matrices
        Mloc = area[:, None, None] / 12.0 * (np.ones((3, 3)) + np.eye(3))[None]
        Kloc = area[:, None, None] * np.einsum('tid,tjd->tij', G, G)
        wg = np.einsum('tqd,tjd->tqj', W, G)                    # w(x_q).grad phi_j
        Bloc = area[:, None, None] / 3.0 * np.einsum('qi,tqj->tij', qb, wg)
        Sloc = (tau * area / 3.0)[:, None, None] * np.einsum('tqi,tqj->tij', wg, wg)
        floc = area[:, None] / 3.0 * np.einsum('qi,tq->ti', qb, Fq)
        fsloc = (tau * area / 3.0)[:, None] * np.einsum('tq,tqi->ti', Fq, wg)
        rows = np.repeat(T, 3, axis=1).ravel()
        cols = np.tile(T, (1, 3)).ravel()
        asm = lambda L: sp.csr_matrix((L.ravel(), (rows, cols)), shape=(Nn, Nn))
        self.Mf, self.Kf = asm(Mloc), asm(Kloc)
        Bf, Sf = asm(Bloc), asm(Sloc)
        ff = np.bincount(T.ravel(), (floc + fsloc).ravel(), minlength=Nn)
        self.free = np.where(self.x > 1e-12)[0]
        fr = self.free
        sub = lambda A: A[fr][:, fr].tocsc()
        self.M, self.K = sub(self.Mf), sub(self.Kf)
        self.B, self.S = sub(Bf), sub(Sf)
        self.N = (self.B + self.S).tocsc()
        self.X = (self.M + self.K).tocsc()     # H^1 inner product
        self.F = ff[fr]
        self.Nh = len(fr)
        self.wnorm_nodes = np.linalg.norm(alpha * w0(self.x, self.y), axis=-1)

    def A(self, mu):
        return (lam(mu) * self.K + self.N).tocsc()

    def solve(self, mu):
        import scipy.sparse.linalg as spla
        return spla.spsolve(self.A(mu), self.F)

    def full(self, u0):
        """homogeneous free-dof vector -> full nodal field (with lifting u=1)."""
        u = np.ones(len(self.x))
        u[self.free] += u0
        return u

# Author: Kimia Nadjahi
# Some parts of this code are taken from https://github.com/skolouri/swgmm

import numpy as np
from scipy import interp
import ot
import HilbertCode_caller
import swapsweep


def wass_distance(X, Y, order=2, type='exact'):
    """
    Computes the (exact or approximate) Wasserstein distance of order 1 or 2 between empirical distributions
    """
    if order == 2:
        M = ot.dist(X, Y)
    elif order == 1:
        M = ot.dist(X, Y, metric='euclidean')
    else:
        raise Exception("Order should be 1 or 2.")
    a = np.ones((X.shape[0],))/X.shape[0]
    b = np.ones((Y.shape[0],))/Y.shape[0]
    if type == 'approx':
        # Regularized problem solved with Sinkhorn
        reg = 1
        return ot.sinkhorn2(a, b, M, reg)**(1/order)
    else:
        # Compute exactly
        return ot.emd2(a, b, M)**(1/order)


def wass_gaussians(mu1, mu2, Sigma1, Sigma2):
    """
    Computes the Wasserstein distance of order 2 between two Gaussian distributions
    """
    d = mu1.shape[0]
    if d == 1:
        w2 = (mu1 - mu2)**2 + (np.sqrt(Sigma1) - np.sqrt(Sigma2))**2
    else:
        prodSigmas = Sigma2**(1/2)*Sigma1*Sigma2**(1/2)
        w2 = np.linalg.norm(mu1 - mu2)**2 + np.trace(Sigma1 + Sigma2 - 2*(prodSigmas)**(1/2))
    return np.sqrt(w2)


def sw_distance(X, Y, n_montecarlo=1, L=100, p=2):
    """
    Computes the Sliced-Wasserstein distance between empirical distributions
    """
    X = np.stack([X] * n_montecarlo)
    M, N, d = X.shape
    order = p
    # Project data
    theta = np.random.randn(M, L, d)
    theta = theta / (np.sqrt((theta ** 2).sum(axis=2)))[:, :, None]  # Normalize
    theta = np.transpose(theta, (0, 2, 1))
    xproj = np.matmul(X, theta)
    yproj = np.matmul(Y, theta)
    # Compute percentiles
    T = 100
    t = np.linspace(0, 100, T + 2)
    t = t[1:-1]
    xqf = (np.percentile(xproj, q=t, axis=1))
    yqf = (np.percentile(yproj, q=t, axis=1))
    # Compute expected SW distance
    diff = (xqf - yqf).transpose((1, 0, 2))
    sw_dist = (np.abs(diff) ** order).mean()
    sw_dist = sw_dist ** (1/order)
    return sw_dist


def sw_gaussians(mu1, mu2, Sigma1, Sigma2, n_proj=100):
    """
    Computes the Sliced-Wasserstein distance of order 2 between two Gaussian distributions
    """
    d = mu1.shape[0]
    # Project data
    thetas = np.random.randn(n_proj, d)
    thetas = thetas / (np.sqrt((thetas ** 2).sum(axis=1)))[:, None]  # Normalize
    proj_mu1 = thetas @ mu1
    proj_mu2 = thetas @ mu2
    sw2 = 0
    for l in range(n_proj):
        th = thetas[l]
        proj_sigma1 = (th @ Sigma1) @ th
        proj_sigma2 = (th @ Sigma2) @ th
        sw2 += wass_gaussians(np.array([proj_mu1[l]]), np.array([proj_mu2[l]]),
                              np.array([proj_sigma1]), np.array([proj_sigma2]))**2
    sw2 /= n_proj
    return np.sqrt(sw2)


def hilbert_distance(X, Y, p=2):
    """
    Computes the Hilbert distance of order p
    """
    # We consider N_X = N_Y
    xordered = X[HilbertCode_caller.hilbert_order_(X.T)]
    yordered = Y[HilbertCode_caller.hilbert_order_(Y.T)]
    hilbert_dist = (np.abs(xordered - yordered) ** p).sum()
    hilbert_dist /= X.shape[0]
    hilbert_dist = hilbert_dist ** (1/p)
    return hilbert_dist


def swapsweep_py(permutation, M, total_cost):
    """
    Sweeping operation
    """
    N = M.shape[0]
    for i in range(N):
        perm_i = permutation[i]
        for j in range(i+1, N):
            perm_j = permutation[j]
            current_cost = M[i, perm_i] + M[j, perm_j]
            proposed_cost = M[i, perm_j] + M[j, perm_i]
            if proposed_cost < current_cost:
                permutation[i] = perm_j
                permutation[j] = perm_i
                perm_i = perm_j
                total_cost = total_cost - current_cost + proposed_cost
    return permutation, total_cost


def swap_distance(X, Y, n_sweeps=10000, tol=1e-8, p=2):
    """
    Computes the swapping distance
    """
    # We consider N_X = N_Y
    if p == 2:
        M = ot.dist(X, Y)  # Cost matrix
    o1 = HilbertCode_caller.hilbert_order_(X.T)
    o2 = HilbertCode_caller.hilbert_order_(Y.T)
    permutation = o2[np.argsort(o1)]
    total_cost = list(map(lambda k: M[k, permutation[k]], range(X.shape[0])))
    total_cost = np.array(total_cost).sum()
    previous_total_cost = total_cost
    i_sweep = 0
    while i_sweep < n_sweeps:
        i_sweep += 1
        # permutation, total_cost = swapsweep_py(permutation, M, total_cost)  # Slow!
        permutation, total_cost = swapsweep.swapsweep(permutation, M, total_cost)
        error = np.abs(total_cost - previous_total_cost) / X.shape[0]
        if error < tol:
            break
        previous_total_cost = total_cost
    swap_distance = total_cost / X.shape[0]
    swap_distance = swap_distance ** (1/p)
    return swap_distance


def kl_gaussians(sigma1_2, sigma2_2, dim):
    return dim/2 * (np.log(sigma2_2/sigma1_2) + sigma1_2/sigma2_2 - 1)


def kl_empirical(X, Y):
    n, d = X.shape
    m, _ = Y.shape
    kl = np.log(m / (n-1))
    for i in range(n):
        min_norms_xy = np.linalg.norm(X[i] - Y, axis=1)
        min_norms_xy = np.min(min_norms_xy)
        min_norms_xx = np.linalg.norm(X[i] - X, axis=1)
        min_norms_xx = np.min(min_norms_xx[min_norms_xx != np.min(min_norms_xx)])
        kl += np.log(min_norms_xy / min_norms_xx)
    kl = (d/n) * kl
    return kl

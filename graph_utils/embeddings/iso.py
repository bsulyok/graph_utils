from typing import Dict, List, Any
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.special import zeta

MIN_SAMPLE_SIZE = 50
FALLBACK_GAMMA = 3


def _find_gamma(degree_list: list) -> float:
    count = np.bincount(degree_list)
    count = count[:-1]
    ccdf = 1 - np.cumsum(count) / len(degree_list)
    degree = np.arange(ccdf.shape[0])
    best_gamma, lowest_max_deviation = FALLBACK_GAMMA, np.inf
    for min_degree in np.unique(degree_list):
        sample_size = np.sum(count[min_degree:])
        if MIN_SAMPLE_SIZE < sample_size:
            gamma = 1 + 1 / np.average(np.log(degree[min_degree:]/(min_degree-0.5)), weights=count[min_degree:])
            const = np.exp(np.mean(np.log(ccdf[min_degree:])+(gamma-1)*np.log(degree[min_degree:]))) * (gamma-1)
            max_deviation = np.max(np.abs(ccdf[min_degree:] / const - degree[min_degree:]**(1-gamma)/(gamma-1))) / zeta(gamma, min_degree)
            if max_deviation < lowest_max_deviation:
                best_gamma, lowest_max_deviation = gamma, max_deviation
    return best_gamma


def _convert_to_hyperbolic(coord: np.ndarray, G: nx.Graph) -> np.ndarray:
    degree_list = np.array(list(dict(G.degree).values()))
    gamma = _find_gamma(degree_list)
    beta = 1 / (gamma - 1)
    degree_rank = np.argsort(np.argsort(degree_list)[::-1])
    r_hyp = 2 / (coord.shape[0]-1) * (beta * np.log(np.arange(1, coord.shape[0]+1)) + (1-beta) * np.log(coord.shape[0]))
    r_euc = np.linalg.norm(coord, axis=1)
    coord = coord / r_euc[:, np.newaxis] * r_hyp[degree_rank, np.newaxis]
    return coord


def embed_iso(G: nx.Graph, dim: int = 2, hyperbolic=False) -> Dict[Any, List]:
    N = len(G)
    D = shortest_path(nx.adjacency_matrix(G))
    H = np.eye(N) - 1 / N
    D = - H @ (D*D) @ H / 2
    U, S, VH = np.linalg.svd(D, full_matrices=False)
    U = U[:, :dim]
    VH = VH[:dim,:]
    S = np.diag(S[:dim])
    coord_emb = np.transpose(np.sqrt(S) @ VH)
    if hyperbolic:
        coord_emb = _convert_to_hyperbolic(coord_emb, G)
    return dict(zip(G.nodes, coord_emb.tolist()))


def embed_nciso(G: nx.Graph, dim: int = 2, hyperbolic=False) -> Dict[Any, List]:
    N = len(G)
    D = shortest_path(nx.adjacency_matrix(G))
    H = np.eye(N) - 1 / N
    D = - H @ (D*D) @ H / 2
    U, S, VH = np.linalg.svd(D, full_matrices=False)
    U = U[:, :dim+1]
    VH = VH[:dim+1,:]
    S = np.diag(S[:dim+1])
    coord_emb = np.transpose(np.sqrt(S) @ VH)[:, 1:]
    if hyperbolic:
        coord_emb = _convert_to_hyperbolic(coord_emb, G)
    return dict(zip(G.nodes, coord_emb.tolist()))
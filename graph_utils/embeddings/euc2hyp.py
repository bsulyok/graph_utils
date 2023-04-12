from typing import Dict, List, Any
import networkx as nx
import numpy as np
from scipy.special import zeta

MIN_SAMPLE_SIZE = 50
FALLBACK_GAMMA = 3


def find_gamma(degree_list: list) -> float:
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


def embed(coord_dict: np.ndarray, G: nx.Graph) -> np.ndarray:
    coord = np.array(list(coord_dict.values()))
    degree_list = np.array(list(dict(G.degree).values()))
    gamma = find_gamma(degree_list)
    beta = 1 / (gamma - 1)
    degree_rank = np.argsort(np.argsort(degree_list)[::-1])
    r_hyp = 2 / (coord.shape[0]-1) * (beta * np.log(np.arange(1, coord.shape[0]+1)) + (1-beta) * np.log(coord.shape[0]))
    r_euc = np.linalg.norm(coord, axis=1)
    coord = coord / r_euc[:, np.newaxis] * r_hyp[degree_rank, np.newaxis]
    return dict(zip(coord_dict, map(list, coord)))
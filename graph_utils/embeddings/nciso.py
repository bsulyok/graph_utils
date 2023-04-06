from typing import Dict, List, Any
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path


def embed(G: nx.Graph, dim: int = 2) -> Dict[Any, List]:
    N = len(G)
    D = shortest_path(nx.adjacency_matrix(G))
    H = np.eye(N) - 1 / N
    D = - H @ (D*D) @ H / 2
    U, S, VH = np.linalg.svd(D, full_matrices=False)
    U = U[:, :dim+1]
    VH = VH[:dim+1,:]
    S = np.diag(S[:dim+1])
    coord_emb = np.transpose(np.sqrt(S) @ VH)[:, 1:]
    return dict(zip(G.nodes, coord_emb.tolist()))
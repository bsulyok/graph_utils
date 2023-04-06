import mercator # pip install git+https://github.com/networkgeometry/mercator.git@master
from typing import Dict, List, Any
from tempfile import TemporaryDirectory
import networkx as nx
import pandas as pd
import numpy as np

def embed(G: nx.Graph) -> Dict[Any, List]:
    with TemporaryDirectory() as tempdir:
        nx.write_edgelist(G, f'{tempdir}/graph.edgelist', data=False)
        try:
            mercator.embed(f'{tempdir}/graph.edgelist', clean_mode=True, quiet_mode=True)
        except:
            pass
        inf_coord = pd.read_csv(f'{tempdir}/graph.inf_coord', delim_whitespace=True, comment='#', names=['vertex', 'kappa', 'theta', 'r'])
        inf_coord['x'] = inf_coord['r'] * np.cos(inf_coord['theta'])
        inf_coord['y'] = inf_coord['r'] * np.sin(inf_coord['theta'])
        return dict(zip(inf_coord['vertex'], map(list, inf_coord[['x', 'y']].values)))
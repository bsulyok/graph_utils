from typing import Dict, List, Any
import numpy as np
from openTSNE import TSNE


def embed(coords: Dict[Any, List], **kwargs) -> Dict[Any, List]:
    args = {'perplexity': 30, 'n_jobs': 8}
    args.update(kwargs)
    return dict(zip(coords, map(list, TSNE(**args).fit(np.array(list(coords.values()))))))
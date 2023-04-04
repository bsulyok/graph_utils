from typing import Dict, List, Any
from node2vec import Node2Vec

from typing import Dict, List, Any
def embed(G: Dict[Any, List], **kwargs) -> Dict[Any, List]:
    model_params = {
        'p': 1,
        'q': 1,
        'dimensions': 128,
        'walk_length': 80,
        'num_walks': 10,
        'workers': 1,
        'weight_key': 'weight',
        'sampling_strategy': None,
        'seed': None,
        'quiet': True
    }
    fit_params = {
        'window': 10,
        'min_count': 1,
        'batch_words': 4,
        'seed': None
    }
    model_params.update(filter(lambda item: item[0] in model_params, kwargs.items()))
    fit_params.update(filter(lambda item: item[0] in fit_params, kwargs.items()))
    model = Node2Vec(G, **model_params).fit(**fit_params)
    return dict(zip(map(type(list(G.nodes)[0]), model.wv.index_to_key), model.wv.vectors.tolist()))
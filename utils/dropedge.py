import torch
from torch.distributions import Bernoulli

class BaseTransform:
    r"""An abstract class for writing transforms."""
    def __call__(self, g):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DropEdge(BaseTransform):
    r"""Randomly drop edges, as described in
    `DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
    <https://arxiv.org/abs/1907.10903>`__ and `Graph Contrastive Learning with Augmentations
    <https://arxiv.org/abs/2010.13902>`__.

    Parameters
    ----------
    p : float, optional
        Probability of an edge to be dropped.

    Example
    -------

    >>> import dgl
    >>> import torch
    >>> from dgl import DropEdge

    >>> transform = DropEdge()
    >>> g = dgl.rand_graph(5, 20)
    >>> g.edata['h'] = torch.arange(g.num_edges())
    >>> new_g = transform(g)
    >>> print(new_g)
    Graph(num_nodes=5, num_edges=12,
          ndata_schemes={}
          edata_schemes={'h': Scheme(shape=(), dtype=torch.int64)})
    >>> print(new_g.edata['h'])
    tensor([0, 1, 3, 7, 8, 10, 11, 12, 13, 15, 18, 19])
    """
    def __init__(self, p=0.5):
        self.p = p
        self.dist = Bernoulli(p)
        # 

    def __call__(self, g):
        # Fast path
        if self.p == 0:
            return g

        for c_etype in g.canonical_etypes:
            samples = self.dist.sample(torch.Size([g.num_edges(c_etype)]))
            eids_to_remove = g.edges(form='eid', etype=c_etype)[samples.bool().to(g.device)]
            g.remove_edges(eids_to_remove, etype=c_etype)
        return g, eids_to_remove
import pytest
import torch
import torch.nn.functional as F
from evidence.representative_descriptor import representatives


def test_medoid_matches_explicit_pairwise_objective():
    torch.manual_seed(17)
    d = F.normalize(torch.randn(15, 8), dim=1)
    bins = torch.tensor([0] * 10 + [1] * 3 + [2] * 2)
    w = torch.rand(15)
    reference = F.normalize(torch.randn(8), dim=0)
    out, indices = representatives(d, bins, w, reference)
    mass = torch.stack([w[bins == b].sum() for b in bins])
    expected = ((d @ d.T) @ (w / mass)).argmax()
    assert indices[0] == expected
    assert indices[1] == (d @ reference).argmax()
    assert torch.allclose(out, d[indices])


def test_bin_duplication_does_not_change_representative():
    d = F.normalize(torch.tensor([[1., 0.], [0., 1.], [.8, .6]]), dim=1)
    a, _ = representatives(d, torch.arange(3), torch.ones(3), d[2])
    b, _ = representatives(torch.cat([d, d[:1].repeat(20, 1)]),
                           torch.cat([torch.arange(3), torch.zeros(20, dtype=torch.long)]), torch.ones(23), d[2])
    assert torch.allclose(a, b)


def test_singleton_and_invalid_weights():
    d = torch.tensor([[3., 4.]])
    out, indices = representatives(d, torch.tensor([0]), torch.ones(1), d[0])
    assert torch.equal(indices, torch.zeros(2, dtype=torch.long))
    assert torch.allclose(out, torch.tensor([[.6, .8], [.6, .8]]))
    with pytest.raises(ValueError):
        representatives(d, torch.tensor([0]), torch.tensor([float('nan')]), d[0])

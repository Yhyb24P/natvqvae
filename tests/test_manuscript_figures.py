import numpy as np
import pytest
from natvqvae.manuscript_figures import component_means, paired_geometry


def test_component_mean_preserves_equal_component_weight_and_pairing():
    rows=[{'component':'b','nll':'2'},{'component':'a','nll':'1'},
          {'component':'b','nll':'4'},{'component':'b','nll':'6'}]
    np.testing.assert_allclose(component_means(rows,'nll'),[1,4])
    esm=np.array([1.,1.,1.,1.])
    x,y,counts=paired_geometry(esm,np.array([1.4,.8,1.3,.7]),np.array([.9,.7,1.2,1.1]))
    assert counts=={'task_worse_mix_better':1,'both_better':1,'both_worse':1,
                    'task_better_mix_worse':1,'mix_better_than_esm':2,'mix_better_than_both':2}
    np.testing.assert_allclose(x,[.4,-.2,.3,-.3]);np.testing.assert_allclose(y,[-.1,-.3,.2,.1])
    with pytest.raises(ValueError,match='Nonfinite'):
        paired_geometry(esm,esm,np.array([1,1,1,np.nan]))

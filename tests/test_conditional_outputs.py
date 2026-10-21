import numpy as np
from natvqvae.conditional_outputs import mixture_statistics


def test_same_marginals_different_joint_distribution():
    # 00/11各半的共同潜变量，与四种组合等概率的独立边缘。
    p=np.array([[[1.,0.],[1.,0.]],[[0.,1.],[0.,1.]]])
    shared,independent=mixture_statistics(p)
    assert np.allclose(shared,[[.5,0],[0,.5]])
    assert np.allclose(independent,np.full((2,2),.25))
    assert np.allclose(shared.sum(0),independent.sum(0))
    assert np.log(shared[0,0])-np.log(independent[0,0])==np.log(2)

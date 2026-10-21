import itertools
from types import SimpleNamespace
import pytest
np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')
from natvqvae.conditional_generation import (ConditionalModel, sample_letters, categorical_kl,
    categorical_samples, categorical_log_prob, rloo_objective, target_tensors, reconstruction_nll)
from natvqvae.conditional_data import mask_panels, region_intervals


def test_sampler_retains_wt_and_handles_cdf_zero_boundary():
    logits=np.full((2,20),-1000.);logits[:,0]=0
    assert list(sample_letters(logits,[0.,.999]))==['A','A']
    with pytest.raises(ValueError):sample_letters(logits,[1.,0.])


def test_posterior_separation_and_shared_initialization():
    torch.set_num_threads(1)
    torch.manual_seed(7);q=ConditionalModel('QVAE',hidden=16,factors=2,states=3,feature_dim=12)
    torch.manual_seed(7);d=ConditionalModel('deterministic',hidden=16,factors=2,states=3,feature_dim=12)
    assert all(torch.equal(q.state_dict()[k],v) for k,v in d.state_dict().items())
    masked=torch.randn(2,7,12);full=torch.randn_like(masked)
    valid=torch.ones(2,7,dtype=torch.bool);mask=torch.zeros_like(valid);mask[:,1:4]=True
    labels=torch.zeros(2,7,dtype=torch.long)
    h=q.encode_context(masked,valid,mask);p=q.prior(h,valid,mask)
    posterior=q.posterior(full,h,valid,mask)
    assert not torch.allclose(posterior,q.posterior(full+1,h,valid,mask))
    assert torch.equal(p,q.prior(h,valid,mask))
    targets,y,use,_=target_tensors(h,mask,labels)
    generator=torch.Generator().manual_seed(1)
    codes=q.sample_prior(h,valid,mask,4,generator)
    assert codes.shape==(4,2,2)
    prediction=q.decode_codes(targets,codes)
    q.posterior(full+1,h,valid,mask)
    assert torch.equal(prediction,q.decode_codes(targets,codes))
    with pytest.raises(TypeError):q.prior(h,valid,mask,full)
    with pytest.raises(ValueError):d.posterior(full,h,valid,mask)
    nll=reconstruction_nll(prediction,y,use)
    assert nll.shape==(4,2) and torch.isfinite(nll).all()
    loss=rloo_objective(nll,categorical_log_prob(posterior,codes),categorical_kl(posterior,p),1.,torch.ones(2))
    loss.backward()
    for name in ['full_projection.weight','posterior_head.2.weight','condition_head.2.weight','decoder_film.0.weight']:
        assert dict(q.named_parameters())[name].grad.abs().sum()>0


def test_exact_expected_rloo_gradient_for_two_categorical_factors():
    # 穷举 K=2 的81种采样结果，验证估计器的期望梯度，而非只检查梯度存在。
    torch.set_num_threads(1)
    q=torch.tensor([[[.2,-.4,.1],[.5,-.2,.1]]],dtype=torch.float64,requires_grad=True)
    p=torch.tensor([[[.1,.1,-.1],[-.1,.2,.1]]],dtype=torch.float64,requires_grad=True)
    rewards=torch.linspace(.1,2.,9,dtype=torch.float64).square().requires_grad_()
    codes=torch.tensor(list(itertools.product(range(3),repeat=2)))[:,None,:]
    logq=categorical_log_prob(q,codes)[:,0];prob=logq.exp()
    kl=categorical_kl(q,p)
    exact=(prob*rewards).sum()+.7*kl.sum()
    reference=torch.autograd.grad(exact,(q,p,rewards),retain_graph=True)
    estimates=[]
    for a,b in itertools.product(range(9),repeat=2):
        indices=torch.tensor([a,b])
        estimate=rloo_objective(rewards[indices,None],logq[indices,None],kl,.7,torch.ones(1))
        estimates.append((prob[a]*prob[b]).detach()*estimate)
    result=torch.autograd.grad(sum(estimates),(q,p,rewards))
    for actual,expected in zip(result,reference):
        assert torch.allclose(actual,expected,atol=3e-7,rtol=3e-6)
    with pytest.raises(ValueError):rloo_objective(torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1),1.,torch.ones(1))


def test_short_linker_masks_and_mixture_weights():
    row=SimpleNamespace(sequence_id='a'*64,length=300,nmnat_cdd_from=4,nmnat_cdd_to=181,nudix_cdd_from=191,nudix_cdd_to=295)
    panels=mask_panels(row,123)
    assert len(panels)==27 and np.isclose(sum(p['weight'] for p in panels),1)
    intervals=region_intervals(row)
    for p in panels:
        assert p['positions'] and len(p['positions'])==len(set(p['positions']))
        if p['region']!='random':
            low,high=intervals[p['region']]
            assert all(low<=i<high for i in p['positions'])
    fixed=next(p for p in panels if p['panel']=='linker_16_count')
    relative=next(p for p in panels if p['panel']=='linker_16_density')
    assert len(fixed['positions'])==9 and len(relative['positions'])==7
    assert panels==mask_panels(row,123)

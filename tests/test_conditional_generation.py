import pytest
np=pytest.importorskip('numpy')
torch=pytest.importorskip('torch')
from natvqvae.conditional_generation import ConditionalModel,sample_letters,AA


def test_sampler_retains_wt_and_handles_cdf_zero_boundary():
    logits=np.full((2,20),-1000.);logits[:,0]=0
    assert list(sample_letters(logits,[0.,.999]))==['A','A']
    with pytest.raises(ValueError):sample_letters(logits,[1.,0.])


def test_posterior_reads_targets_and_prior_cannot_read_them():
    torch.set_num_threads(1)
    torch.manual_seed(7);q=ConditionalModel('QVAE',hidden=16,factors=2,states=3)
    torch.manual_seed(7);d=ConditionalModel('deterministic',hidden=16,factors=2,states=3)
    assert all(torch.equal(q.state_dict()[k],v) for k,v in d.state_dict().items())
    h=torch.randn(2,5,1280);b=torch.randn(2,5,20);valid=torch.ones(2,5,dtype=torch.bool);mask=valid.clone()
    labels=torch.zeros(2,5,dtype=torch.long)
    with pytest.raises(ValueError):q(h,b,valid,mask)
    context=q.encode_context(h,valid)
    scores=q.posterior_scores(context,valid,mask,labels)
    assert not torch.allclose(scores,q.posterior_scores(context,valid,mask,labels+1))
    # 可见位点的标签不应改变后验；真实目标只从 mask 内进入。
    partial=mask.clone();partial[:,0]=False
    changed=labels.clone();changed[:,0]=1
    assert torch.equal(q.posterior_scores(context,valid,partial,labels),q.posterior_scores(context,valid,partial,changed))
    optimizer=torch.optim.AdamW(q.parameters(),lr=.01)
    for _ in range(2):
        optimizer.zero_grad();logits,kl=q(h,b,valid,mask,labels=labels);loss=logits.square().mean()+.01*kl;loss.backward();optimizer.step()
    assert q.global_map.weight.grad.abs().sum()>0
    assert q.global_projection.weight.grad.abs().sum()>0
    assert q.target_embedding.weight.grad.abs().sum()>0
    q.eval()
    with pytest.raises(ValueError):q(h,b,valid,mask)
    z=q.sample_prior(2,torch.Generator().manual_seed(9),'cpu')
    assert q(h,b,valid,mask,z=z)[0].shape==(2,5,20)
    with pytest.raises(ValueError):q(h,b,valid,mask,z=z,labels=labels)
    assert q(h,b,valid,mask,z='posterior',labels=labels)[0].shape==(2,5,20)
    with pytest.raises(ValueError):d(h,b,valid,mask,z=z)
    with pytest.raises(ValueError):q(h,b,valid,mask,z=z*2)

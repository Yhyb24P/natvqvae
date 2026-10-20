from pathlib import Path
import pytest

torch = pytest.importorskip('torch')
from natvqvae.aligned_generator import AlignedGenerator
from natvqvae.intake import source_root, load_manifest


@pytest.fixture
def models():
    source=source_root(load_manifest())
    if not (source/'src/nad_qvae/design/sequence_generator.py').is_file():
        pytest.skip('Verified external implementation unavailable')
    torch.set_num_threads(1)
    config={'hidden_dim':8,'encoder_layers':1,'decoder_layers':1,'attention_heads':2,
            'latent_dim':2,'latent_states':2,'dropout':0.}
    embedding=torch.zeros(341,4)
    torch.manual_seed(1);a=AlignedGenerator(source,config,embedding,'categorical_vae')
    torch.manual_seed(1);b=AlignedGenerator(source,config,embedding,'no_latent')
    return a,b


def test_control_target_identity_cannot_reach_decoder(models):
    vae,control=models
    control.eval()
    first=control(torch.zeros(2,341,dtype=torch.long))[0]
    changed=control(torch.full((2,341),22,dtype=torch.long))[0]
    assert torch.equal(first,changed)
    assert sum(p.numel() for p in vae.parameters())==sum(p.numel() for p in control.parameters())


def test_control_independently_learns_from_targets(models):
    _,control=models
    logits,kl,_=control(torch.zeros(2,341,dtype=torch.long))
    loss,_,_=control.objective(logits,torch.zeros(2,341,dtype=torch.long),kl,1.)
    loss.backward()
    assert control.network.output.weight.grad.abs().sum()>0
    assert control.network.encoder.layers[0].linear1.weight.grad.abs().sum()>0
    assert torch.equal(kl,torch.zeros_like(kl))

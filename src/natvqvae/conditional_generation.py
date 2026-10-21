"""Matched conditional categorical QVAE and deterministic global-context comparator."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
AA='ACDEFGHIKLMNPQRSTVWY'


def sample_letters(logits, uniforms, temperature=1.):
    """Canonical-AA inverse CDF; WT remains eligible. No resampling."""
    scores=np.asarray(logits,dtype=np.float64)/temperature
    if temperature<=0 or scores.shape[-1]!=20 or not np.isfinite(scores).all():raise ValueError('Invalid logits/temperature')
    u=np.asarray(uniforms)
    if u.shape!=scores.shape[:-1] or ((u<0)|(u>=1)).any():raise ValueError('Invalid uniforms')
    weights=np.exp(scores-scores.max(-1,keepdims=True));p=weights/weights.sum(-1,keepdims=True)
    index=(u[...,None]>=np.cumsum(p,axis=-1)).sum(-1).clip(max=19)
    return np.asarray(list(AA))[index]


class ConditionalModel(nn.Module):
    """Identical active parameters; categorical sampling vs deterministic context branch.

    Both receive only freshly masked frozen-ESM representations and logits.
    The deterministic comparator has no random latent or latent regularizer.
    """
    def __init__(self,arm,hidden=128,factors=32,states=8):
        super().__init__()
        if arm not in {'QVAE','deterministic'}:raise ValueError('Unknown arm')
        self.arm=arm;self.factors=factors;self.states=states
        self.sequence=nn.Linear(1280,hidden)
        def transformer(n):
            layer=nn.TransformerEncoderLayer(hidden,4,2*hidden,dropout=0.,batch_first=True)
            return nn.TransformerEncoder(layer,n,enable_nested_tensor=False)
        self.encoder=transformer(2);self.decoder=transformer(1)
        self.global_map=nn.Linear(hidden,factors*states)
        self.global_projection=nn.Linear(factors*states,hidden)
        self.output=nn.Linear(hidden,20)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def sample_prior(self,count,generator,device):
        if self.arm!='QVAE':raise ValueError('Deterministic model has no prior')
        codes=torch.randint(self.states,(count,self.factors),generator=generator,device=device)
        return F.one_hot(codes,self.states).float()

    def forward(self,features,baseline,valid,mask,z=None):
        if features.requires_grad or baseline.requires_grad:raise ValueError('Backbone must be frozen')
        if mask.shape!=valid.shape or (mask & ~valid).any() or not mask.any(1).all():raise ValueError('Invalid target mask')
        if z is not None and self.arm!='QVAE':raise ValueError('Comparator cannot read z')
        length=features.shape[1];d=self.sequence.out_features
        phase=torch.arange(length,device=features.device)[:,None]*torch.exp(torch.arange(0,d,2,device=features.device)*(-math.log(10000)/d))
        position=torch.zeros(length,d,device=features.device);position[:,0::2]=phase.sin();position[:,1::2]=phase.cos()
        h=self.sequence(features.float().masked_fill(~valid[...,None],0))+position
        h=self.encoder(h,src_key_padding_mask=~valid)
        pooled=(h*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)
        global_scores=self.global_map(pooled);regularizer=h.sum()*0
        if self.arm=='QVAE':
            scores=global_scores.reshape(-1,self.factors,self.states);prob=scores.softmax(-1)
            regularizer=(prob*(prob.clamp_min(1e-12).log()+math.log(self.states))).sum(-1).mean()
            if z is None:
                if not self.training:raise ValueError('Generation/evaluation must declare prior or posterior')
                latent=F.gumbel_softmax(scores,tau=1.,hard=False)
            elif isinstance(z,str):
                if z!='posterior':raise ValueError('Unknown evaluation mode')
                latent=prob
            else:
                if z.shape!=prob.shape or (z<0).any() or not torch.isfinite(z).all() or not torch.allclose(z.sum(-1),torch.ones_like(prob[...,0])):raise ValueError('Invalid prior')
                latent=z
            context=latent.flatten(1)
        else:context=F.gelu(global_scores)
        decoded=self.decoder(h+self.global_projection(context)[:,None],src_key_padding_mask=~valid)
        return baseline.float()+self.output(decoded),regularizer


class FrozenEncoder:
    def __init__(self,path):
        from transformers import AutoModelForMaskedLM,AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForMaskedLM.from_pretrained(path,local_files_only=True,use_safetensors=True).cuda().eval()
        self.model.requires_grad_(False)
        self.aa_ids=self.tokenizer.convert_tokens_to_ids(list(AA))
        if len(set(self.aa_ids))!=20:raise ValueError('Tokenizer differs')

    @torch.inference_mode()
    def __call__(self,sequences,positions):
        tokens=self.tokenizer(sequences,padding=True,truncation=False,return_tensors='pt').to('cuda')
        width=max(map(len,sequences))
        if tokens.input_ids.shape[1]!=width+2 or not torch.equal(tokens.attention_mask.sum(1),torch.tensor([len(s)+2 for s in sequences],device='cuda')):raise ValueError('Tokenizer changed residue coordinates')
        valid=torch.arange(width,device='cuda')[None]<torch.tensor(list(map(len,sequences)),device='cuda')[:,None]
        mask=torch.zeros_like(valid)
        for i,(seq,pos) in enumerate(zip(sequences,positions)):
            if not pos or len(set(pos))!=len(pos) or min(pos)<0 or max(pos)>=len(seq):raise ValueError('Invalid mask')
            mask[i,pos]=True
        tokens.input_ids[:,1:width+1].masked_fill_(mask,self.tokenizer.mask_token_id)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=self.model.esm(**tokens).last_hidden_state
            baseline=self.model.lm_head(result)[:,1:width+1,self.aa_ids]
        return result[:,1:width+1],baseline,valid,mask

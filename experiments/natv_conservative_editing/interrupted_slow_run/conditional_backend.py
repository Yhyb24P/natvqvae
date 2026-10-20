"""Verified legacy SE / categorical editing on freshly masked current sequences."""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

from .intake import sha256_file
from .conservative_editing import AA


class ConditionalBackend:
    def __init__(self, source:Path, run:Path):
        self.source=source;self.run=run
        checks=json.loads((run/'inputs/source_checks.json').read_text())
        for name,item in checks['files'].items():
            if sha256_file(name)!=item['sha256']:raise ValueError('Pinned source differs: '+name)
        os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
        os.environ['HF_HUB_OFFLINE']='1';sys.dont_write_bytecode=True
        sys.path.insert(0,str(source/'src'))
        from nad_qvae.design.h2_generation_v1 import H2Retrieval,ROOT
        from nad_qvae.design.h2_model_v1 import H2RecoveryModel
        from nad_qvae.design.vnext_evaluation_contracts import FrozenHFMaskedEncoder
        if Path(ROOT).resolve()!=source.resolve():raise ValueError('Legacy source root differs')
        plan=json.loads((source/'data/processed/releases/natv_formal_training_data_v1/input_plan.json').read_text())
        self.retrieval=H2Retrieval(SimpleNamespace(plan=plan))
        self.frozen=FrozenHFMaskedEncoder(source/'nad_models/esm2',device='cuda')
        self.tokens=self.frozen.tokenizer.convert_tokens_to_ids(list(AA))
        self.model_class=H2RecoveryModel;self.models={}
        self.freeze=json.loads((source/'experiments/run_reports/natv_h2_architecture_generation_v1_attempt1/generation/checkpoint_freeze.json').read_text())
        self.loaded={};self.forward_sequences=0;self.retrieval_sequences=0
        # Record every actually imported legacy module, without importing its runners.
        self.module_hashes={str(Path(m.__file__).resolve()):sha256_file(m.__file__) for n,m in sys.modules.items()
                            if n.startswith('nad_qvae') and getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'}

    def model(self,arm,seed):
        if arm not in {'SE','QVAE'}:return None
        key=(arm,seed)
        if key not in self.models:
            architecture='no_latent' if arm=='SE' else 'categorical'
            entry=self.freeze[f'{architecture}/{seed}'];path=self.source/entry['path']
            if sha256_file(path)!=entry['sha256']:raise ValueError('Checkpoint differs')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['aa_token_ids']!=self.tokens:raise ValueError('Output tokenizer differs')
            model=self.model_class(aa_token_ids=self.tokens,architecture=architecture,hidden_dim=128,latent_dim=32,latent_states=8).cuda().eval()
            model.load_state_dict(saved['state_dict'],strict=True)
            self.models[key]=model
            self.loaded[f'{arm}/{seed}']={**entry,'parameters':sum(p.numel() for p in model.parameters()),
                                         'architecture':architecture,'training_epoch':saved['epoch']}
        return self.models[key]

    def features(self,sequences,positions,folder,with_encoder=True):
        unique=[];index={};inverse=[]
        for sequence,mask in zip(sequences,positions):
            key=(sequence,tuple(mask))
            if key not in index:index[key]=len(unique);unique.append(key)
            inverse.append(index[key])
        queries=[{'query_id':f'q{i}','current_sequence':sequence,'positions':list(mask),
                  'masked_sequence':''.join('X' if j in mask else a for j,a in enumerate(sequence))}
                 for i,(sequence,mask) in enumerate(unique)]
        profiles,receipts=self.retrieval.search(queries,folder)
        self.retrieval_sequences+=len(unique)
        evolution=torch.tensor(np.stack([profiles[f'q{i}'] for i in inverse]),device='cuda')
        encoded=None
        if with_encoder:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                encoded=self.frozen([q['current_sequence'] for q in queries],[q['positions'] for q in queries])
            self.forward_sequences+=len(unique)
            idx=torch.tensor(inverse,device='cuda')
            encoded={k:v.index_select(0,idx) if isinstance(v,torch.Tensor) else v for k,v in encoded.items()}
        return encoded,evolution,[receipts[i] for i in inverse],len(unique)

    @torch.no_grad()
    def logits(self,arm,model,encoded,evolution,z):
        if arm=='profile':return evolution[...,:20].clamp_min(1e-12).log(),None
        if arm=='ESM2':
            return encoded['logits'].index_select(-1,torch.tensor(self.tokens,device='cuda')).float(),None
        result=model(encoded,encoded['target_mask'],evolution=evolution,**({'latent_override':z} if arm=='QVAE' else {}))
        if arm=='QVAE' and result['latent_source']!='external_override':raise ValueError('Posterior entered generation')
        return result['logits'],result

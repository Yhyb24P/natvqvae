"""执行已绑定预算的 NatV 条件类别模型研究。"""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode=True
from natvqvae.conditional_data import prepare_features, status
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--from-calibration', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--finish-only', action='store_true')
    args=parser.parse_args();run=args.run.resolve()
    if not run.is_relative_to(ROOT/'experiments'):
        raise ValueError('Run must be inside experiments/')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    plan=json.loads((run/'plan.json').read_text())
    if plan['task']!='NatV_conditional_categorical_joint_generation':
        raise ValueError('Historical runs must use their recorded Git revision')
    if not args.finish_only:
        prepare_features(run,plan)
    if args.prepare_only:
        status(run,'FROZEN_FEATURES_READY');return
    from natvqvae.conditional_study import run_study
    if args.finish_only:
        run_study(run,plan,finish_only=True)
    else:
        run_study(run,plan,calibrate_only=args.calibrate_only,
                  from_calibration=args.from_calibration,train_only=args.train_only)

if __name__=='__main__':main()

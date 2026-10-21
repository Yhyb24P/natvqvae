"""从既有条件生成训练记录绘图并核对所选权重，不运行模型推理。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / 'experiments'):
        raise ValueError('报告目录须位于 experiments/')
    inputs = [run / name for name in (
        'training_history.json', 'plan.json', 'checkpoint_manifest.json', 'inputs/identity.json')]
    history, plan, manifest, identity = [json.loads(p.read_text()) for p in inputs]
    epochs = list(range(1, plan['training']['epochs'] + 1))
    seeds = plan['seeds']
    fractions = plan['features']['mask_fractions']
    if len(fractions) != 3:
        raise ValueError('此图对应三个轮换掩码面板')
    indexed = {(r['key'], r['epoch']): r for r in history}
    expected = {(f'{arm}/{seed}', epoch)
                for arm in plan['arms'] for seed in seeds for epoch in epochs}
    if len(indexed) != len(history) or set(indexed) != expected:
        raise ValueError('训练记录重复或不完整')
    metrics = ['training_CE', 'training_KL', 'prior_CE', 'posterior_CE']
    arrays = {arm: {metric: np.array([
        [indexed[f'{arm}/{seed}', epoch][metric] for epoch in epochs]
        for seed in seeds]) for metric in metrics} for arm in plan['arms']}
    if not all(np.isfinite(a).all() for values in arrays.values() for a in values.values()):
        raise ValueError('存在非有限指标')

    # 对照运行时源码身份；不修改历史 identity 或哈希快照。
    reviewed = {}
    for relative in ['scripts/run_conditional_study.py', 'src/natvqvae/conditional_generation.py']:
        path = ROOT / relative
        recorded = identity['sha256'][str(path)]
        actual = digest(path)
        if actual != recorded:
            raise ValueError(f'当前源码与运行快照不同，须重新审阅：{relative}')
        reviewed[relative] = actual
    selections = {}
    for key in sorted(manifest):
        entry = manifest[key]
        path = ROOT / entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError(f'权重校验失败：{key}')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        best = min((indexed[key, e] for e in epochs), key=lambda r: r['prior_CE'])
        if (saved['epoch'] != best['epoch']
                or saved['validation']['prior_CE'] != best['prior_CE']
                or entry['selected_prior_CE'] != best['prior_CE']
                or saved['arm'] != key.split('/')[0]
                or saved['seed'] != int(key.split('/')[1])):
            raise ValueError(f'所选权重与记录不一致：{key}')
        selections[key] = {'epoch': saved['epoch'], 'validation_CE': best['prior_CE'],
                           'sha256': entry['sha256']}
        del saved

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    colors = {'QVAE': '#1767a6', 'deterministic': '#c95621'}
    labels = {'QVAE': 'QVAE', 'deterministic': '确定性对照'}
    x = np.array(epochs)

    def curve(ax, arm, values, selected=None):
        selected = np.ones(len(x), dtype=bool) if selected is None else selected
        data = values[:, selected]
        ax.plot(x[selected], data.mean(0), color=colors[arm], label=labels[arm],
                marker='o', markersize=3, linewidth=1.6)
        ax.fill_between(x[selected], data.min(0), data.max(0), color=colors[arm], alpha=.18)

    for arm, values in arrays.items():
        curve(axes[0, 0], arm, values['prior_CE'])
        for panel in range(3):
            curve(axes[1, panel], arm, values['training_CE'], (x - 1) % 3 == panel)
    axes[0, 0].set(title='验证 CE：每轮覆盖三个面板', ylabel='组分平衡 CE')
    for epoch in sorted({s['epoch'] for s in selections.values()}):
        axes[0, 0].axvline(epoch, color='#555555', linestyle='--', linewidth=1,
                           label=f'所选权重：第 {epoch} 轮')
    curve(axes[0, 1], 'QVAE', arrays['QVAE']['training_KL'])
    axes[0, 1].set(title='QVAE 训练 KL：每轮一个面板', ylabel='KL（类别因子平均，未乘权重）', yscale='log')
    curve(axes[0, 2], 'QVAE', arrays['QVAE']['prior_CE'] - arrays['QVAE']['posterior_CE'])
    axes[0, 2].axhline(0, color='#555555', linewidth=.8)
    axes[0, 2].set(title='QVAE 验证接口 CE 差', ylabel='固定先验采样 CE − 类别概率向量 CE')
    for panel, fraction in enumerate(fractions):
        axes[1, panel].set(title=f'训练 CE：{fraction:.0%} 掩码面板', ylabel='训练日志 CE（批次均值）')
    for ax in axes.flat:
        ax.set_xlabel('训练轮次')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('条件生成训练诊断｜线为三个种子均值，阴影为最小值–最大值（非置信区间）', fontsize=15)
    fig.savefig(run / 'training_curves.png', dpi=180)
    fig.savefig(run / 'training_curves.pdf')
    plt.close(fig)

    summary = {}
    summary_epochs = sorted({epochs[0], epochs[-1]} | {s['epoch'] for s in selections.values()})
    for arm, values in arrays.items():
        summary[arm] = {metric: {str(epoch): {
            'mean': float(a[:, epoch - 1].mean()),
            'min': float(a[:, epoch - 1].min()), 'max': float(a[:, epoch - 1].max())}
            for epoch in summary_epochs} for metric, a in values.items()}
    result = {
        'analysis_git_revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'training_git_revision': identity['git_revision'],
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        'analysis_code_sha256': digest(Path(__file__)), 'reviewed_source_sha256': reviewed,
        'plotting_versions': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
                              'torch': torch.__version__},
        'records': len(history), 'verified_checkpoints': selections,
        'training_panel_fraction_by_epoch': {str(e): fractions[(e - 1) % 3] for e in epochs},
        'validation_per_panel': 'UNKNOWN：历史日志未保留逐输入或逐面板验证 CE',
        'latent_usage': 'UNKNOWN：KL 与验证 CE 差不能独立确认 z 的实际使用程度',
        'summary': summary,
        'figure_sha256': {name: digest(run / name) for name in ['training_curves.png', 'training_curves.pdf']},
    }
    (run / 'training_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'records': len(history), 'selected_epochs': {k: v['epoch'] for k, v in selections.items()},
                      'figures': ['training_curves.png', 'training_curves.pdf']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

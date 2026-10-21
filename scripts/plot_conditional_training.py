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
import pandas as pd

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
    epochs = list(range(1, max(r['epoch'] for r in history) + 1))
    seeds = plan['seeds']
    fractions = plan['features']['mask_fractions']
    if len(fractions) != 3:
        raise ValueError('此图对应三个轮换掩码面板')
    indexed = {(r['key'], r['epoch']): r for r in history}
    expected = {(f'{arm}/{seed}', epoch)
                for arm in plan['arms'] for seed in seeds
                for epoch in range(1, max(r['epoch'] for r in history if r['key']==f'{arm}/{seed}')+1)}
    if len(indexed) != len(history) or set(indexed) != expected:
        raise ValueError('训练记录重复或不完整')
    metrics = ['training_CE', 'training_KL', 'prior_CE', 'posterior_CE']
    arrays = {arm: {metric: np.array([
        [indexed.get((f'{arm}/{seed}', epoch), {}).get(metric, np.nan) for epoch in epochs]
        for seed in seeds]) for metric in metrics} for arm in plan['arms']}
    if not all(np.isfinite(r[m]) for r in history for m in metrics):
        raise ValueError('存在非有限指标')

    # 对照运行时源码身份；不修改历史 identity 或哈希快照。
    reviewed = {}
    for relative in ['scripts/run_conditional_study.py', 'src/natvqvae/conditional_generation.py']:
        path = ROOT / relative
        recorded = identity['sha256'][str(path)]
        actual = digest(path)
        if actual != recorded:
            snapshot = subprocess.check_output(['git', 'show', f"{identity['git_revision']}:{relative}"], cwd=ROOT)
            if hashlib.sha256(snapshot).hexdigest() != recorded:
                raise ValueError(f'Git 中无法恢复运行时源码：{relative}')
        reviewed[relative] = recorded
    selections = {}
    for key in sorted(manifest):
        entry = manifest[key]
        path = ROOT / entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError(f'权重校验失败：{key}')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        best = min((r for r in history if r['key']==key), key=lambda r: r['prior_CE'])
        if (saved['epoch'] != best['epoch']
                or saved['validation']['prior_CE'] != best['prior_CE']
                or entry['selected_prior_CE'] != best['prior_CE']
                or saved['arm'] != key.split('/')[0]
                or saved['seed'] != int(key.split('/')[1])):
            raise ValueError(f'所选权重与记录不一致：{key}')
        selections[key] = {'epoch': saved['epoch'], 'validation_CE': best['prior_CE'],
                           'sha256': entry['sha256']}
        del saved

    validation_audit = None
    if plan['task'] == 'target_conditioned_posterior_training':
        raw = ROOT / 'data/results' / run.name / 'raw/validation_records.jsonl.gz'
        frame = pd.read_json(raw, lines=True)
        if frame.duplicated(['key', 'epoch', 'id']).any():
            raise ValueError('逐输入验证记录重复')
        expected_rows = plan['validation_population'] * len(fractions)
        groups = frame.groupby(['key', 'epoch'])
        if set(groups.groups) != set(indexed) or not groups.size().eq(expected_rows).all():
            raise ValueError('逐输入验证记录不完整')
        for (key, epoch), part in groups:
            record = indexed[key, epoch]
            for column, metric in [('CE', 'prior_CE'), ('posterior_CE', 'posterior_CE'),
                                   ('posterior_sample_CE', 'posterior_sample_CE'), ('KL', 'validation_KL')]:
                actual = part.groupby('component')[column].mean().mean()
                if not np.isclose(actual, record[metric], atol=1e-10, rtol=0):
                    raise ValueError(f'验证汇总不符：{key} {epoch} {metric}')
            for panel, panel_rows in part.groupby('panel'):
                actual = panel_rows.groupby('component').CE.mean().mean()
                if not np.isclose(actual, record['panels'][str(panel)]['prior_CE'], atol=1e-10, rtol=0):
                    raise ValueError('逐面板验证汇总不符')
        fit_status = json.loads((run / 'fit_status.json').read_text())
        for key, stop in fit_status['stopped'].items():
            rows = sorted((r for r in history if r['key'] == key), key=lambda r: r['epoch'])
            best_ce, stale = float('inf'), 0
            for i, row in enumerate(rows):
                if row['prior_CE'] < best_ce:
                    best_ce, stale = row['prior_CE'], 0
                else:
                    stale += 1
                if stale >= plan['training']['patience'] and i != len(rows) - 1:
                    raise ValueError('达到早停条件后仍继续训练')
            if stop['reason'] == 'early_stopping' and stale != plan['training']['patience']:
                raise ValueError('早停计数不符')
            if rows[-1]['epoch'] != stop['epoch']:
                raise ValueError('停止轮次不符')
        validation_audit = {'records': len(frame), 'groups': len(groups), 'raw_sha256': digest(raw),
                            'component_and_panel_aggregation_verified': True,
                            'early_stopping_verified': True}

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    colors = {'QVAE': '#1767a6', 'deterministic': '#c95621'}
    labels = {'QVAE': 'QVAE', 'deterministic': '确定性对照'}
    x = np.array(epochs)

    def curve(ax, arm, values, selected=None):
        selected = np.ones(len(x), dtype=bool) if selected is None else selected
        selected = selected & np.isfinite(values).any(0)
        data = values[:, selected]
        ax.plot(x[selected], np.nanmean(data, axis=0), color=colors[arm], label=labels[arm],
                marker='o', markersize=3, linewidth=1.6)
        ax.fill_between(x[selected], np.nanmin(data, axis=0), np.nanmax(data, axis=0), color=colors[arm], alpha=.18)

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
    fig.suptitle('条件生成训练诊断｜线为仍有记录的种子均值，阴影为最小值–最大值（非置信区间）', fontsize=15)
    fig.savefig(run / 'training_curves.png', dpi=180)
    fig.savefig(run / 'training_curves.pdf')
    plt.close(fig)

    figure_names = ['training_curves.png', 'training_curves.pdf']
    if all('panels' in r for r in history):
        panel_fig, panel_axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for panel, ax in enumerate(panel_axes):
            for arm in plan['arms']:
                values = np.array([[indexed.get((f'{arm}/{seed}', e), {}).get('panels', {}).get(str(panel), {}).get('prior_CE', np.nan)
                                    for e in epochs] for seed in seeds])
                curve(ax, arm, values)
            ax.set(title=f'{fractions[panel]:.0%} 掩码验证 CE', xlabel='训练轮次', ylabel='组分平衡先验 CE')
            ax.grid(alpha=.2);ax.legend()
        panel_fig.suptitle('每轮覆盖全部面板｜线为有记录种子均值，阴影为极差；早停后种子数可减少')
        panel_fig.savefig(run / 'validation_by_mask.png', dpi=180);plt.close(panel_fig)
        figure_names.append('validation_by_mask.png')
    summary = {}
    summary_epochs = sorted({epochs[0], epochs[-1]} | {s['epoch'] for s in selections.values()})
    for arm, values in arrays.items():
        summary[arm] = {metric: {str(epoch): {
            'mean': float(np.nanmean(a[:, epoch - 1])),
            'min': float(np.nanmin(a[:, epoch - 1])), 'max': float(np.nanmax(a[:, epoch - 1]))}
            for epoch in summary_epochs if np.isfinite(a[:, epoch-1]).any()} for metric, a in values.items()}
    result = {
        'analysis_git_revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'training_git_revision': identity['git_revision'],
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        'analysis_code_sha256': digest(Path(__file__)), 'reviewed_source_sha256': reviewed,
        'plotting_versions': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
                              'torch': torch.__version__},
        'records': len(history), 'verified_checkpoints': selections,
        'validation_audit': validation_audit,
        'training_panel_fraction_by_epoch': {str(e): fractions[(e - 1) % 3] for e in epochs},
        'validation_per_panel': 'validation_by_mask.png' if len(figure_names)>2 else 'UNKNOWN：历史日志未保留逐输入或逐面板验证 CE',
        'seed_counts_by_epoch': {arm: np.isfinite(values['prior_CE']).sum(0).tolist() for arm, values in arrays.items()},
        'latent_usage': 'UNKNOWN：KL 与验证 CE 差不能独立确认 z 的实际使用程度',
        'summary': summary,
        'figure_sha256': {name: digest(run / name) for name in figure_names},
    }
    (run / 'training_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'records': len(history), 'selected_epochs': {k: v['epoch'] for k, v in selections.items()},
                      'figures': figure_names}, ensure_ascii=False))


if __name__ == '__main__':
    main()

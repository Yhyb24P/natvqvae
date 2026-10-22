# NatV QVAE

KVP40 NatV 是 341 aa 的 NMNAT–Nudix 双域蛋白。本仓库只保留当前可运行的 NatV 序列生成研究：用天然演化序列训练条件 Categorical QVAE，从条件先验生成完整 KVP40 变体，并用计算证据筛选候选。活性、底物特异性、kcat/Km 与 NAD⁺ 净积累均为 UNKNOWN。

## 结构

```
config/        target、raw source 与训练/采样配置
data/raw/      原始序列、结构与参考文件（不含清洗数据或运行产物）
src/natvqvae/  data、model、train、generate、evaluate、cli
tests/         现役功能测试
docs/          RESEARCH_PLAN.md、RESULTS.md
```

## 安装与检查

```bash
python -m pip install -e ".[dev]"
natvqvae doctor
pytest -q
```

`config/sources.yaml` 指定原始群体、精确 split、冻结 ESM-2 路径和当前 QVAE checkpoint 目录。模型权重位于仓库外，不复制进 Git：

- ESM-2：`/mnt/e/wsl_cold/nad_qvae_v2/nad_models/esm2`
- QVAE：`/home/yhshy/models/natvqvae/QVAE/seed8301/best.pt` 等 5 个 seed
- 校验值：`config/checkpoints.sha256`、`config/inputs.sha256`

## 运行

训练、生成和评价都接受显式配置与输出路径；输出必须放在仓库外，避免把运行产物写入 `data/`。

```bash
natvqvae train \
  --config config/study.yaml --sources config/sources.yaml --target config/target.yaml \
  --output /path/to/run/train

natvqvae generate \
  --checkpoint /home/yhshy/models/natvqvae/QVAE/seed8301/best.pt \
  --output /path/to/run/generate

natvqvae evaluate \
  --checkpoint /home/yhshy/models/natvqvae/QVAE/seed8301/best.pt \
  --split validation --output /path/to/run/evaluate
```

生成只从条件先验采样，使用 checkpoint 的模型参数，不依赖训练流程或研究期假设。评价只读取 validation；test split 只保留 ID，不加载序列。

科学目标、数据边界与评价原则见 [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)；已完成结论与负结果见 [docs/RESULTS.md](docs/RESULTS.md)。

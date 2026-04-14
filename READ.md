# Weibo 情感分类模型详细说明（READ.md）

## 1. 模型概览

本项目在单卡 RTX 4090（24GB）上，基于 `Qwen/Qwen2.5-0.5B` 进行 LoRA 微调，完成中文评论二分类任务。

- 任务类型: 情感二分类（Sequence Classification）
- 标签定义:
  - `0` = negative（差评）
  - `1` = positive（好评）
- 数据来源: `dirtycomputer/weibo_senti_100k`
- 训练框架: PyTorch + Transformers + PEFT(LoRA)

## 2. 最终核心结果（测试集）

以下为最终模型在测试集上的结果（已完成训练并验证）：

- `test_loss`: `0.042604777961969376`
- `test_accuracy`: `0.9824152012667722`
- `test_f1`: `0.982101959453728`
- `test_precision`: `0.9998272884283247`
- `test_recall`: `0.9649941656942824`
- `roc_auc`: `0.9982334694671334`

训练总时长（3 epoch）：

- `train_runtime`: `2521.1603` 秒（约 42 分钟）
- `train_steps_per_second`: `3.57`

## 3. 混淆矩阵与分类报告

混淆矩阵（真实标签为行，预测标签为列）：

| 真\预测 | negative | positive |
|---|---:|---:|
| negative | 5999 | 1 |
| positive | 210 | 5789 |

分类报告（test）：

- negative:
  - precision: `0.9661781285231116`
  - recall: `0.9998333333333334`
  - f1-score: `0.9827176672946187`
  - support: `6000`
- positive:
  - precision: `0.9998272884283247`
  - recall: `0.9649941656942824`
  - f1-score: `0.982101959453728`
  - support: `5999`
- macro avg f1: `0.9824098133741734`
- weighted avg f1: `0.9824098390308048`

结果解读：

- positive 的 precision 极高，表示模型判为好评时非常可靠。
- positive 的 recall 略低于 precision，说明仍存在一部分好评被判为差评（210 条）。
- 整体 accuracy 与 f1 均达到 98%+，适合评论好坏快速筛查场景。

## 4. 数据集与拆分细节

原始数据：

- 数据集: `dirtycomputer/weibo_senti_100k`
- 样本数: `119,988`
- 字段:
  - `review`: 评论文本
  - `label`: 标签

拆分策略：分层抽样（Stratified Split），比例 `8:1:1`。

- train: `95,990`
- validation: `11,999`
- test: `11,999`

标签分布：

- train: `{0: 47996, 1: 47994}`
- validation: `{0: 5999, 1: 6000}`
- test: `{0: 6000, 1: 5999}`

## 5. 训练配置详情

基座模型：

- `Qwen/Qwen2.5-0.5B`

LoRA 配置：

- `r`: `16`
- `lora_alpha`: `32`
- `lora_dropout`: `0.05`
- `target_modules`: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `modules_to_save`: `score`

关键训练参数：

- `max_length`: `256`
- `per_device_train_batch_size`: `16`
- `per_device_eval_batch_size`: `32`
- `gradient_accumulation_steps`: `2`
- `learning_rate`: `2e-4`
- `num_train_epochs`: `3`
- `weight_decay`: `0.01`
- `warmup_ratio`: `0.06`
- `fp16`: `true`
- `seed`: `42`

可训练参数比例：

- trainable params: `2,164,480`
- all params: `496,199,040`
- trainable%: `0.4362%`

## 6. 训练与评估环境

- OS: Linux
- GPU: NVIDIA GeForce RTX 4090 24GB
- Python: 3.10
- CUDA: 可用
- 包管理源: 清华镜像（pip）
- 模型下载端点: `HF_ENDPOINT=https://hf-mirror.com`

## 7. 仓库中的模型与必要文件清单

核心模型产物（已包含在仓库）：

- `outputs/qwen2.5-0.5b-weibo-senti/final/adapter_model.safetensors`
- `outputs/qwen2.5-0.5b-weibo-senti/final/adapter_config.json`
- `outputs/qwen2.5-0.5b-weibo-senti/final/tokenizer.json`
- `outputs/qwen2.5-0.5b-weibo-senti/final/tokenizer_config.json`
- `outputs/qwen2.5-0.5b-weibo-senti/final/vocab.json`
- `outputs/qwen2.5-0.5b-weibo-senti/final/merges.txt`
- `outputs/qwen2.5-0.5b-weibo-senti/final/added_tokens.json`
- `outputs/qwen2.5-0.5b-weibo-senti/final/special_tokens_map.json`
- `outputs/qwen2.5-0.5b-weibo-senti/final/training_args.bin`

训练记录与指标：

- `outputs/qwen2.5-0.5b-weibo-senti/test_metrics.json`
- `outputs/train.log`
- `outputs/qwen2.5-0.5b-weibo-senti/runs/`（TensorBoard 日志）

代码与配置：

- `src/`
- `scripts/`
- `configs/train_qwen2_0_5b.json`
- `requirements.txt`
- `README.md`
- `READ.md`（本文档）

## 8. 如何推理与部署

### 8.1 命令行单条预测

```bash
python3 src/predict.py \
  --model_dir outputs/qwen2.5-0.5b-weibo-senti/final \
  --text "这个商品非常好用，下次还会买"
```

### 8.2 FastAPI 服务

```bash
bash scripts/start_api.sh outputs/qwen2.5-0.5b-weibo-senti/final 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

预测调用：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"包装破损，体验很差"}'
```

## 9. 如何复现实验

1. 安装依赖（清华源）

```bash
bash scripts/install_deps.sh
```

2. 数据准备（分层拆分）

```bash
HF_ENDPOINT=https://hf-mirror.com python3 src/prepare_data.py \
  --dataset_repo dirtycomputer/weibo_senti_100k \
  --output_dir data/processed \
  --seed 42
```

3. 启动训练

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/start_train.sh
```

4. 评估测试集

```bash
HF_ENDPOINT=https://hf-mirror.com python3 src/evaluate.py \
  --model_dir outputs/qwen2.5-0.5b-weibo-senti/final \
  --data_dir data/processed \
  --max_length 256
```

5. 可视化训练过程

```bash
bash scripts/start_tensorboard.sh outputs/qwen2.5-0.5b-weibo-senti/runs 6006
```

## 10. 适用场景与限制

适用场景：

- 中文电商评论快速正负向筛查
- 风险评论预警前置过滤
- 运营看板情感统计

限制与注意：

- 本仓库保存的是 LoRA 适配器，不是基座模型完整权重。
- 推理时会自动读取基座模型 `Qwen/Qwen2.5-0.5B`。
- 对讽刺、反问、上下文依赖评论仍可能误判。
- 若用于生产，建议结合业务词典、规则引擎与在线监控。

## 11. 版本与可追溯信息

- 项目仓库分支: `main`
- 训练配置文件: `configs/train_qwen2_0_5b.json`
- 核心指标文件: `outputs/qwen2.5-0.5b-weibo-senti/test_metrics.json`
- 训练日志: `outputs/train.log`

以上信息可用于审计、复现实验和后续持续迭代。

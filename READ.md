# Weibo 情感分类模型详细说明（READ.md）

## 1. 项目目标

本项目基于 Hugging Face 数据集 `dirtycomputer/weibo_senti_100k` 训练一个中文二分类情感模型，用于判断评论是正向（好评）还是负向（差评）。

- 任务类型: Sequence Classification
- 标签定义:
  - `0`: negative
  - `1`: positive

## 2. 训练环境

- 操作系统: Linux
- GPU: NVIDIA GeForce RTX 4090 (24GB)
- 显卡数量: 1
- Python: 3.10
- 深度学习框架: PyTorch + Transformers + PEFT (LoRA)

## 3. 数据集与拆分

- 原始数据集: `dirtycomputer/weibo_senti_100k`
- 原始样本总数: `119,988`
- 字段:
  - `review`: 文本
  - `label`: 标签

数据拆分策略（分层抽样）:
- 训练集: 80% (`95,990`)
- 验证集: 10% (`11,999`)
- 测试集: 10% (`11,999`)

标签分布（拆分后）:
- train: `{0: 47996, 1: 47994}`
- validation: `{0: 5999, 1: 6000}`
- test: `{0: 6000, 1: 5999}`

## 4. 模型与训练配置

基座模型:
- `Qwen/Qwen2.5-0.5B`

训练方式:
- LoRA 微调（仅训练小部分参数）
- 训练可学习参数占比约 `0.4362%`

关键超参数:
- `max_length`: 256
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `gradient_accumulation_steps`: 2
- `learning_rate`: 2e-4
- `num_train_epochs`: 3
- `weight_decay`: 0.01
- `warmup_ratio`: 0.06
- `fp16`: true
- `seed`: 42

LoRA 参数:
- `r`: 16
- `alpha`: 32
- `dropout`: 0.05
- `target_modules`: `q_proj,k_proj,v_proj,o_proj`

## 5. 训练结果（测试集）

最终测试指标:
- `test_loss`: 0.042604777961969376
- `test_accuracy`: 0.9824152012667722
- `test_f1`: 0.982101959453728
- `test_precision`: 0.9998272884283247
- `test_recall`: 0.9649941656942824
- `epoch`: 3.0

训练总耗时:
- `train_runtime`: 2521.1603 秒（约 42 分钟）

## 6. 产物目录说明

- `outputs/qwen2.5-0.5b-weibo-senti/final/adapter_model.safetensors`: LoRA 权重
- `outputs/qwen2.5-0.5b-weibo-senti/final/adapter_config.json`: LoRA 配置
- `outputs/qwen2.5-0.5b-weibo-senti/final/tokenizer.json` 等: 分词器文件
- `outputs/qwen2.5-0.5b-weibo-senti/test_metrics.json`: 测试集指标
- `outputs/train.log`: 训练日志

## 7. 推理与部署

### 7.1 单句预测

```bash
python3 src/predict.py \
  --model_dir outputs/qwen2.5-0.5b-weibo-senti/final \
  --text "这个商品非常好用，下次还会买"
```

### 7.2 API 服务

```bash
bash scripts/start_api.sh outputs/qwen2.5-0.5b-weibo-senti/final 8000
```

调用示例:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"包装破损，体验很差"}'
```

## 8. 训练可视化

TensorBoard 启动:

```bash
bash scripts/start_tensorboard.sh outputs/qwen2.5-0.5b-weibo-senti/runs 6006
```

浏览器访问:
- `http://<服务器IP>:6006`

## 9. 复现实验步骤

1. 安装依赖（清华源）

```bash
bash scripts/install_deps.sh
```

2. 准备数据

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

4. 测试评估

```bash
python3 src/evaluate.py \
  --model_dir outputs/qwen2.5-0.5b-weibo-senti/final \
  --data_dir data/processed \
  --max_length 256
```

## 10. 注意事项

- 本仓库保存的是 LoRA 适配器权重，不是完整基座模型全量权重。
- 推理时会自动从 Hugging Face 拉取基座模型，需保持网络可访问（推荐设置 `HF_ENDPOINT=https://hf-mirror.com`）。
- 若用于生产环境，建议增加:
  - 数据清洗规则
  - 更丰富测试集
  - 线上监控与漂移检测

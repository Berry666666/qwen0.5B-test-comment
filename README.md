# Weibo 中文电商评论情感分类训练与部署

本项目可在单卡 RTX4090 (24GB) 上训练一个中文情感分类模型，区分评论好评/差评。

## 0. 已完成训练结果

- 测试集 `accuracy`: `0.9824`
- 测试集 `f1`: `0.9821`
- 测试集 `precision`: `0.9998`
- 测试集 `recall`: `0.9650`
- 训练时长: 约 `42` 分钟（3 个 epoch）

详细说明见 `READ.md`，指标文件见 `outputs/qwen2.5-0.5b-weibo-senti/test_metrics.json`。

## 1. 目录说明

- `requirements.txt`: 依赖清单
- `configs/train_qwen2_0_5b.json`: 训练配置
- `src/prepare_data.py`: 下载并按 8:1:1 划分 train/validation/test
- `src/train.py`: 使用 Qwen2.5-0.5B + LoRA 训练分类模型
- `src/evaluate.py`: 在测试集上评估 accuracy/f1/precision/recall
- `src/predict.py`: 命令行单条预测
- `src/app.py`: FastAPI 推理服务
- `scripts/install_deps.sh`: 使用清华源安装依赖
- `scripts/start_train.sh`: 数据准备 + 训练
- `scripts/start_tensorboard.sh`: TensorBoard 实时可视化
- `scripts/start_api.sh`: 启动推理 API

## 2. 安装依赖（清华源）

```bash
cd /root/shared-nvme/weibo-senti-train
bash scripts/install_deps.sh
```

## 3. 开始训练

```bash
cd /root/shared-nvme/weibo-senti-train
bash scripts/start_train.sh
```

## 4. 可视化训练过程

新开一个终端执行：

```bash
cd /root/shared-nvme/weibo-senti-train
bash scripts/start_tensorboard.sh outputs/qwen2.5-0.5b-weibo-senti/runs 6006
```

然后在浏览器打开：`http://<服务器IP>:6006`

## 5. 测试集评估

```bash
cd /root/shared-nvme/weibo-senti-train
python3 src/evaluate.py \
  --model_dir outputs/qwen2.5-0.5b-weibo-senti/final \
  --data_dir data/processed \
  --max_length 256
```

## 6. 单条预测

```bash
cd /root/shared-nvme/weibo-senti-train
python3 src/predict.py \
  --model_dir outputs/qwen2.5-0.5b-weibo-senti/final \
  --text "这个商品质量很好，物流也很快"
```

## 7. 部署 API

```bash
cd /root/shared-nvme/weibo-senti-train
bash scripts/start_api.sh outputs/qwen2.5-0.5b-weibo-senti/final 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

预测：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"包装太差了，体验很不好"}'
```

## 8. GitHub 上传建议保留文件

- `README.md`
- `READ.md`
- `requirements.txt`
- `configs/`
- `src/`
- `scripts/`
- `outputs/qwen2.5-0.5b-weibo-senti/final/`
- `outputs/qwen2.5-0.5b-weibo-senti/test_metrics.json`
- `outputs/train.log`

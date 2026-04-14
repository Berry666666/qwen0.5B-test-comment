# Ollama 部署说明（重要）

## 1. 兼容性结论

当前仓库里训练出的模型是 **Qwen2.5-0.5B 的序列分类 LoRA 适配器**（用于 `AutoModelForSequenceClassification`）。

Ollama 主要面向**生成式聊天模型**，不能直接加载这个分类头训练产物作为同等推理图，因此：
- 不能 100% 直接把本项目 `final` 下这套分类模型原样塞给 Ollama 使用。

## 2. 你在 Ollama 里可运行的替代方案

你可以在本地 Ollama 创建一个“提示词分类模型”（基于 `qwen2.5:0.5b`），用于可视化聊天式分类：

```bash
cd ollama
ollama create qwen-comment-senti -f Modelfile
ollama run qwen-comment-senti
```

输入示例：

```text
这款商品真的很差，做工粗糙还贵
```

它会输出 JSON，如：

```json
{"label":"negative","reason":"评论表达了明显不满"}
```

## 3. 真正使用你训练权重的可视化方案（推荐）

请用本项目的 Gradio 可视化界面（这套才是你训练出的模型）：

```bash
bash scripts/start_gradio_ui.sh outputs/qwen2.5-0.5b-weibo-senti/final 7860
```

浏览器打开：

```text
http://<服务器IP>:7860
```

## 4. 如果你必须“严格 Ollama + 你的训练权重”

需要把训练任务改造成**生成式微调**（Causal LM），再导出 Ollama 兼容权重格式（通常是 GGUF 或 Ollama 支持的导入路径）。
这属于新的训练流程，不是当前这次分类头训练产物能直接转换得到。

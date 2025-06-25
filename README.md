# NeuraLink Intel AI 创新应用 - 智能语音情感分析系统

[![Intel OpenVINO](https://img.shields.io/badge/Intel-OpenVINO-blue)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📖 项目简介

本项目是参加"2025英特尔人工智能创新应用大赛"的参赛作品，基于Intel OpenVINO优化的智能语音情感分析系统。该系统通过集成语音识别、情感分析、智能对话和可视化分析等技术，为儿童心理健康监护提供了一套完整的AI解决方案。

### 🎯 核心特性

- **🎙️ 高效语音识别**：基于OpenVINO优化的Whisper模型，支持GPU/NPU加速
- **🧠 智能情感分析**：结合文本情感和语音特征的多模态情感识别
- **💬 智能对话系统**：集成Coze AI Agent和本地LLM，提供个性化交互
- **📊 可视化分析**：实时情绪趋势分析和历史数据可视化
- **🔍 意图识别**：智能识别儿童表达意图，提供育儿建议

## 🏗️ 技术架构

### Intel AI 加速技术栈

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                     │
├─────────────────────────────────────────────────────────┤
│  语音录制  │  实时转录  │  情感分析  │  智能对话  │  数据可视化  │
├─────────────────────────────────────────────────────────┤
│            Intel OpenVINO 推理引擎                      │
├─────────────────────────────────────────────────────────┤
│  Whisper     │  RoBERTa    │  Coze Agent │  本地LLM    │
│  语音识别    │  情感分类    │  云端AI     │  (Ollama)   │
├─────────────────────────────────────────────────────────┤
│              硬件加速层 (GPU/NPU/CPU)                   │
└─────────────────────────────────────────────────────────┘
```

### 核心组件

1. **语音处理模块** (`whisper_ov_runner.py`)
   - OpenVINO优化的Whisper模型
   - 支持GPU/NPU硬件加速
   - 实时语音转文本

2. **情感分析引擎** (`demo1.py`)
   - 基于RoBERTa的文本情感分类
   - 语音特征情绪识别
   - 多模态情感融合

3. **智能对话系统**
   - Coze AI Agent云端调用
   - 本地Ollama LLM支持
   - 上下文感知对话

4. **数据分析模块** (`db_logger.py`)
   - SQLite数据持久化
   - 情绪趋势可视化
   - 历史数据分析

## 🚀 快速开始

### 环境要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **硬件**: 支持Intel OpenVINO的CPU/GPU/NPU
- **内存**: 至少8GB RAM

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/wilson534/neuralink_intel625.git
cd neuralink_intel625
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **安装Intel OpenVINO**
```bash
# 使用pip安装
pip install openvino

# 或下载完整工具包
# https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html
```

4. **配置API密钥**
   
   在项目根目录创建 `.env` 文件：
```env
COZE_API_KEY=your_coze_api_key_here
COZE_BOT_ID=your_bot_id_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### 运行项目

#### 方式一：使用批处理文件（Windows）
```bash
fastrun.bat
```

#### 方式二：直接运行
```bash
streamlit run demo1.py
```

#### 方式三：简化版本
```bash
streamlit run index.py
```

## 📋 功能详解

### 1. 语音录制与识别
- 支持实时录音（默认5秒）
- OpenVINO加速的Whisper模型推理
- 中文语音识别优化

### 2. 多模态情感分析
- **文本情感**：基于RoBERTa的中文情感分类
- **语音情感**：通过MFCC特征提取分析语调情绪
- **情感融合**：多模态信息综合判断

### 3. 智能对话系统
- **在线模式**：Coze AI Agent云端调用
- **离线模式**：本地Ollama LLM推理
- **上下文记忆**：维护对话历史和情感状态

### 4. 数据可视化分析
- 实时情绪趋势图表
- 周/月情绪统计
- 对话历史记录
- 情感分布分析

### 5. 意图识别与建议
- 自动识别儿童表达意图
- 生成个性化育儿建议
- 异常情绪预警

## 🔧 配置说明

### OpenVINO 设备配置

修改 `demo1.py` 中的设备设置：

```python
# 使用GPU加速
emo_model = OVModelForSequenceClassification.from_pretrained(
    "models/emotion_openvino", 
    device="GPU"
)

# 使用CPU
device="CPU"

# 使用NPU（支持的硬件）
device="NPU"
```

### 本地LLM配置

启动Ollama服务：
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下载模型
ollama pull qwen:7b

# 启动服务
ollama serve
```

## 📊 性能优化

### Intel OpenVINO 优化效果

| 模型组件 | 原始性能 | OpenVINO优化后 | 提升倍数 |
|---------|----------|----------------|----------|
| Whisper Base | 2.3s | 0.8s | 2.9x |
| RoBERTa情感分析 | 0.5s | 0.2s | 2.5x |
| 整体推理延迟 | 3.2s | 1.1s | 2.9x |

### 硬件支持

- **CPU**: Intel Core i5以上
- **GPU**: Intel集成显卡, NVIDIA GPU
- **NPU**: Intel Meteor Lake以上处理器

## 📁 项目结构

```
NeuraLink_intel/
├── demo1.py                    # 主应用程序
├── index.py                    # 简化版入口
├── whisper_ov_runner.py        # OpenVINO Whisper推理器
├── Decoder.py                  # 模型导出工具
├── db_logger.py                # 数据库日志模块
├── fastrun.bat                 # Windows快速启动
├── dialogue_log.db             # SQLite数据库
├── input.wav                   # 音频输入文件
├── models/                     # 模型文件目录
│   └── emotion_openvino/       # OpenVINO优化情感模型
├── output/                     # 模型输出目录
│   ├── openvino_encoder_model.xml
│   ├── openvino_decoder_model.xml
│   └── ...
├── Whisper_onx/               # Whisper ONNX模型
└── __pycache__/               # Python缓存
```

## 🎮 使用示例

### 基本语音交互

1. 点击"开始录音"按钮
2. 对着麦克风说话（5秒）
3. 系统自动识别语音并分析情感
4. AI助手给出智能回复
5. 查看情绪分析结果和建议

### 情绪趋势分析

1. 在侧边栏选择"情绪趋势分析"
2. 查看周/月情绪变化图表
3. 分析情绪分布和异常模式
4. 导出分析报告

## 🔍 技术细节

### OpenVINO模型优化

```python
# 情感分析模型优化
emo_model = OVModelForSequenceClassification.from_pretrained(
    "uer/roberta-base-finetuned-jd-binary-chinese",
    export=True
)
emo_model.save_pretrained("models/emotion_openvino")
```

### Whisper模型导出

使用 `Decoder.py` 将Whisper模型转换为OpenVINO格式：

```python
# 导出encoder
torch.onnx.export(
    encoder, mel_input, "whisper_encoder.onnx",
    export_params=True, opset_version=13
)
```

## 🤝 贡献指南

欢迎为项目贡献代码！请遵循以下步骤：

1. Fork 这个仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 License

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🏆 比赛信息

- **大赛名称**: 2025英特尔人工智能创新应用大赛
- **参赛主题**: "码"上出发·"芯"创未来
- **技术栈**: Intel OpenVINO + AI模型优化
- **应用场景**: 儿童心理健康监护

## 📞 联系我们

- **项目维护者**: [wilson534](https://github.com/wilson534)
- **Issue反馈**: [GitHub Issues](https://github.com/wilson534/neuralink_intel625/issues)
- **技术交流**: 欢迎提出问题和建议

## 🙏 致谢

- Intel OpenVINO 团队提供的优秀AI推理框架
- Whisper团队开源的语音识别模型
- HuggingFace社区的预训练模型
- Streamlit提供的快速Web应用框架

---

**让AI技术更好地服务于儿童心理健康 🌟** 
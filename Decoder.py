import whisper
import torch
from whisper.model import Whisper

# 加载 Whisper 模型
model_name = "base"
print(f"Loading Whisper model: {model_name}")
whisper_model = whisper.load_model(model_name)
whisper_model.eval()

# 提取 encoder 部分
class WhisperEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, mel):
        return self.encoder(mel)

# 构造输入：[batch_size=1, channels=80, time_steps=3000]
mel_input = torch.randn(1, 80, 3000)

# 导出 encoder
encoder = WhisperEncoder(whisper_model)
torch.onnx.export(
    encoder,
    mel_input,
    "whisper_encoder.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["mel"],
    output_names=["encoder_out"],
    dynamic_axes={
        "mel": {0: "batch", 2: "time"},
        "encoder_out": {0: "batch", 1: "time"},
    },
)
# 提取 decoder 单步推理部分
class WhisperDecoderStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.token_embedding = model.decoder.token_embedding
        self.positional_embedding = model.decoder.positional_embedding
        self.blocks = model.decoder.blocks
        self.ln = model.decoder.ln
        self.proj = model.decoder.proj

    def forward(self, tokens, encoder_out):
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding[: x.shape[-2], :]
        x = x @ self.blocks[0].attn.query.weight  # 简化示意，可替换为实际 attn 实现
        for block in self.blocks:
            x = block(x, encoder_out)
        x = self.ln(x)
        x = self.proj(x)
        return x

# 构造输入
tokens_step = torch.zeros(1, 1).long()  # [B, T=1]
encoder_out = whisper_model.encoder(mel_input)

# 导出 decoder 单步
decoder_step = WhisperDecoderStep(whisper_model)
torch.onnx.export(
    decoder_step,
    (tokens_step, encoder_out),
    "whisper_decoder_step.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["tokens", "encoder_out"],
    output_names=["logits"],
    dynamic_axes={
        "tokens": {0: "batch", 1: "sequence"},
        "encoder_out": {0: "batch", 1: "time"},
        "logits": {0: "batch", 1: "sequence"},
    },
)

print("Whisper decoder step exported to whisper_decoder_step.onnx")
print("Whisper encoder exported to whisper_encoder.onnx")

import torch

# 获取 traced model
traced_model = torch.jit.trace(model, dummy_input)

# 打印图形结构
print(traced_model.graph)

# 或者保存为 dot 文件用工具可视化
torch.jit.save(traced_model, "model.pt")
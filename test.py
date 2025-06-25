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

print("Whisper encoder exported to whisper_encoder.onnx")
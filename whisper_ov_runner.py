import numpy as np
import self
from openvino import Core
from whisper.audio import load_audio, log_mel_spectrogram
from whisper.tokenizer import get_tokenizer

class WhisperOVRunner:
    def __init__(self,
                 encoder_path="output/openvino_encoder_model.xml",
                 decoder_path="output/openvino_decoder_model.xml",
                 device="GPU",
                 max_tokens=100,
                 language="zh",
                 target_length=3000):
        self.core = Core()
        self.encoder_model = self.core.compile_model(encoder_path, device)
        self.decoder_model = self.core.compile_model(decoder_path, device)
        self.tokenizer = get_tokenizer(multilingual=True, language=language, task="transcribe")
        self.SOT = self.tokenizer.sot
        self.EOT = self.tokenizer.eot
        self.MAX_TOKENS = max_tokens
        self.target_length = target_length  # 模型输入时间步长固定3000



    def pad_or_trim_mel(self, mel):
        # mel shape: [batch, n_mels=80, time_steps]
        _, n_mels, t = mel.shape
        if t == self.target_length:
            return mel
        elif t > self.target_length:
            return mel[:, :, :self.target_length]
        else:
            pad_width = self.target_length - t
            pad_shape = ((0, 0), (0, 0), (0, pad_width))
            return np.pad(mel, pad_shape, mode='constant', constant_values=0)

    def transcribe(self, audio_path):
        # Step 1: 加载音频，转 mel 频谱
        audio = load_audio(audio_path)
        mel = log_mel_spectrogram(audio).numpy()[np.newaxis, ...]
        mel = self.pad_or_trim_mel(mel)

        # Step 2: encoder 推理
        encoder_input = {self.encoder_model.input(0): mel}
        encoder_output = self.encoder_model(encoder_input)[self.encoder_model.output(0)]

        # Step 3: decoder 推理（逐 token 解码）
        tokens = [self.SOT]
        for _ in range(self.MAX_TOKENS):
            decoder_inputs = {
                self.decoder_model.input(0): np.array([tokens], dtype=np.int32),  # decoder_input_ids
                self.decoder_model.input(1): encoder_output  # encoder_hidden_states
            }
            out = self.decoder_model(decoder_inputs)
            logits = out[self.decoder_model.output(0)]
            next_token = np.argmax(logits[0, -1])  # 取最后一个 token 的预测
            if next_token == self.EOT:
                break
            tokens.append(next_token)

        # Step 4: token → 文本
        text = self.tokenizer.decode(tokens[1:])  # 去掉起始 token
        return text


whisper_runner = WhisperOVRunner()

def transcribe_with_openvino(audio_path):
    return whisper_runner.transcribe(audio_path)


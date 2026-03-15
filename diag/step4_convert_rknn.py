"""Step 4: 수정된 ONNX → RKNN 변환"""
import numpy as np
import sys

try:
    from rknnlite.api import RKNNLite as RKNN
    is_lite = True
except ImportError:
    from rknn.api import RKNN
    is_lite = False

print(f"Using {'RKNNLite' if is_lite else 'RKNN'}")

onnx_path = '../models/porting/BCResNet-t2-rknn-compatible.onnx'
rknn_path  = '../models/porting/BCResNet-t2-rknn-compatible.rknn'

rknn = RKNN(verbose=False)

# 설정
print("\n--> Config...")
ret = rknn.config(
    target_platform='rk3588',
    mean_values=[[0]],
    std_values=[[1]],
)
print(f"    config ret={ret}")

# ONNX 로드
print(f"\n--> Loading ONNX: {onnx_path}")
ret = rknn.load_onnx(model=onnx_path)
if ret != 0:
    print(f"ERROR: load_onnx failed (ret={ret})")
    sys.exit(1)
print(f"    load_onnx ret={ret} OK")

# Build
print("\n--> Building RKNN model...")
ret = rknn.build(do_quantization=False)
if ret != 0:
    print(f"ERROR: build failed (ret={ret})")
    sys.exit(1)
print(f"    build ret={ret} OK")

# Export
print(f"\n--> Exporting RKNN: {rknn_path}")
ret = rknn.export_rknn(rknn_path)
if ret != 0:
    print(f"ERROR: export_rknn failed (ret={ret})")
    sys.exit(1)
print(f"    export_rknn ret={ret} OK")

# 변환 후 시뮬레이터로 추론 테스트
print("\n--> Init runtime (simulator)...")
ret = rknn.init_runtime()
if ret != 0:
    print(f"ERROR: init_runtime failed (ret={ret})")
    rknn.release()
    sys.exit(1)

# 웨이크워드 샘플로 추론
import wave

class LogMel:
    def __init__(self, sample_rate=16000, hop_length=160, win_length=480, n_fft=512, n_mels=40):
        self.sr = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.mel_basis = self._create_mel_filterbank()
        self.window = np.hanning(win_length)
    def _create_mel_filterbank(self):
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.0)
        def mel_to_hz(mel): return 700 * (10**(mel / 2595.0) - 1)
        all_freqs = np.linspace(0, self.sr / 2, self.n_fft // 2 + 1)
        mel_points = np.linspace(hz_to_mel(0), hz_to_mel(self.sr / 2), self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            left, center, right = hz_points[i:i+3]
            for j, f in enumerate(all_freqs):
                if left < f < center:
                    filterbank[i, j] = (f - left) / (center - left)
                elif center <= f < right:
                    filterbank[i, j] = (right - f) / (right - center)
        return filterbank
    def __call__(self, waveform):
        target_length = 24000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        elif len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
        pad_size = self.n_fft // 2
        waveform = np.pad(waveform, (pad_size, pad_size), mode='reflect')
        frames = []
        for i in range(0, len(waveform) - self.n_fft + 1, self.hop_length):
            chunk = waveform[i:i+self.win_length]
            if len(chunk) < self.n_fft:
                chunk = np.pad(chunk, (0, self.n_fft - len(chunk)), mode='constant')
            chunk = chunk * np.pad(self.window, (0, self.n_fft - len(self.window)), mode='constant')
            spec = np.abs(np.fft.rfft(chunk, n=self.n_fft))**2
            frames.append(spec)
        spectrogram = np.array(frames).T
        mel_spec = np.dot(self.mel_basis, spectrogram)
        return np.log(mel_spec + 1e-6).astype(np.float32)

audio_path = '../wallpad_HiWonder_251113/lkk/lkk_1_2.wav'
with wave.open(audio_path, 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

logmel = LogMel()
feat = logmel(audio)[np.newaxis, np.newaxis, ...]  # (1,1,40,151)

print(f"\n--> Running inference on RKNN simulator...")
outputs = rknn.inference(inputs=[feat], data_format='nchw')
logits = outputs[0][0]
print(f"    logits={logits}")

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

probs = softmax(logits)
print(f"    probs={probs}")
print(f"\n기대값: probs ≈ [0.077, 0.923]")

if probs[1] > 0.8:
    print("✓ PASS: 웨이크워드 클래스 확률이 0.8 이상입니다")
else:
    print(f"✗ 웨이크워드 확률 낮음: {probs[1]:.4f}")

rknn.release()
print(f"\n완료! RKNN 파일: {rknn_path}")

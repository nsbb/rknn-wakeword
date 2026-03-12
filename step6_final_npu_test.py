"""Step 6: 수정된 RKNN 모델을 실제 NPU에서 추론하여 ONNX 결과와 비교"""
import numpy as np
import wave, sys

# 1. ONNX 기준 결과
import onnxruntime as ort

orig_onnx_path = '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-Focal-ep110.onnx'
mod_onnx_path  = '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-rknn-compatible.onnx'
rknn_path      = '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-rknn-compatible.rknn'
audio_path     = '/home/rk3588/travail/rk3588/Rockchip_VT/wallpad_HiWonder_251113/lkk/lkk_1_2.wav'

class LogMel:
    def __init__(self, sample_rate=16000, hop_length=160, win_length=480, n_fft=512, n_mels=40):
        self.sr = sample_rate; self.hop_length = hop_length
        self.win_length = win_length; self.n_fft = n_fft; self.n_mels = n_mels
        self.mel_basis = self._create_mel_filterbank()
        self.window = np.hanning(win_length)
    def _create_mel_filterbank(self):
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.0)
        def mel_to_hz(mel): return 700 * (10**(mel / 2595.0) - 1)
        all_freqs = np.linspace(0, self.sr / 2, self.n_fft // 2 + 1)
        mel_points = np.linspace(hz_to_mel(0), hz_to_mel(self.sr / 2), self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        fb = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(self.n_mels):
            l, c, r = hz_points[i:i+3]
            for j, f in enumerate(all_freqs):
                if l < f < c: fb[i, j] = (f - l) / (c - l)
                elif c <= f < r: fb[i, j] = (r - f) / (r - c)
        return fb
    def __call__(self, w):
        t = 24000
        if len(w) > t: w = w[:t]
        elif len(w) < t: w = np.pad(w, (0, t - len(w)))
        w = np.pad(w, (self.n_fft//2, self.n_fft//2), mode='reflect')
        frames = []
        for i in range(0, len(w) - self.n_fft + 1, self.hop_length):
            chunk = w[i:i+self.win_length]
            if len(chunk) < self.n_fft: chunk = np.pad(chunk, (0, self.n_fft - len(chunk)))
            chunk = chunk * np.pad(self.window, (0, self.n_fft - len(self.window)))
            frames.append(np.abs(np.fft.rfft(chunk, n=self.n_fft))**2)
        spec = np.array(frames).T
        return np.log(self.mel_basis @ spec + 1e-6).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open(audio_path, 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

print("=" * 60)
print("ONNX 기준 결과")
print("=" * 60)
sess_orig = ort.InferenceSession(orig_onnx_path, providers=['CPUExecutionProvider'])
out_orig = sess_orig.run(None, {'input': feat})[0]
probs_orig = softmax(out_orig[0])
print(f"원본 ONNX  logits={out_orig[0]}, probs={probs_orig}")

sess_mod = ort.InferenceSession(mod_onnx_path, providers=['CPUExecutionProvider'])
out_mod = sess_mod.run(None, {'input': feat})[0]
probs_mod = softmax(out_mod[0])
print(f"수정 ONNX  logits={out_mod[0]}, probs={probs_mod}")

# 2. RKNN NPU 추론
print("\n" + "=" * 60)
print("RKNN NPU 추론")
print("=" * 60)

try:
    from rknnlite.api import RKNNLite as RKNN
    is_lite = True
except ImportError:
    from rknn.api import RKNN
    is_lite = False

print(f"Using {'RKNNLite' if is_lite else 'RKNN'}")

rknn = RKNN(verbose=False)
ret = rknn.load_rknn(rknn_path)
print(f"load_rknn ret={ret}")
if ret != 0:
    print("ERROR: cannot load rknn model"); sys.exit(1)

if is_lite:
    ret = rknn.init_runtime()
else:
    try:
        ret = rknn.init_runtime(target='rk3588')
    except Exception:
        ret = rknn.init_runtime()
print(f"init_runtime ret={ret}")
if ret != 0:
    print("ERROR: cannot init runtime"); sys.exit(1)

outputs = rknn.inference(inputs=[feat], data_format='nchw')
logits_npu = outputs[0][0]
probs_npu = softmax(logits_npu)
print(f"RKNN NPU   logits={logits_npu}, probs={probs_npu}")

# 3. 비교
print("\n" + "=" * 60)
print("비교 결과")
print("=" * 60)
diff_logits = np.abs(out_orig[0] - logits_npu)
print(f"원본ONNX vs NPU  max_diff(logits): {diff_logits.max():.4f}")
print(f"원본ONNX probs: {probs_orig}")
print(f"NPU      probs: {probs_npu}")

if probs_npu[1] > 0.8:
    print("\n✓ PASS: NPU가 웨이크워드를 올바르게 인식합니다 (class1 > 0.8)")
else:
    print(f"\n✗ FAIL: NPU 웨이크워드 확률 낮음: {probs_npu[1]:.4f}")

rknn.release()

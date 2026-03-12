"""Step 3: 수정된 ONNX가 원본과 동일한 출력을 내는지 검증"""
import numpy as np
import onnxruntime as ort
import wave, sys

orig_path  = '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-Focal-ep110.onnx'
mod_path   = '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-rknn-compatible.onnx'
audio_path = '/home/rk3588/travail/rk3588/Rockchip_VT/wallpad_HiWonder_251113/lkk/lkk_1_2.wav'

# LogMel 클래스 (inference_rknn.py에서 복사)
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

# 오디오 로드
print(f"Loading audio: {audio_path}")
with wave.open(audio_path, 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

logmel = LogMel()
feat = logmel(audio)[np.newaxis, np.newaxis, ...]  # (1,1,40,151)
print(f"Feature shape: {feat.shape}")

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# 원본 추론
sess_orig = ort.InferenceSession(orig_path, providers=['CPUExecutionProvider'])
out_orig = sess_orig.run(None, {'input': feat})[0]
probs_orig = softmax(out_orig[0])
print(f"\n[원본 ONNX] logits={out_orig[0]}, probs={probs_orig}")

# 수정된 모델 추론
sess_mod = ort.InferenceSession(mod_path, providers=['CPUExecutionProvider'])
out_mod = sess_mod.run(None, {'input': feat})[0]
probs_mod = softmax(out_mod[0])
print(f"[수정 ONNX] logits={out_mod[0]}, probs={probs_mod}")

# 차이 계산
max_diff = np.max(np.abs(out_orig - out_mod))
print(f"\nmax_diff (logits): {max_diff:.2e}")

if max_diff < 1e-5:
    print("✓ PASS: 출력이 원본과 일치합니다 (max_diff < 1e-5)")
elif max_diff < 1e-3:
    print("△ 허용 범위: max_diff가 1e-5보다 크지만 1e-3 미만 (부동소수점 누적 오차 가능)")
else:
    print(f"✗ FAIL: max_diff={max_diff:.2e} (너무 큼!)")
    sys.exit(1)

# 랜덤 입력으로 추가 검증
print("\n--- 랜덤 입력 10개로 추가 검증 ---")
np.random.seed(42)
max_diffs = []
for i in range(10):
    rand_input = np.random.randn(1, 1, 40, 151).astype(np.float32)
    o1 = sess_orig.run(None, {'input': rand_input})[0]
    o2 = sess_mod.run(None, {'input': rand_input})[0]
    d = np.max(np.abs(o1 - o2))
    max_diffs.append(d)

print(f"랜덤 입력 max_diff: min={min(max_diffs):.2e}, max={max(max_diffs):.2e}, mean={np.mean(max_diffs):.2e}")
if max(max_diffs) < 1e-5:
    print("✓ PASS: 모든 랜덤 입력에서 출력이 일치합니다")
else:
    print(f"△ 최대 차이: {max(max_diffs):.2e}")

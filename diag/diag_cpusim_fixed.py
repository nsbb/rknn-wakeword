"""
BCResNet-t2-npu-fixed.onnx → CPU 시뮬레이션 (rknn.api만 사용, inference_rknn 미사용)
"""
from rknn.api import RKNN  # must be first, before rknnlite
import numpy as np, sys, wave

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def logmel_numpy(audio, sr=16000, n_mels=40, n_fft=400, hop=160, fmin=0, fmax=8000):
    """Numpy-only LogMel (periodic Hann window)"""
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)  # periodic
    # STFT
    frames = 1 + (len(audio) - n_fft) // hop
    feat = np.zeros((frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(frames):
        x = audio[i * hop: i * hop + n_fft] * win
        spec = np.fft.rfft(x)
        feat[i] = np.abs(spec) ** 2
    # Mel filterbank
    mel_low = 2595 * np.log10(1 + fmin / 700)
    mel_high = 2595 * np.log10(1 + fmax / 700)
    mel_pts = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f_m_minus = bins[m]; f_m = bins[m + 1]; f_m_plus = bins[m + 2]
        for k in range(f_m_minus, f_m):
            fbank[m, k] = (k - bins[m]) / (bins[m + 1] - bins[m])
        for k in range(f_m, f_m_plus):
            fbank[m, k] = (bins[m + 2] - k) / (bins[m + 2] - bins[m + 1])
    mel = np.dot(feat, fbank.T)
    mel = np.log(np.maximum(mel, 1e-9))
    return mel.T  # (n_mels, frames)

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat_2d = logmel_numpy(audio)
# crop/pad to 151 frames
if feat_2d.shape[1] > 151:
    feat_2d = feat_2d[:, :151]
elif feat_2d.shape[1] < 151:
    feat_2d = np.pad(feat_2d, ((0,0),(0,151-feat_2d.shape[1])))
feat = feat_2d[np.newaxis, np.newaxis, ...]  # (1,1,40,151)
print(f'Feature: {feat.shape}, range=[{feat.min():.2f},{feat.max():.2f}]')

# RKNN CPU 시뮬레이션 (load_onnx + build + init_runtime() without target)
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='BCResNet-t2-npu-fixed.onnx')
rknn.build(do_quantization=False)

ret = rknn.init_runtime()  # CPU sim
print(f'CPU sim init: {ret}')
raw_cpu = rknn.inference(inputs=[feat], data_format='nchw')[0]
print(f'CPU sim raw: {raw_cpu.squeeze()}')
print(f'CPU sim probs: {softmax(raw_cpu.squeeze())}')

raw_z = rknn.inference(inputs=[np.zeros_like(feat)], data_format='nchw')[0]
print(f'CPU sim zeros: {raw_z.squeeze()}')
print(f'Constant (bad)?: {np.allclose(raw_cpu, raw_z)}')
rknn.release()

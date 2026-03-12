"""Step 8: perf_debug로 NPU에서 어떤 op이 CPU fallback되는지 확인"""
import numpy as np, wave, sys
sys.path.insert(0, '/home/rk3588/travail/rk3588/Rockchip_VT')
from inference_rknn import LogMel
from rknn.api import RKNN

rknn_path = '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-rknn-compatible.rknn'
audio_path = '/home/rk3588/travail/rk3588/Rockchip_VT/wallpad_HiWonder_251113/lkk/lkk_1_2.wav'

with wave.open(audio_path, 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, :]

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

print("=== 수정 모델 (AveragePool) ===")
rknn = RKNN(verbose=False)
rknn.load_rknn(rknn_path)
ret = rknn.init_runtime(target='rk3588', perf_debug=True)
print(f"init_runtime ret={ret}")
raw = rknn.inference(inputs=[feat], data_format='nchw')[0]
print(f"probs: {softmax(raw.squeeze())}")
print(f"raw: {raw}")
print("\n--- eval_perf ---")
rknn.eval_perf()
rknn.release()

"""Step 7: build 직후 NPU에서 직접 추론 (load_onnx → build → init_runtime(target='rk3588') → inference)"""
import numpy as np, wave, sys
sys.path.insert(0, '/home/rk3588/travail/rk3588/Rockchip_VT')
from inference_rknn import LogMel

onnx_path = '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-rknn-compatible.onnx'
audio_path = '/home/rk3588/travail/rk3588/Rockchip_VT/wallpad_HiWonder_251113/lkk/lkk_1_2.wav'

try:
    from rknnlite.api import RKNNLite as RKNN
    is_lite = True
except ImportError:
    from rknn.api import RKNN
    is_lite = False

print(f"Using {'RKNNLite' if is_lite else 'RKNN'}")

with wave.open(audio_path, 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
lm = LogMel()
feat = lm(audio)[np.newaxis, np.newaxis, :]
print(f"Feature shape: {feat.shape}")

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

rknn = RKNN(verbose=False)

print("\n--> config")
rknn.config(target_platform='rk3588', mean_values=[[0]], std_values=[[1]])

print("--> load_onnx")
ret = rknn.load_onnx(model=onnx_path)
print(f"  ret={ret}")

print("--> build")
ret = rknn.build(do_quantization=False)
print(f"  ret={ret}")

print("--> init_runtime (NPU target='rk3588')")
try:
    ret = rknn.init_runtime(target='rk3588')
except Exception as e:
    print(f"  exception: {e}")
    ret = -1
print(f"  ret={ret}")

if ret != 0:
    print("  Failed to init NPU, trying simulator...")
    rknn2 = RKNN(verbose=False)
    rknn2.config(target_platform='rk3588', mean_values=[[0]], std_values=[[1]])
    rknn2.load_onnx(model=onnx_path)
    rknn2.build(do_quantization=False)
    ret2 = rknn2.init_runtime()
    print(f"  sim init ret={ret2}")
    if ret2 == 0:
        outputs = rknn2.inference(inputs=[feat], data_format='nchw')
        logits = outputs[0][0]
        probs = softmax(logits)
        print(f"\n[Simulator] logits={logits}")
        print(f"[Simulator] probs={probs}")
    rknn2.release()
else:
    outputs = rknn.inference(inputs=[feat], data_format='nchw')
    logits = outputs[0][0]
    probs = softmax(logits)
    print(f"\n[NPU] logits={logits}")
    print(f"[NPU] probs={probs}")
    if probs[1] > 0.8:
        print("✓ PASS")
    else:
        print(f"✗ FAIL (class1={probs[1]:.4f})")

rknn.release()

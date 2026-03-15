"""시뮬레이터 vs NPU 직접 비교 - init_runtime 방식 차이 확인"""
import numpy as np, wave, sys
sys.path.insert(0, '..')
from inference_rknn import LogMel

rknn_path = '../models/porting/BCResNet-t2-rknn-compatible.rknn'
audio_path = '../wallpad_HiWonder_251113/lkk/lkk_1_2.wav'

try:
    from rknnlite.api import RKNNLite as RKNN_LITE
    HAS_LITE = True
except ImportError:
    HAS_LITE = False

from rknn.api import RKNN

with wave.open(audio_path, 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
lm = LogMel()
feat = lm(audio)[np.newaxis, np.newaxis, :]
print(f"Feature shape: {feat.shape}")

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def run_rknn_sim(path):
    """RKNN Toolkit2 시뮬레이터 (target=None)"""
    print("\n--- RKNN Toolkit2 Simulator (target=None) ---")
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(path)
    print(f"  load_rknn ret={ret}")
    ret = rknn.init_runtime()  # target=None → simulator
    print(f"  init_runtime(sim) ret={ret}")
    if ret == 0:
        outputs = rknn.inference(inputs=[feat], data_format='nchw')
        logits = outputs[0][0]
        probs = softmax(logits)
        print(f"  logits={logits}")
        print(f"  probs={probs}")
        if probs[1] > 0.8: print("  → PASS")
        else: print(f"  → FAIL (class1={probs[1]:.4f})")
    rknn.release()

def run_rknn_npu(path):
    """RKNN Toolkit2 → 실제 NPU (target='rk3588')"""
    print("\n--- RKNN Toolkit2 NPU (target='rk3588') ---")
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(path)
    print(f"  load_rknn ret={ret}")
    try:
        ret = rknn.init_runtime(target='rk3588')
    except Exception as e:
        print(f"  init_runtime exception: {e}")
        ret = -1
    print(f"  init_runtime(NPU) ret={ret}")
    if ret == 0:
        outputs = rknn.inference(inputs=[feat], data_format='nchw')
        logits = outputs[0][0]
        probs = softmax(logits)
        print(f"  logits={logits}")
        print(f"  probs={probs}")
        if probs[1] > 0.8: print("  → PASS")
        else: print(f"  → FAIL (class1={probs[1]:.4f})")
    rknn.release()

def run_rknnlite_npu(path):
    """RKNNLite (실기기용 lite runtime)"""
    if not HAS_LITE:
        print("\n--- RKNNLite: Not available ---")
        return
    print("\n--- RKNNLite (실기기 NPU) ---")
    rknn = RKNN_LITE(verbose=False)
    ret = rknn.load_rknn(path)
    print(f"  load_rknn ret={ret}")
    ret = rknn.init_runtime()
    print(f"  init_runtime ret={ret}")
    if ret == 0:
        outputs = rknn.inference(inputs=[feat], data_format='nchw')
        logits = outputs[0][0]
        probs = softmax(logits)
        print(f"  logits={logits}")
        print(f"  probs={probs}")
        if probs[1] > 0.8: print("  → PASS")
        else: print(f"  → FAIL (class1={probs[1]:.4f})")
    rknn.release()

run_rknn_sim(rknn_path)
run_rknnlite_npu(rknn_path)
run_rknn_npu(rknn_path)

print("\nHAS_LITE:", HAS_LITE)

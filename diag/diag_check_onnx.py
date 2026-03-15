"""
check*.onnx 파일들로 RKNN 변환 테스트
"""
import numpy as np, sys, wave, os
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknn.api import RKNN
import onnxruntime as ort

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

candidates = [
    '../models/BCResNet-t2-Focal-ep110.onnx',
    '../models/porting/check0_base_optimize.onnx',
    '../models/porting/check1_fold_constant.onnx',
    '../models/porting/check2_correct_ops.onnx',
    '../models/porting/check3_fuse_ops.onnx',
]

for onnx_path in candidates:
    if not os.path.exists(onnx_path):
        continue
    size_kb = os.path.getsize(onnx_path) / 1024

    # ONNX output
    try:
        sess = ort.InferenceSession(onnx_path)
        onnx_out = sess.run(None, {sess.get_inputs()[0].name: feat})[0]
        onnx_probs = softmax(onnx_out.squeeze())
    except Exception as e:
        print(f'[{onnx_path}] ONNX error: {e}')
        continue

    # RKNN 변환 + NPU 추론
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f'[{onnx_path}] load_onnx failed'); rknn.release(); continue
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f'[{onnx_path}] build failed'); rknn.release(); continue
    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print(f'[{onnx_path}] init_runtime failed'); rknn.release(); continue

    raw = rknn.inference(inputs=[feat], data_format='nchw')[0]
    rknn_probs = softmax(raw.squeeze())
    rknn.release()

    match = np.allclose(onnx_probs, rknn_probs, atol=0.05)
    print(f'[{onnx_path}] ({size_kb:.0f}KB)')
    print(f'  ONNX : {onnx_probs}')
    print(f'  RKNN : {rknn_probs}  {"✓ OK" if match else "✗ WRONG"}')

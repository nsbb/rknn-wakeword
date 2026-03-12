"""
1) 변환 당시 validation 조건(uniform 0~1)으로 재현
2) native.rknn 비교
3) 실제 logmel 범위로 ONNX vs RKNN 비교
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from inference_rknn import RKNNInferenceEngine
import onnxruntime as ort

session = ort.InferenceSession('BCResNet-t2-Focal-ep110.onnx')
input_name = session.get_inputs()[0].name

def test(feat, label):
    onnx_out = session.run(None, {input_name: feat})[0]
    print(f'[{label}] ONNX : {onnx_out}')

np.random.seed(0)

# 1) uniform(0,1) - 변환 validation 조건
feat_uniform = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)
test(feat_uniform, 'uniform(0,1)')

# 2) 실제 logmel 범위 (-12 ~ 8.5)
feat_logmel = np.random.uniform(-12, 8.5, (1, 1, 40, 151)).astype(np.float32)
test(feat_logmel, 'uniform(-12,8.5)')

engine = RKNNInferenceEngine('BCResNet-t2-Focal-ep110.rknn', target='rk3588')
if engine.load_model():
    print('\n--- BCResNet-t2-Focal-ep110.rknn ---')
    for feat, label in [(feat_uniform, 'uniform(0,1)'), (feat_logmel, 'uniform(-12,8.5)')]:
        onnx_out = session.run(None, {input_name: feat})[0]
        rknn_raw = engine.rknn.inference(inputs=[feat], data_format='nchw')[0]
        diff = np.abs(onnx_out - rknn_raw)
        print(f'[{label}]')
        print(f'  ONNX : {onnx_out}')
        print(f'  RKNN : {rknn_raw}')
        print(f'  diff  max={diff.max():.6f} mean={diff.mean():.6f}')
    engine.release()

engine2 = RKNNInferenceEngine('BCResNet-t2-Focal-ep110_native.rknn', target='rk3588')
if engine2.load_model():
    print('\n--- BCResNet-t2-Focal-ep110_native.rknn ---')
    import wave
    from inference_rknn import LogMel
    with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
        data = wf.readframes(wf.getnframes())
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    feat_real = LogMel()(audio)[np.newaxis, np.newaxis, ...]
    rknn_raw = engine2.rknn.inference(inputs=[feat_real], data_format='nchw')[0]
    print(f'  RKNN_native raw: {rknn_raw}')
    engine2.release()

"""
BCResNet-t2-npu-fixed.onnx → RKNN 변환 + rknnlite NPU 테스트
"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknn.api import RKNN
from rknnlite.api import RKNNLite
import onnxruntime as ort

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

# ONNX reference
sess = ort.InferenceSession('BCResNet-t2-npu-fixed.onnx')
onnx_out = sess.run(None, {sess.get_inputs()[0].name: feat})[0]
onnx_probs = softmax(onnx_out.squeeze())
print(f'ONNX probs: {onnx_probs}  pred={np.argmax(onnx_probs)}')

# RKNN 변환
print('\nConverting to RKNN...')
rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588')
ret = rknn.load_onnx(model='BCResNet-t2-npu-fixed.onnx')
print(f'load_onnx: {ret}')
ret = rknn.build(do_quantization=False)
print(f'build: {ret}')
ret = rknn.export_rknn('BCResNet-t2-npu-fixed.rknn')
print(f'export: {ret}')
rknn.release()
print('Saved: BCResNet-t2-npu-fixed.rknn')

# rknnlite로 NPU 추론
print('\nTesting with rknnlite...')
rknn_lite = RKNNLite(verbose=False)
ret = rknn_lite.load_rknn('BCResNet-t2-npu-fixed.rknn')
print(f'load_rknn: {ret}')
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
print(f'init_runtime: {ret}')

raw = rknn_lite.inference(inputs=[feat], data_format='nchw')[0]
npu_probs = softmax(raw.squeeze())
print(f'NPU  probs: {npu_probs}  pred={np.argmax(npu_probs)}')
print(f'raw: {raw.squeeze()}')

# zeros test
raw_z = rknn_lite.inference(inputs=[np.zeros_like(feat)], data_format='nchw')[0]
print(f'zeros raw: {raw_z.squeeze()}')
print(f'Constant output?: {np.allclose(raw, raw_z)}')

print(f'\nMatch ONNX vs NPU (atol=0.05): {np.allclose(onnx_probs, npu_probs, atol=0.05)}')
rknn_lite.release()

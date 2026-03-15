"""
optimization_level=0으로 변환 - op fusion 비활성화
"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknn.api import RKNN
import onnxruntime as ort

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

sess = ort.InferenceSession('../models/BCResNet-t2-Focal-ep110.onnx')
onnx_out = sess.run(None, {sess.get_inputs()[0].name: feat})[0]
print('ONNX probs:', softmax(onnx_out.squeeze()))

# optimization_level=0으로 재변환
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588', optimization_level=0)
rknn.load_onnx(model='../models/BCResNet-t2-Focal-ep110.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn('../models/porting/BCResNet-t2-opt0.rknn')

rknn.init_runtime(target='rk3588')
raw = rknn.inference(inputs=[feat], data_format='nchw')[0]
print('RKNN opt0 probs:', softmax(raw.squeeze()), '  raw:', raw)

# zeros input도 확인
raw_z = rknn.inference(inputs=[np.zeros_like(feat)], data_format='nchw')[0]
print('RKNN opt0 zeros:', raw_z)

rknn.release()

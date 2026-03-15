"""
convert.py로 새로 만든 rknn 테스트 + op 분석
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

# convert.py로 만든 모델
rknn = RKNN(verbose=False)
rknn.load_rknn('../models/porting/BCResNet-t2-Focal-ep110_conv.rknn')
ret = rknn.init_runtime(target='rk3588', perf_debug=True)
print('init_runtime ret:', ret)
raw = rknn.inference(inputs=[feat], data_format='nchw')[0]
print('RKNN probs:', softmax(raw.squeeze()))
print('RKNN raw:', raw)

# 어떤 op이 CPU fallback 됐는지 확인
print('\n--- perf ---')
rknn.eval_perf()
rknn.release()

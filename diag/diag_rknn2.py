"""
RKNN vs ONNX raw output 비교 - 원인 추적
"""
import numpy as np
import sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel, RKNNInferenceEngine
import onnxruntime as ort

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

wav_path = 'wallpad_HiWonder_251113/lkk/lkk_1_2.wav'
with wave.open(wav_path, 'rb') as wf:
    data = wf.readframes(wf.getnframes())
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

# ONNX raw logits
session = ort.InferenceSession('../models/BCResNet-t2-Focal-ep110.onnx')
onnx_out = session.run(None, {session.get_inputs()[0].name: feat})[0]
print('=== ONNX ===')
print('  raw output:', onnx_out)
print('  softmax   :', softmax(onnx_out.squeeze()))

# RKNN raw output (softmax 적용 전)
engine = RKNNInferenceEngine('../models/porting/BCResNet-t2-Focal-ep110.rknn', target='rk3588')
if engine.load_model():
    raw = engine.rknn.inference(inputs=[feat], data_format='nchw')
    print('\n=== RKNN ===')
    print('  raw output:', raw[0])
    print('  softmax   :', softmax(raw[0].squeeze()))
    print('  num outputs:', len(raw))
    print('  output shapes:', [r.shape for r in raw])

    # 혹시 NHWC로 넣으면 어떻게 되는지 테스트
    feat_nhwc = feat.transpose(0, 2, 3, 1)  # (1,40,151,1)
    raw_nhwc = engine.rknn.inference(inputs=[feat_nhwc], data_format='nhwc')
    print('\n=== RKNN (nhwc input) ===')
    print('  raw output:', raw_nhwc[0])
    print('  softmax   :', softmax(raw_nhwc[0].squeeze()))

    engine.release()

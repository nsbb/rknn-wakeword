import numpy as np
import sys
import wave
sys.path.insert(0, '.')
from inference_rknn import LogMel, RKNNInferenceEngine
import onnxruntime as ort

wav_path = 'wallpad_HiWonder_251113/lkk/lkk_1_2.wav'
with wave.open(wav_path, 'rb') as wf:
    data = wf.readframes(wf.getnframes())
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

logmel = LogMel()
feat = logmel(audio)[np.newaxis, np.newaxis, ...]
print('feat shape:', feat.shape, ' min=%.3f max=%.3f mean=%.3f' % (feat.min(), feat.max(), feat.mean()))

# ONNX
session = ort.InferenceSession('../models/BCResNet-t2-Focal-ep110.onnx')
logits_onnx = session.run(None, {session.get_inputs()[0].name: feat})[0]
exp = np.exp(logits_onnx - logits_onnx.max())
probs_onnx = exp / exp.sum()
print('ONNX  logits:', logits_onnx)
print('ONNX  probs  (cls0, cls1):', probs_onnx)

# RKNN
engine = RKNNInferenceEngine('../models/porting/BCResNet-t2-Focal-ep110.rknn', target='rk3588')
if engine.load_model():
    probs_rknn = engine.infer(feat)
    print('RKNN  probs  (cls0, cls1):', probs_rknn)
    engine.release()
else:
    print('RKNN load failed')

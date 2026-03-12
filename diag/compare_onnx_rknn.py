import numpy as np
import onnxruntime as ort
import os

from inference_rknn import RKNNInferenceEngine, LogMel, AudioPreprocessor

# Dummy data
audio = np.random.randn(16000).astype(np.float32)

logmel = LogMel(apply_preemph=False)
feat = logmel(audio)[np.newaxis, np.newaxis, ...]

print("LogMel Shape:", feat.shape)
print("LogMel Min/Max/Mean:", feat.min(), feat.max(), feat.mean())

# ONNX Runtime
onnx_path = 'BCResNet-t2-Focal-ep110.onnx'
ort_session = ort.InferenceSession(onnx_path)
onnx_input = {ort_session.get_inputs()[0].name: feat}
onnx_out = ort_session.run(None, onnx_input)[0]

print("ONNX Logits:", onnx_out)

exp_logits = np.exp(onnx_out - np.max(onnx_out, axis=-1, keepdims=True))
onnx_probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
print("ONNX Probs: ", onnx_probs)

# RKNN Runtime
rknn_path = 'BCResNet-t2-Focal-ep110.rknn'
engine = RKNNInferenceEngine(rknn_path, target='rk3588')
engine.load_model()
rknn_probs = engine.infer(feat)

print("RKNN Probs:", rknn_probs)


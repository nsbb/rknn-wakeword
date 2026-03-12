"""
onnxsim으로 모델 단순화 후 RKNN 변환 테스트
"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknn.api import RKNN
import onnxruntime as ort
import onnx
from onnxsim import simplify

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

# ONNX simplify
model = onnx.load('BCResNet-t2-Focal-ep110.onnx')
simplified, check = simplify(model, input_shapes={'input': [1, 1, 40, 151]})
print('Simplified:', check)
onnx.save(simplified, 'BCResNet-t2-simplified.onnx')
print(f'Original size: {len(model.SerializeToString())/1024:.1f}KB')
print(f'Simplified size: {len(simplified.SerializeToString())/1024:.1f}KB')

# ONNX check
sess = ort.InferenceSession('BCResNet-t2-simplified.onnx')
onnx_out = sess.run(None, {sess.get_inputs()[0].name: feat})[0]
print('Simplified ONNX probs:', softmax(onnx_out.squeeze()))

# RKNN 변환
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='BCResNet-t2-simplified.onnx')
rknn.build(do_quantization=False)
rknn.export_rknn('BCResNet-t2-simplified.rknn')
rknn.init_runtime(target='rk3588')

raw = rknn.inference(inputs=[feat], data_format='nchw')[0]
rknn_probs = softmax(raw.squeeze())
print('Simplified RKNN probs:', rknn_probs)
print('raw:', raw)
print('Match:', np.allclose(softmax(onnx_out.squeeze()), rknn_probs, atol=0.05))
rknn.release()

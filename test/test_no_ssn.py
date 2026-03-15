"""BCResNet-t2-no-ssn.rknn NPU 테스트"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknnlite.api import RKNNLite
import onnxruntime as ort

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

sess = ort.InferenceSession('../models/porting/BCResNet-t2-no-ssn.onnx')
onnx_out = sess.run(None, {sess.get_inputs()[0].name: feat})[0]
print(f'No-SSN ONNX probs: {softmax(onnx_out.squeeze())}')

rknn_lite = RKNNLite(verbose=False)
rknn_lite.load_rknn('../models/porting/BCResNet-t2-no-ssn.rknn')
rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

raw = rknn_lite.inference(inputs=[feat], data_format='nchw')[0]
print(f'No-SSN NPU  probs: {softmax(raw.squeeze())}')
print(f'raw: {raw.squeeze()}')

raw_z = rknn_lite.inference(inputs=[np.zeros_like(feat)], data_format='nchw')[0]
print(f'Constant output?: {np.allclose(raw, raw_z)} (zeros: {raw_z.squeeze()})')
rknn_lite.release()

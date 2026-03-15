"""
RKNN 재변환 - mean/std 없이, fp16
변환 후 즉시 NPU(target='rk3588')로 검증
"""
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN

ONNX_PATH = '../models/BCResNet-t2-Focal-ep110.onnx'
RKNN_PATH = '../models/porting/BCResNet-t2-Focal-ep110_v2.rknn'

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

# 변환
rknn = RKNN(verbose=True)

print('--> Config')
rknn.config(target_platform='rk3588')  # mean/std 제거

print('--> Load ONNX')
ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, 'load_onnx failed'

print('--> Build (fp, no quantization)')
ret = rknn.build(do_quantization=False)
assert ret == 0, 'build failed'

print('--> Export')
ret = rknn.export_rknn(RKNN_PATH)
assert ret == 0, 'export failed'

# NPU에서 즉시 검증
print('\n--> init_runtime on NPU (target=rk3588)')
ret = rknn.init_runtime(target='rk3588')
print('ret:', ret)

# 테스트 입력 3종
import wave, sys
sys.path.insert(0, '.')
from inference_rknn import LogMel

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat_real = LogMel()(audio)[np.newaxis, np.newaxis, ...]

feat_zero = np.zeros((1, 1, 40, 151), dtype=np.float32)
feat_uniform = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)

sess = ort.InferenceSession(ONNX_PATH)
iname = sess.get_inputs()[0].name

for feat, label in [(feat_zero, 'zeros'), (feat_uniform, 'uniform(0,1)'), (feat_real, 'real_audio')]:
    onnx_out = sess.run(None, {iname: feat})[0]
    rknn_out = rknn.inference(inputs=[feat], data_format='nchw')[0]
    diff = np.abs(onnx_out - rknn_out).max()
    print(f'\n[{label}]')
    print(f'  ONNX  logits={onnx_out}  probs={softmax(onnx_out.squeeze())}')
    print(f'  RKNN  logits={rknn_out}  probs={softmax(rknn_out.squeeze())}')
    print(f'  max_diff={diff:.6f}')

rknn.release()
print('\nDone.')

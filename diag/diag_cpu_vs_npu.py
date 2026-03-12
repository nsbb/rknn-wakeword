"""
CPU 시뮬레이션 vs NPU 비교 - 어디서 틀리는지 확인
"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknn.api import RKNN

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

# CPU 시뮬레이션 (target=None)
rknn_cpu = RKNN(verbose=False)
rknn_cpu.load_rknn('BCResNet-t2-Focal-ep110_conv.rknn')
rknn_cpu.init_runtime()  # target 없이 = CPU simulation
raw_cpu = rknn_cpu.inference(inputs=[feat], data_format='nchw')[0]
print('CPU sim probs:', softmax(raw_cpu.squeeze()), '  raw:', raw_cpu)
rknn_cpu.release()

# NPU
rknn_npu = RKNN(verbose=False)
rknn_npu.load_rknn('BCResNet-t2-Focal-ep110_conv.rknn')
rknn_npu.init_runtime(target='rk3588')
raw_npu = rknn_npu.inference(inputs=[feat], data_format='nchw')[0]
print('NPU       probs:', softmax(raw_npu.squeeze()), '  raw:', raw_npu)
rknn_npu.release()

print('\nDiff CPU vs NPU:', np.abs(raw_cpu - raw_npu).max())

"""
NPU 코어 조합 테스트
"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknnlite.api import RKNNLite

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

print(f'NPU_CORE_AUTO={RKNNLite.NPU_CORE_AUTO}')
print(f'NPU_CORE_0={RKNNLite.NPU_CORE_0}')
print(f'NPU_CORE_0_1={RKNNLite.NPU_CORE_0_1}')

for core_name, core_mask in [
    ('NPU_CORE_AUTO', RKNNLite.NPU_CORE_AUTO),
    ('NPU_CORE_0', RKNNLite.NPU_CORE_0),
    ('NPU_CORE_0_1', RKNNLite.NPU_CORE_0_1),
]:
    rknn_lite = RKNNLite(verbose=False)
    rknn_lite.load_rknn('../models/BCResNet-t2-npu-fixed.rknn')
    ret = rknn_lite.init_runtime(core_mask=core_mask)
    if ret != 0:
        print(f'  {core_name}: init failed ({ret})'); continue

    raw = rknn_lite.inference(inputs=[feat], data_format='nchw')[0]
    probs = softmax(raw.squeeze())
    print(f'  {core_name}: raw={raw.squeeze()}, probs={probs}, pred={np.argmax(probs)}')
    rknn_lite.release()

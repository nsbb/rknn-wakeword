"""
data_format 옵션 테스트 - NCHW vs NHWC vs None
"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknnlite.api import RKNNLite

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat_nchw = LogMel()(audio)[np.newaxis, np.newaxis, ...]  # (1,1,40,151)
feat_nhwc = feat_nchw.transpose(0, 2, 3, 1)               # (1,40,151,1)
feat_flat  = feat_nchw.reshape(1, 40, 151)                  # (1,40,151) - 3D

print(f'NCHW: {feat_nchw.shape}, NHWC: {feat_nhwc.shape}')

rknn_lite = RKNNLite(verbose=False)
rknn_lite.load_rknn('../models/BCResNet-t2-npu-fixed.rknn')
rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

for fmt, data in [
    ('nchw', feat_nchw),
    ('nhwc', feat_nchw),
    ('nhwc+transpose', feat_nhwc),
    (None, feat_nchw),
]:
    try:
        if fmt is None:
            raw = rknn_lite.inference(inputs=[data])[0]
        elif fmt == 'nhwc+transpose':
            raw = rknn_lite.inference(inputs=[data], data_format='nhwc')[0]
        else:
            raw = rknn_lite.inference(inputs=[data], data_format=fmt)[0]
        probs = softmax(raw.squeeze())
        print(f'  data_format={fmt!r}: raw={raw.squeeze()}, probs={probs}, pred={np.argmax(probs)}')
    except Exception as e:
        print(f'  data_format={fmt!r}: ERROR {e}')

rknn_lite.release()

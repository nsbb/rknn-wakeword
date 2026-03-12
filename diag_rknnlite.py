"""
rknnlite로 NPU 추론 테스트 - 올바른 온보드 런타임 사용
"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknnlite.api import RKNNLite

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]
print(f'Feature shape: {feat.shape}, range: [{feat.min():.2f}, {feat.max():.2f}]')

# BCResNet-t2-Focal-ep110_conv.rknn 시도 (convert.py로 만든 버전)
for rknn_path in [
    'BCResNet-t2-Focal-ep110_conv.rknn',
    'BCResNet-t2-Focal-ep110.rknn',
    'BCResNet-t2-rknn-compatible.rknn',
]:
    print(f'\n--- {rknn_path} ---')
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(rknn_path)
    if ret != 0:
        print(f'  load_rknn failed: {ret}'); continue

    # NPU core: RKNN_NPU_CORE_AUTO=0, CORE_0=1, CORE_0_1=3
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print(f'  init_runtime failed: {ret}'); rknn_lite.release(); continue

    # NCHW input
    raw = rknn_lite.inference(inputs=[feat], data_format='nchw')[0]
    probs = softmax(raw.squeeze())
    print(f'  raw: {raw.squeeze()}')
    print(f'  probs: {probs}')
    print(f'  pred: {np.argmax(probs)} (expected 1 for wake word)')

    # zeros input
    raw_z = rknn_lite.inference(inputs=[np.zeros_like(feat)], data_format='nchw')[0]
    print(f'  zeros raw: {raw_z.squeeze()}')
    print(f'  same as audio?: {np.allclose(raw, raw_z)}')

    rknn_lite.release()

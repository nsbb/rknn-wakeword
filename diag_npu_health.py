"""
NPU 하드웨어 동작 확인 - depth_anything_v2_small_336.rknn 테스트
"""
import numpy as np
from rknnlite.api import RKNNLite

# 임의 입력으로 NPU 기본 동작 확인
rknn_lite = RKNNLite(verbose=True)
ret = rknn_lite.load_rknn('/home/rk3588/Downloads/depth_anything_v2_small_336.rknn')
print(f'load_rknn: {ret}')
if ret != 0:
    print('Failed to load model'); exit(1)

ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
print(f'init_runtime: {ret}')
if ret != 0:
    print('Failed to init runtime'); exit(1)

# 임의 입력 (336x336x3 BGR)
inputs = [np.random.randint(0, 255, (1, 336, 336, 3), dtype=np.uint8)]
out1 = rknn_lite.inference(inputs=inputs)[0]
out2 = rknn_lite.inference(inputs=inputs)[0]
print(f'output shape: {out1.shape}')
print(f'output range: [{out1.min():.3f}, {out1.max():.3f}]')
print(f'same for same input?: {np.allclose(out1, out2)}')  # should be True

inputs2 = [np.zeros((1, 336, 336, 3), dtype=np.uint8)]
out_z = rknn_lite.inference(inputs=inputs2)[0]
print(f'zero input range: [{out_z.min():.3f}, {out_z.max():.3f}]')
print(f'constant output?: {np.allclose(out1, out_z)}')  # should be False if NPU works

rknn_lite.release()

"""
RKNN 모델 op 분석 - 어떤 op이 NPU에서 문제인지 확인
"""
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.load_rknn('BCResNet-t2-Focal-ep110.rknn')

# NPU에서 실행
ret = rknn.init_runtime(target='rk3588', perf_debug=True)
print('init_runtime ret:', ret)

import numpy as np
dummy = np.zeros((1, 1, 40, 151), dtype=np.float32)
outputs = rknn.inference(inputs=[dummy], data_format='nchw')
print('zero input output:', outputs[0])

dummy2 = np.ones((1, 1, 40, 151), dtype=np.float32)
outputs2 = rknn.inference(inputs=[dummy2], data_format='nchw')
print('ones input output:', outputs2[0])

rknn.release()

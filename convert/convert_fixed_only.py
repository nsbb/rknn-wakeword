"""
BCResNet-t2-npu-fixed.onnx → BCResNet-t2-npu-fixed.rknn 변환만
"""
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588')
ret = rknn.load_onnx(model='../models/BCResNet-t2-npu-fixed.onnx')
print(f'load_onnx: {ret}')
ret = rknn.build(do_quantization=False)
print(f'build: {ret}')
ret = rknn.export_rknn('../models/BCResNet-t2-npu-fixed.rknn')
print(f'export: {ret}')
rknn.release()
print('Done: ../models/BCResNet-t2-npu-fixed.rknn')

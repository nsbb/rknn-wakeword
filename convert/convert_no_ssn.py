"""BCResNet-t2-no-ssn.onnx → RKNN"""
from rknn.api import RKNN
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='../models/porting/BCResNet-t2-no-ssn.onnx')
ret = rknn.build(do_quantization=False)
print(f'build: {ret}')
rknn.export_rknn('../models/porting/BCResNet-t2-no-ssn.rknn')
rknn.release()
print('Done: ../models/porting/BCResNet-t2-no-ssn.rknn')

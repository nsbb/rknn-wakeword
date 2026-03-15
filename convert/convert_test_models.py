"""테스트 모델들 RKNN 변환"""
from rknn.api import RKNN
import os

for mname in ['test_conv_c1', 'test_conv_bn_c1', 'test_conv_c16']:
    if not os.path.exists(f'../models/porting/{mname}.onnx'):
        print(f'{mname}.onnx not found'); continue
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588')
    rknn.load_onnx(model=f'../models/porting/{mname}.onnx')
    ret = rknn.build(do_quantization=False)
    if ret == 0:
        rknn.export_rknn(f'../models/porting/{mname}.rknn')
        print(f'{mname}: build OK')
    else:
        print(f'{mname}: build FAILED {ret}')
    rknn.release()

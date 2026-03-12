"""Stack 모델들 RKNN 변환"""
from rknn.api import RKNN
import os

for n in [1, 2, 4, 8]:
    mname = f'test_stack_{n}'
    if not os.path.exists(f'{mname}.onnx'):
        continue
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588')
    rknn.load_onnx(model=f'{mname}.onnx')
    ret = rknn.build(do_quantization=False)
    if ret == 0:
        rknn.export_rknn(f'{mname}.rknn')
        print(f'{mname}: OK')
    else:
        print(f'{mname}: FAIL {ret}')
    rknn.release()

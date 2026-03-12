"""서브 모델 RKNN 변환"""
from rknn.api import RKNN
import os

for desc in ['after_first_conv', 'after_first_relu', 'after_f2_conv', 'after_block0', 'after_block1']:
    path = f'sub_{desc}.onnx'
    if not os.path.exists(path):
        print(f'{path}: not found'); continue
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588')
    rknn.load_onnx(model=path)
    ret = rknn.build(do_quantization=False)
    if ret == 0:
        rknn.export_rknn(f'sub_{desc}.rknn')
        print(f'{desc}: OK')
    else:
        print(f'{desc}: FAIL {ret}')
    rknn.release()

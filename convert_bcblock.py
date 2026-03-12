"""BCBlock 테스트 모델 RKNN 변환"""
from rknn.api import RKNN
rknn = RKNN(verbose=False)
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='test_bcblock.onnx')
ret = rknn.build(do_quantization=False)
print(f'build: {ret}')
rknn.export_rknn('test_bcblock.rknn')
rknn.release()
print('Done: test_bcblock.rknn')

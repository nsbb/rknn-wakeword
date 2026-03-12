"""원본 ONNX와 수정 ONNX 모두를 RKNN 변환하여 에러 비교"""
import numpy as np, sys

try:
    from rknnlite.api import RKNNLite as RKNN
    is_lite = True
except ImportError:
    from rknn.api import RKNN
    is_lite = False

def convert_and_test(onnx_path, label):
    print(f"\n{'='*60}")
    print(f"[{label}] {onnx_path}")
    print(f"{'='*60}")

    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3588', mean_values=[[0]], std_values=[[1]])
    ret = rknn.load_onnx(model=onnx_path)
    print(f"load_onnx ret={ret}")
    ret = rknn.build(do_quantization=False)
    print(f"build ret={ret}")
    rknn.release()

convert_and_test(
    '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-Focal-ep110.onnx',
    '원본 ONNX (ReduceMean 포함)'
)
convert_and_test(
    '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-rknn-compatible.onnx',
    '수정 ONNX (AveragePool으로 교체)'
)

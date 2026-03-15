"""Step 5: verbose=True로 어떤 op가 target=0인지 진단"""
import numpy as np, sys, io, contextlib

try:
    from rknnlite.api import RKNNLite as RKNN
    is_lite = True
except ImportError:
    from rknn.api import RKNN
    is_lite = False

onnx_path = '../models/porting/BCResNet-t2-rknn-compatible.onnx'

rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588', mean_values=[[0]], std_values=[[1]])
rknn.load_onnx(model=onnx_path)
ret = rknn.build(do_quantization=False)
print(f"\nbuild ret={ret}")
rknn.release()

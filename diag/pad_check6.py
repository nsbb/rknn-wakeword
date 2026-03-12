from rknn.api import RKNN
import numpy as np

rknn = RKNN(verbose=False)
rknn.config(
    mean_values=[[0]], std_values=[[1]], target_platform='rk3588'
)
rknn.load_onnx('BCResNet-t2-Focal-ep110.onnx')
rknn.build(do_quantization=False)

dummy_input = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)

rknn.init_runtime(target='rk3588')
outputs_rknn_target = rknn.inference(inputs=[dummy_input], data_format='nchw')[0]
print("Direct build and Target run Output: ", outputs_rknn_target)
rknn.release()


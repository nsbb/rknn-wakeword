from rknn.api import RKNN
import numpy as np

rknn = RKNN(verbose=False)
rknn.load_rknn('BCResNet-t2-Focal-ep110_native.rknn')
rknn.init_runtime(target='rk3588')

zero_input = np.zeros((1, 1, 40, 151)).astype(np.float32)
one_input = np.ones((1, 1, 40, 151)).astype(np.float32)

print("Zero output:", rknn.inference(inputs=[zero_input], data_format='nchw')[0])
print("One output:", rknn.inference(inputs=[one_input], data_format='nchw')[0])

rknn.release()

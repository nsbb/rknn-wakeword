import numpy as np
dummy_input = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)

from rknn.api import RKNN
import onnxruntime as ort

ort_session = ort.InferenceSession('../models/BCResNet-t2-Focal-ep110.onnx')
onnx_out = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input})[0]

rknn = RKNN(verbose=False)
rknn.load_rknn('../models/porting/BCResNet-t2-Focal-ep110.rknn')

# 1. Target=rk3588
rknn.init_runtime(target='rk3588')
outputs_rknn_target = rknn.inference(inputs=[dummy_input], data_format='nchw')[0]
rknn.release()

# 2. Target=Simulator (default)
rknn = RKNN(verbose=False)
rknn.load_rknn('../models/porting/BCResNet-t2-Focal-ep110.rknn')
rknn.init_runtime()
outputs_rknn_sim = rknn.inference(inputs=[dummy_input], data_format='nchw')[0]
rknn.release()

print("ONNX: ", onnx_out)
print("RKNN(rk3588): ", outputs_rknn_target)
print("RKNN(simulator): ", outputs_rknn_sim)


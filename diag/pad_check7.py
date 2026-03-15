from rknn.api import RKNN
import numpy as np
import onnxruntime as ort

dummy_input_nchw = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)

# ONNX Runtime (NCHW)
ort_session = ort.InferenceSession('../models/BCResNet-t2-Focal-ep110.onnx')
onnx_out = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_input_nchw})[0]

# RKNN
rknn = RKNN(verbose=False)
rknn.load_rknn('../models/porting/BCResNet-t2-Focal-ep110.rknn')
rknn.init_runtime(target='rk3588')

# 1. Try passing NCHW (current)
out_nchw = rknn.inference(inputs=[dummy_input_nchw], data_format='nchw')[0]

# 2. Try passing NHWC (Shape: 1, 40, 151, 1). RKNN will auto transpose to NCHW internally if the model expects NCHW.
dummy_input_nhwc = np.transpose(dummy_input_nchw, (0, 2, 3, 1))
out_nhwc = rknn.inference(inputs=[dummy_input_nhwc], data_format='nhwc')[0]

print("ONNX: ", onnx_out)
print("RKNN (NCHW): ", out_nchw)
print("RKNN (NHWC): ", out_nhwc)

rknn.release()

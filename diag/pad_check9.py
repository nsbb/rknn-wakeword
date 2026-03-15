from rknn.api import RKNN
import numpy as np
import onnxruntime as ort

dummy_input_nchw = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)

print("--- Testing FP16 Model ---")
rknn_fp16 = RKNN(verbose=False)
rknn_fp16.load_rknn('../models/porting/BCResNet-t2-Focal-ep110.rknn')
rknn_fp16.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_0_1_2)
out_fp16 = rknn_fp16.inference(inputs=[dummy_input_nchw], data_format='nchw')[0]
rknn_fp16.release()
print("Target FP16 output: ", out_fp16)


print("--- Testing FP32 Model Conversion & Targeting ---")
rknn_fp32 = RKNN(verbose=False)
rknn_fp32.config(mean_values=[[0]], std_values=[[1]], target_platform='rk3588')
rknn_fp32.load_onnx('../models/BCResNet-t2-Focal-ep110.onnx')
rknn_fp32.build(do_quantization=False)
rknn_fp32.init_runtime(target='rk3588', core_mask=RKNN.NPU_CORE_0)
out_fp32 = rknn_fp32.inference(inputs=[dummy_input_nchw], data_format='nchw')[0]
rknn_fp32.release()
print("Target FP32 output: ", out_fp32)

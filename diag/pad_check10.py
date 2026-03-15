from rknn.api import RKNN
import numpy as np

dummy_input_nchw = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)

print("--- Testing Target NPU execution VS CPU Execution in API ---")
rknn_fp16 = RKNN(verbose=False)
rknn_fp16.load_rknn('../models/porting/BCResNet-t2-Focal-ep110.rknn')
# Try using CPU on target
try:
    rknn_fp16.init_runtime(target='rk3588', core_mask=0) # core_mask=0 stands for CPU
    out_fp16_cpu = rknn_fp16.inference(inputs=[dummy_input_nchw], data_format='nchw')[0]
    print("Target CPU output: ", out_fp16_cpu)
except Exception as e:
    print("CPU execution error: ", e)

rknn_fp16.release()

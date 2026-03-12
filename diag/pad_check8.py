from rknn.api import RKNN
import numpy as np

dummy_input_nchw = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)

rknn_sim = RKNN(verbose=False)
rknn_sim.config(mean_values=[[0]], std_values=[[1]], target_platform='rk3588')
rknn_sim.load_onnx('BCResNet-t2-Focal-ep110.onnx')
rknn_sim.build(do_quantization=False)

# Simulator inference
rknn_sim.init_runtime()
out_sim = rknn_sim.inference(inputs=[dummy_input_nchw], data_format='nchw')[0]

print("Simulator output: ", out_sim)
rknn_sim.release()


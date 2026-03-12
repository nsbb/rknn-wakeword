import numpy as np

# from convert_to_rknn.py
dummy_input = np.random.uniform(0, 1, (1, 1, 40, 151)).astype(np.float32)

from inference_rknn import RKNNInferenceEngine
import onnxruntime as ort

ort_session = ort.InferenceSession('BCResNet-t2-Focal-ep110.onnx')
onnx_input = {ort_session.get_inputs()[0].name: dummy_input}
onnx_out = ort_session.run(None, onnx_input)[0]

engine = RKNNInferenceEngine('BCResNet-t2-Focal-ep110.rknn', target='rk3588')
engine.load_model()
outputs_rknn = engine.rknn.inference(inputs=[dummy_input], data_format='nchw')

print("ONNX Output: ", onnx_out)
print("RKNN Output: ", outputs_rknn[0])


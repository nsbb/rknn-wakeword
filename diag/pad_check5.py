import onnx
import onnxruntime as ort
import numpy as np

model_path = 'BCResNet-t2-Focal-ep110.onnx'
ort_session = ort.InferenceSession(model_path)
inputs = ort_session.get_inputs()
print("ONNX Inputs:")
for i in inputs:
    print(i.name, i.shape, i.type)

outputs = ort_session.get_outputs()
print("ONNX Outputs:")
for o in outputs:
    print(o.name, o.shape, o.type)


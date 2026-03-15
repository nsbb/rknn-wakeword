"""
Zero input → ONNX와 NPU 비교, FC bias 확인
"""
import numpy as np, sys
sys.path.insert(0, '.')
from rknnlite.api import RKNNLite
import onnxruntime as ort
import onnx
from onnx import numpy_helper

# ONNX with zero input
sess = ort.InferenceSession('../models/BCResNet-t2-npu-fixed.onnx')
zero = np.zeros((1,1,40,151), dtype=np.float32)
onnx_zero = sess.run(None, {'input': zero})[0]
print(f'ONNX(zeros): {onnx_zero.squeeze()}')

# FC bias (last Conv/Gemm layer)
model = onnx.load('../models/BCResNet-t2-npu-fixed.onnx')
last_node = model.graph.node[-1]
print(f'Last op: {last_node.op_type} - {last_node.name}')

# 초기화값에서 마지막 classifier Conv의 bias 찾기
for init in model.graph.initializer:
    if 'classifier' in init.name and 'bias' in init.name.lower():
        arr = numpy_helper.to_array(init)
        print(f'  Found bias: {init.name} = {arr}')

# NPU probs
rknn_lite = RKNNLite(verbose=False)
rknn_lite.load_rknn('../models/BCResNet-t2-npu-fixed.rknn')
rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
raw_npu = rknn_lite.inference(inputs=[zero], data_format='nchw')[0]
print(f'NPU(zeros):  {raw_npu.squeeze()}')
print(f'Match (ONNX==NPU?): {np.allclose(onnx_zero.squeeze(), raw_npu.squeeze(), atol=0.01)}')
rknn_lite.release()

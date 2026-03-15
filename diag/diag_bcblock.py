"""
BCBlock 핵심 패턴 테스트: depthwise Conv(H,1) + broadcast Add
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

C, H, W = 16, 20, 151

# Model: Conv → depthwiseConv(H,1) → Add(broadcast) → ReLU → output
w_conv = np.random.randn(C, 1, 3, 3).astype(np.float32) * 0.1
w_dw   = (1.0/H) * np.ones((C, 1, H, 1), dtype=np.float32)

model = helper.make_model(
    helper.make_graph(
        nodes=[
            helper.make_node('Conv', ['input','w_conv'], ['conv_out'],
                             kernel_shape=[3,3], pads=[1,1,1,1], group=1),
            helper.make_node('Conv', ['conv_out','w_dw'], ['dw_out'],
                             kernel_shape=[H,1], pads=[0,0,0,0], group=C,
                             dilations=[1,1], strides=[1,1]),
            helper.make_node('Add', ['conv_out','dw_out'], ['add_out']),
            helper.make_node('Relu', ['add_out'], ['output']),
        ],
        name='bcblock_test',
        inputs=[helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,1,H,W])],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,C,H,W])],
        initializer=[
            numpy_helper.from_array(w_conv, name='w_conv'),
            numpy_helper.from_array(w_dw,   name='w_dw'),
        ]
    ),
    opset_imports=[helper.make_opsetid("", 13)]
)
model = shape_inference.infer_shapes(model)
onnx.save(model, '../models/porting/test_bcblock.onnx')

feat = np.random.randn(1,1,H,W).astype(np.float32)
sess = ort.InferenceSession('../models/porting/test_bcblock.onnx')
out = sess.run(None, {'input': feat})[0]
print(f'ONNX bcblock: shape={out.shape}, range=[{out.min():.3f},{out.max():.3f}]')
print('Saved: test_bcblock.onnx')

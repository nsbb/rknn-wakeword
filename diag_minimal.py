"""
최소 모델 NPU 테스트 - C=1 float32 input으로 단순 Conv 동작 확인
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

def make_test_model(name, input_shape, ops_fn):
    """Create minimal ONNX model"""
    return ops_fn(name, input_shape)

# Model 1: Single Conv, C=1 input
def model_single_conv(name, inp_shape=(1,1,40,151)):
    C_in, H, W = inp_shape[1], inp_shape[2], inp_shape[3]
    C_out = 16
    kernel_h, kernel_w = 3, 3
    # Weight
    w = np.random.randn(C_out, C_in, kernel_h, kernel_w).astype(np.float32) * 0.1
    b = np.random.randn(C_out).astype(np.float32) * 0.1

    graph = helper.make_graph(
        nodes=[
            helper.make_node('Conv', inputs=['input','weight','bias'], outputs=['output'],
                             kernel_shape=[kernel_h,kernel_w], pads=[1,1,1,1])
        ],
        name=name,
        inputs=[helper.make_tensor_value_info('input', TensorProto.FLOAT, list(inp_shape))],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,C_out,H,W])],
        initializer=[
            numpy_helper.from_array(w, name='weight'),
            numpy_helper.from_array(b, name='bias'),
        ]
    )
    return onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# Model 2: Conv + BN (standalone, no reshape)
def model_conv_bn(name, inp_shape=(1,1,40,151)):
    C_in, H, W = inp_shape[1], inp_shape[2], inp_shape[3]
    C_out = 16
    w = np.random.randn(C_out, C_in, 3, 3).astype(np.float32) * 0.1
    scale = np.ones(C_out).astype(np.float32)
    bias = np.zeros(C_out).astype(np.float32)
    mean = np.zeros(C_out).astype(np.float32)
    var = np.ones(C_out).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node('Conv', inputs=['input','weight'], outputs=['conv_out'],
                             kernel_shape=[3,3], pads=[1,1,1,1]),
            helper.make_node('BatchNormalization', inputs=['conv_out','scale','bias','mean','var'],
                             outputs=['output']),
        ],
        name=name,
        inputs=[helper.make_tensor_value_info('input', TensorProto.FLOAT, list(inp_shape))],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,C_out,H,W])],
        initializer=[
            numpy_helper.from_array(w, name='weight'),
            numpy_helper.from_array(scale, name='scale'),
            numpy_helper.from_array(bias, name='bias'),
            numpy_helper.from_array(mean, name='mean'),
            numpy_helper.from_array(var, name='var'),
        ]
    )
    return onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

models = {
    'test_conv_c1': model_single_conv('test1', (1,1,40,151)),
    'test_conv_bn_c1': model_conv_bn('test2', (1,1,40,151)),
    'test_conv_c16': model_single_conv('test3', (1,16,40,151)),
}

feat = np.random.randn(1,1,40,151).astype(np.float32)

for mname, model in models.items():
    model = shape_inference.infer_shapes(model)
    onnx.save(model, f'{mname}.onnx')
    sess = ort.InferenceSession(f'{mname}.onnx')
    out = sess.run(None, {'input': feat[:, :model.graph.input[0].type.tensor_type.shape.dim[1].dim_value, :, :]})[0]
    print(f'{mname}: ONNX output shape={out.shape}, range=[{out.min():.3f},{out.max():.3f}]')

print('Saved test models')

"""
다중 BCBlock 누적 테스트: 전체 모델 구조 점진적 확인
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference
import onnxruntime as ort

def make_bcblock_stack(name, n_blocks=2, C_in=1, C_feat=16, H=20, W=151):
    """n_blocks BCBlocks stacked"""
    np.random.seed(42)
    nodes = []
    inits = []

    # Initial conv to expand channels
    w0 = np.random.randn(C_feat, C_in, 3, 3).astype(np.float32) * 0.05
    inits.append(numpy_helper.from_array(w0, name='w_init'))
    nodes.append(helper.make_node('Conv', ['input','w_init'], ['x'],
                                  kernel_shape=[3,3], pads=[1,1,1,1]))

    prev = 'x'
    for i in range(n_blocks):
        C = C_feat
        # BCBlock f2 branch: depthwise conv 3x3 + broadcast ReduceMean (depthwise H×1) + Add
        w_f2 = np.random.randn(C, 1, 3, 1).astype(np.float32) * 0.05
        w_rm = (1.0/H) * np.ones((C, 1, H, 1), dtype=np.float32)
        inits += [
            numpy_helper.from_array(w_f2, name=f'w_f2_{i}'),
            numpy_helper.from_array(w_rm, name=f'w_rm_{i}'),
        ]
        nodes += [
            helper.make_node('Conv', [prev, f'w_f2_{i}'], [f'f2_{i}'],
                             kernel_shape=[3,1], pads=[1,0,1,0], group=C),
            helper.make_node('Conv', [f'f2_{i}', f'w_rm_{i}'], [f'rm_{i}'],
                             kernel_shape=[H,1], pads=[0,0,0,0], group=C,
                             dilations=[1,1], strides=[1,1]),
            helper.make_node('Add', [f'f2_{i}', f'rm_{i}'], [f'add_{i}']),
            helper.make_node('Relu', [f'add_{i}'], [f'relu_{i}']),
        ]
        prev = f'relu_{i}'

    # Final conv to 2 classes
    w_cls = np.random.randn(2, C_feat, 1, 1).astype(np.float32)
    inits.append(numpy_helper.from_array(w_cls, name='w_cls'))
    nodes += [
        helper.make_node('Conv', [prev, 'w_cls'], ['logits'], kernel_shape=[1,1]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=[helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,C_in,H,W])],
        outputs=[helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1,2,H,W])],
        initializer=inits,
    )
    return onnx.helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)])

feat = np.random.randn(1,1,20,151).astype(np.float32)

for n_blocks in [1, 2, 4, 8]:
    model = make_bcblock_stack(f'stack_{n_blocks}', n_blocks=n_blocks)
    model = shape_inference.infer_shapes(model)
    onnx.save(model, f'../models/porting/test_stack_{n_blocks}.onnx')
    sess = ort.InferenceSession(f'../models/porting/test_stack_{n_blocks}.onnx')
    out = sess.run(None, {'input': feat})[0]
    print(f'stack_{n_blocks}: ONNX range=[{out.min():.3f},{out.max():.3f}]')

print('Saved stack models')

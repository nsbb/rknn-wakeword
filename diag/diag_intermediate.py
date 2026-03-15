"""
모델 중간 출력 추출 - 어느 레이어에서 NPU가 틀리는지 확인
no-SSN 버전 사용 (Reshape 없음)
"""
import numpy as np
import onnx
from onnx import helper, shape_inference
import onnxruntime as ort
import sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

# no-SSN 모델에서 중간 출력 tensor 이름 수집
model = onnx.load('../models/porting/BCResNet-t2-no-ssn.onnx')
model = shape_inference.infer_shapes(model)
graph = model.graph

shapes = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    try:
        s = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = s
    except:
        pass

# 각 Conv output 목록 (주요 레이어)
target_tensors = []
for node in graph.node:
    if node.op_type in ['Conv', 'Add', 'Relu', 'Mul', 'Sigmoid']:
        out = node.output[0]
        s = shapes.get(out)
        if s:
            target_tensors.append((node.op_type, node.name, out, s))

print("=== Key intermediate tensors ===")
for i, (op, name, tensor, shape) in enumerate(target_tensors[:20]):
    print(f"  [{i:2d}] {op} → {tensor[:60]}: {shape}")

# 최초 몇 개 Conv 출력에서 ONNX vs RKNN 비교용 슬라이스 모델 생성
# 첫 번째 Conv 출력 확인
first_conv_out = None
for node in graph.node:
    if node.op_type == 'Conv':
        first_conv_out = node.output[0]
        first_conv_shape = shapes.get(first_conv_out)
        print(f"\nFirst Conv: {node.name} → {first_conv_out}: {first_conv_shape}")
        break

# 첫 번째 Add 출력
first_add_out = None
for node in graph.node:
    if node.op_type == 'Add':
        first_add_out = node.output[0]
        first_add_shape = shapes.get(first_add_out)
        print(f"First Add: {node.name} → {first_add_out}: {first_add_shape}")
        break

# 첫 번째 Relu 출력
first_relu_out = None
for node in graph.node:
    if node.op_type == 'Relu':
        first_relu_out = node.output[0]
        first_relu_shape = shapes.get(first_relu_out)
        print(f"First Relu: {node.name} → {first_relu_out}: {first_relu_shape}")
        break

# 첫 번째 Sigmoid 출력
first_sigmoid_out = None
for node in graph.node:
    if node.op_type == 'Sigmoid':
        first_sigmoid_out = node.output[0]
        first_sigmoid_shape = shapes.get(first_sigmoid_out)
        print(f"First Sigmoid: {node.name} → {first_sigmoid_out}: {first_sigmoid_shape}")
        break

# 각 중간 출력을 ONNX에서 추출
def get_onnx_intermediate(model, feat, output_name):
    """ONNX 모델의 특정 중간 output 추출"""
    m = onnx.ModelProto()
    m.CopyFrom(model)
    m.graph.output.append(helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString())
    outputs = sess.run([output_name], {'input': feat})
    return outputs[0]

for name, tensor_name, shape in [
    ('First Conv', first_conv_out, first_conv_shape),
    ('First Add', first_add_out, first_add_shape),
    ('First Relu', first_relu_out, first_relu_shape),
    ('First Sigmoid', first_sigmoid_out, first_sigmoid_shape),
]:
    if tensor_name is None:
        continue
    out = get_onnx_intermediate(model, feat, tensor_name)
    print(f"\n{name} ({shape}): range=[{out.min():.3f},{out.max():.3f}], nonzero={np.count_nonzero(out)}")

# 슬라이스 모델 생성: 입력~첫번째 Conv까지만
if first_conv_out:
    slice_graph = onnx.ModelProto()
    slice_graph.CopyFrom(model)
    # output을 first_conv_out으로 교체
    del slice_graph.graph.output[:]
    slice_graph.graph.output.append(
        helper.make_tensor_value_info(first_conv_out, onnx.TensorProto.FLOAT, first_conv_shape)
    )
    # 필요없는 노드 제거 - first_conv_out 이후 노드 제거
    slice_graph = shape_inference.infer_shapes(slice_graph)
    onnx.save(slice_graph, '../models/porting/test_slice_conv1.onnx')
    print(f'\nSaved: test_slice_conv1.onnx → output={first_conv_out}')

# ReduceMean 교체 Conv (Broadcast residual) 출력 슬라이스
rm_out = None
for node in graph.node:
    if 'ReduceMean' in node.name and node.op_type == 'Conv':
        rm_out = node.output[0]
        rm_shape = shapes.get(rm_out)
        print(f"First ReduceMean Conv: {node.name} → {rm_out}: {rm_shape}")
        # Slice 모델
        slice_graph2 = onnx.ModelProto()
        slice_graph2.CopyFrom(model)
        del slice_graph2.graph.output[:]
        slice_graph2.graph.output.append(
            helper.make_tensor_value_info(rm_out, onnx.TensorProto.FLOAT, rm_shape)
        )
        onnx.save(slice_graph2, '../models/porting/test_slice_rm.onnx')
        print(f'Saved: test_slice_rm.onnx → output={rm_out}')
        break

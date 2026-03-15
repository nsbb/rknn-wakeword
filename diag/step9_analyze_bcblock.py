"""BCBlocks.0.0 ReduceMean 전후 그래프 구조 상세 분석"""
import onnx
from onnx import shape_inference

model_path = '../models/porting/BCResNet-t2-rknn-compatible.onnx'
model = onnx.load(model_path)
model = shape_inference.infer_shapes(model)
graph = model.graph

# name → shape 맵
shape_map = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    shape = []
    if vi.type.HasField('tensor_type') and vi.type.tensor_type.HasField('shape'):
        for dim in vi.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.HasField('dim_value') else dim.dim_param)
    shape_map[vi.name] = shape

# BCBlocks.0.0 관련 노드 찾기
print("BCBlocks.0.0 관련 노드:")
for i, node in enumerate(graph.node):
    if 'BCBlocks.0.0' in node.name:
        inp_shapes = [f"{inp}:{shape_map.get(inp, '?')}" for inp in node.input[:2]]
        out_shapes = [f"{out}:{shape_map.get(out, '?')}" for out in node.output[:1]]
        print(f"  [{i:3d}] {node.op_type:25s} {node.name}")
        print(f"        in: {inp_shapes}")
        print(f"        out: {out_shapes}")

# AveragePool 직전/직후 노드 확인
print("\n\nAveragePool 노드 상세:")
for i, node in enumerate(graph.node):
    if node.op_type == 'AveragePool':
        print(f"\n  [{i}] {node.name}")
        print(f"    attrs: {dict((a.name, list(a.ints) if a.ints else a.i) for a in node.attribute)}")
        print(f"    input: {node.input[0]} shape={shape_map.get(node.input[0], '?')}")
        print(f"    output: {node.output[0]} shape={shape_map.get(node.output[0], '?')}")

        # 이 AveragePool 출력을 사용하는 노드 찾기
        out_name = node.output[0]
        consumers = [n for n in graph.node if out_name in n.input]
        print(f"    consumers: {[(n.op_type, n.name) for n in consumers]}")

        # 이 AveragePool 입력을 생성하는 노드
        inp_name = node.input[0]
        producers = [n for n in graph.node if inp_name in n.output]
        print(f"    producers: {[(n.op_type, n.name) for n in producers]}")

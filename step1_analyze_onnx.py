"""Step 1: ONNX 그래프 분석 - ReduceMean 노드 위치 및 shape 파악"""
import onnx
import numpy as np
from onnx import shape_inference

model_path = '/home/rk3588/travail/rk3588/Rockchip_VT/BCResNet-t2-Focal-ep110.onnx'
model = onnx.load(model_path)

# shape inference 실행
model = shape_inference.infer_shapes(model)
graph = model.graph

# value_info + input + output → name→shape 매핑 구성
shape_map = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    shape = []
    if vi.type.HasField('tensor_type') and vi.type.tensor_type.HasField('shape'):
        for dim in vi.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            elif dim.HasField('dim_param'):
                shape.append(dim.dim_param)
            else:
                shape.append('?')
    shape_map[vi.name] = shape

# ReduceMean 노드 분석
print("=" * 70)
print("ReduceMean 노드 분석")
print("=" * 70)
reduce_mean_nodes = []
for i, node in enumerate(graph.node):
    if node.op_type == 'ReduceMean':
        # attributes
        axes = None
        keepdims = 1
        for attr in node.attribute:
            if attr.name == 'axes':
                axes = list(attr.ints)
            elif attr.name == 'keepdims':
                keepdims = attr.i

        inp_name = node.input[0] if node.input else "?"
        out_name = node.output[0] if node.output else "?"
        inp_shape = shape_map.get(inp_name, ['?'])
        out_shape = shape_map.get(out_name, ['?'])

        print(f"\n[{i}] Node: {node.name}")
        print(f"    axes={axes}, keepdims={keepdims}")
        print(f"    input : {inp_name} shape={inp_shape}")
        print(f"    output: {out_name} shape={out_shape}")

        reduce_mean_nodes.append({
            'node_idx': i,
            'node': node,
            'axes': axes,
            'keepdims': keepdims,
            'inp_name': inp_name,
            'out_name': out_name,
            'inp_shape': inp_shape,
            'out_shape': out_shape,
        })

print(f"\n총 ReduceMean 노드 수: {len(reduce_mean_nodes)}")

# 전체 노드 op_type 분포
from collections import Counter
op_counts = Counter(n.op_type for n in graph.node)
print("\n전체 op_type 분포:")
for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
    print(f"  {op}: {cnt}")

# 모델 입력/출력 shape 확인
print("\n모델 입력:")
for inp in graph.input:
    shape = shape_map.get(inp.name, [])
    print(f"  {inp.name}: {shape}")
print("모델 출력:")
for out in graph.output:
    shape = shape_map.get(out.name, [])
    print(f"  {out.name}: {shape}")

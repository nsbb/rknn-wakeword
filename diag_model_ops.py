"""
ONNX 모델 내 모든 op 분석 - ReduceMean, Reshape 위치와 shape 확인
"""
import onnx
from onnx import shape_inference
import numpy as np

model = onnx.load('BCResNet-t2-Focal-ep110.onnx')
model = shape_inference.infer_shapes(model)

graph = model.graph

# 모든 value_info shape 수집
shapes = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    try:
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = shape
    except:
        pass

print("=== ReduceMean nodes ===")
for node in graph.node:
    if node.op_type == 'ReduceMean':
        axes = list(node.attribute[0].ints) if node.attribute else 'N/A'
        in_shape = shapes.get(node.input[0], 'unknown')
        out_shape = shapes.get(node.output[0], 'unknown')
        print(f"  {node.name}: axes={axes}, in={in_shape} → out={out_shape}")

print("\n=== Reshape nodes ===")
for node in graph.node:
    if node.op_type == 'Reshape':
        in_shape = shapes.get(node.input[0], 'unknown')
        out_shape = shapes.get(node.output[0], 'unknown')
        print(f"  {node.name}: in={in_shape} → out={out_shape}")

print("\n=== AveragePool nodes ===")
for node in graph.node:
    if node.op_type == 'AveragePool':
        in_shape = shapes.get(node.input[0], 'unknown')
        out_shape = shapes.get(node.output[0], 'unknown')
        attrs = {a.name: list(a.ints) for a in node.attribute}
        print(f"  {node.name}: {attrs}, in={in_shape} → out={out_shape}")

print("\n=== All unique op types ===")
ops = {}
for node in graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1
for op, cnt in sorted(ops.items()):
    print(f"  {op}: {cnt}")

print(f"\nTotal nodes: {len(graph.node)}")

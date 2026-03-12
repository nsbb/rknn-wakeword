"""
npu-fixed.onnx 전체 op 확인 - Sigmoid, Mul, Conv 등
"""
import onnx
from onnx import shape_inference
import numpy as np

model = onnx.load('BCResNet-t2-npu-fixed.onnx')
model = shape_inference.infer_shapes(model)
graph = model.graph

shapes = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    try:
        s = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = s
    except:
        pass

# Sigmoid + Mul 패턴 확인
print("=== Sigmoid nodes ===")
for node in graph.node:
    if node.op_type == 'Sigmoid':
        in_shape = shapes.get(node.input[0], '?')
        out_shape = shapes.get(node.output[0], '?')
        print(f"  {node.name}: in={in_shape} → out={out_shape}")

print("\n=== Mul nodes ===")
for node in graph.node:
    if node.op_type == 'Mul':
        in0_shape = shapes.get(node.input[0], '?')
        in1_shape = shapes.get(node.input[1], '?')
        out_shape = shapes.get(node.output[0], '?')
        print(f"  {node.name}: in0={in0_shape}, in1={in1_shape} → out={out_shape}")

print("\n=== Depthwise Conv (my ReduceMean replacements) ===")
for node in graph.node:
    if node.op_type == 'Conv':
        # Check group attribute
        group = 1
        for attr in node.attribute:
            if attr.name == 'group':
                group = attr.i
        if group > 1:
            in_shape = shapes.get(node.input[0], '?')
            out_shape = shapes.get(node.output[0], '?')
            ks = []
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    ks = list(attr.ints)
            print(f"  {node.name}: group={group}, kernel={ks}, in={in_shape} → out={out_shape}")

print("\n=== All unique op types ===")
ops = {}
for node in graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1
for op, cnt in sorted(ops.items()):
    print(f"  {op}: {cnt}")

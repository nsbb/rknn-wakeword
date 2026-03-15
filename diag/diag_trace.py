"""
BCResNet ONNX 그래프 첫 20개 op trace
"""
import onnx
from onnx import shape_inference

model = onnx.load('../models/BCResNet-t2-npu-fixed.onnx')
model = shape_inference.infer_shapes(model)
graph = model.graph

shapes = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    try:
        s = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = s
    except:
        pass

print("=== First 25 nodes in graph ===")
for i, node in enumerate(graph.node[:25]):
    ins = [(n, shapes.get(n,'?')) for n in node.input if n and not n.startswith('backbone.')]
    outs = [(n, shapes.get(n,'?')) for n in node.output if n]
    print(f"[{i:2d}] {node.op_type:20s} {node.name[:50]}")
    for n,s in ins:
        print(f"      IN:  {n[:60]} {s}")
    for n,s in outs:
        print(f"      OUT: {n[:60]} {s}")

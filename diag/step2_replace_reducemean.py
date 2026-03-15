"""Step 2: ReduceMean → AveragePool 교체
- axes=[2], keepdims=1: AveragePool(kernel=[H,1], stride=[1,1], pads=[0,0,0,0])
- axes=[2,3], keepdims=1: AveragePool(kernel=[H,W], stride=[1,1]) → GlobalAveragePool
"""
import onnx
import onnx_graphsurgeon as gs
import numpy as np

model_path = '../models/BCResNet-t2-Focal-ep110.onnx'
output_path = '../models/porting/BCResNet-t2-rknn-compatible.onnx'

print("Loading model...")
model = onnx.load(model_path)
graph = gs.import_onnx(model)

# shape inference로 tensor shape 파악
import onnx
from onnx import shape_inference
model_si = shape_inference.infer_shapes(onnx.load(model_path))
shape_map = {}
for vi in list(model_si.graph.value_info) + list(model_si.graph.input) + list(model_si.graph.output):
    shape = []
    if vi.type.HasField('tensor_type') and vi.type.tensor_type.HasField('shape'):
        for dim in vi.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(None)
    shape_map[vi.name] = shape

print(f"\n그래프 노드 수: {len(graph.nodes)}")

# ReduceMean 노드를 찾아서 AveragePool로 교체
replaced = 0
nodes_to_remove = []
nodes_to_add = []

# graphsurgeon의 tensor name → tensor 매핑
tensor_map = {t.name: t for t in graph.tensors().values()}

for node in graph.nodes:
    if node.op != 'ReduceMean':
        continue

    # attributes 파악
    axes_attr = node.attrs.get('axes', None)
    keepdims_attr = node.attrs.get('keepdims', 1)

    # graphsurgeon attrs는 raw value 또는 AttributeRef일 수 있음
    # list/int로 변환
    if hasattr(axes_attr, 'values'):
        axes = list(axes_attr.values)
    elif isinstance(axes_attr, (list, tuple)):
        axes = list(axes_attr)
    else:
        axes = axes_attr  # None

    if hasattr(keepdims_attr, 'value'):
        keepdims = int(keepdims_attr.value)
    else:
        keepdims = int(keepdims_attr) if keepdims_attr is not None else 1

    inp = node.inputs[0]
    out = node.outputs[0]
    inp_shape = shape_map.get(inp.name, [])

    print(f"\nReplacing ReduceMean: {node.name}")
    print(f"  axes={axes}, keepdims={keepdims}")
    print(f"  input shape: {inp_shape}")

    if axes == [2] and keepdims == 1:
        # H 차원만 평균: (1,C,H,W) → (1,C,1,W)
        if len(inp_shape) >= 3 and inp_shape[2] is not None:
            H = inp_shape[2]
        else:
            print(f"  WARNING: cannot determine H for {node.name}, skipping")
            continue

        print(f"  → AveragePool(kernel=[{H},1], stride=[1,1], pad=0)")
        node.op = 'AveragePool'
        node.attrs = {
            'kernel_shape': [H, 1],
            'strides': [1, 1],
            'pads': [0, 0, 0, 0],
            'count_include_pad': 0,
        }
        replaced += 1

    elif axes == [2, 3] and keepdims == 1:
        # H,W 모두 평균
        if len(inp_shape) >= 4 and inp_shape[2] is not None and inp_shape[3] is not None:
            H, W = inp_shape[2], inp_shape[3]
        else:
            print(f"  WARNING: cannot determine H,W for {node.name}, skipping")
            continue

        print(f"  → AveragePool(kernel=[{H},{W}], stride=[1,1], pad=0)")
        node.op = 'AveragePool'
        node.attrs = {
            'kernel_shape': [H, W],
            'strides': [1, 1],
            'pads': [0, 0, 0, 0],
            'count_include_pad': 0,
        }
        replaced += 1
    else:
        print(f"  WARNING: unexpected axes={axes}, skipping")

print(f"\n총 {replaced}개 ReduceMean → AveragePool 교체 완료")

# 그래프 정리 및 export
graph.cleanup().toposort()
model_out = gs.export_onnx(graph)

# opset 확인
print(f"\nOpset: {model_out.opset_import[0].version}")

# 저장
onnx.save(model_out, output_path)
print(f"\n저장 완료: {output_path}")

# 검증 (onnx checker)
try:
    onnx.checker.check_model(model_out)
    print("ONNX checker: OK")
except Exception as e:
    print(f"ONNX checker error: {e}")

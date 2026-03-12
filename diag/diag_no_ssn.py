"""
SubSpectralNorm (Reshape→BN→Reshape) 제거 테스트
- NPU 출력이 non-constant이면 SSN이 원인
"""
import numpy as np, sys, wave
sys.path.insert(0, '.')
from inference_rknn import LogMel
from rknnlite.api import RKNNLite
import onnxruntime as ort
import onnx
from onnx import helper, shape_inference

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

with wave.open('wallpad_HiWonder_251113/lkk/lkk_1_2.wav', 'rb') as wf:
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
feat = LogMel()(audio)[np.newaxis, np.newaxis, ...]

model = onnx.load('BCResNet-t2-npu-fixed.onnx')
model = shape_inference.infer_shapes(model)
graph = model.graph

# shape 수집
shapes = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    try:
        s = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = s
    except:
        pass

# tensor name → node that produces it
producer = {}
for node in graph.node:
    for out in node.output:
        if out:
            producer[out] = node

# tensor name → nodes that consume it
consumer = {}
for node in graph.node:
    for inp in node.input:
        if inp not in consumer:
            consumer[inp] = []
        consumer[inp].append(node)

# Reshape→BN→Reshape 패턴 찾기
bypasses = {}  # reshape2_output → reshape1_input
nodes_to_skip = set()

for node in graph.node:
    if node.op_type != 'Reshape':
        continue
    reshape1_out = node.output[0]
    # find consumer(s) of reshape1_out
    consumers1 = consumer.get(reshape1_out, [])
    bn_nodes = [n for n in consumers1 if n.op_type == 'BatchNormalization']
    if not bn_nodes:
        continue
    bn_node = bn_nodes[0]
    bn_out = bn_node.output[0]
    # find consumer(s) of bn_out
    consumers2 = consumer.get(bn_out, [])
    reshape2_nodes = [n for n in consumers2 if n.op_type == 'Reshape']
    if not reshape2_nodes:
        continue
    reshape2_node = reshape2_nodes[0]
    in_shape = shapes.get(node.input[0], '?')
    out_shape = shapes.get(reshape2_node.output[0], '?')
    print(f'  SSN: {node.name} in={in_shape} → out={out_shape}')
    bypasses[reshape2_node.output[0]] = node.input[0]
    nodes_to_skip.update([id(node), id(bn_node), id(reshape2_node)])

print(f'Total SSN blocks to bypass: {len(bypasses)}')

# input 이름 교체
for bypass_out, bypass_in in bypasses.items():
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == bypass_out:
                node.input[i] = bypass_in

# 스킵할 노드 제거
new_nodes = [n for n in graph.node if id(n) not in nodes_to_skip]
del graph.node[:]
graph.node.extend(new_nodes)

model = shape_inference.infer_shapes(model)
onnx.save(model, 'BCResNet-t2-no-ssn.onnx')

# ONNX 확인
sess_ref = ort.InferenceSession('BCResNet-t2-npu-fixed.onnx')
ref_out = sess_ref.run(None, {sess_ref.get_inputs()[0].name: feat})[0]
print(f'Reference  ONNX probs: {softmax(ref_out.squeeze())}')

sess = ort.InferenceSession('BCResNet-t2-no-ssn.onnx')
out = sess.run(None, {sess.get_inputs()[0].name: feat})[0]
print(f'No-SSN     ONNX probs: {softmax(out.squeeze())} (accuracy different, OK)')
print('Saved: BCResNet-t2-no-ssn.onnx')

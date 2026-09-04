#!/usr/bin/env python3
"""Architectural GFLOPs per inference from an ONNX file — the Score's workload constant.

Counts Conv / Gemm / MatMul (the tensor-core-shaped work) by the naive convention
(FMA = 2 ops) at the graph's declared input shapes. This is the ARCHITECTURAL count
for the mix manifest's arch_gflops column — never profiler-measured executed FLOPs.

Usage:
  python3 compute_arch_gflops.py model.onnx [--shape input_name:1x3x512x640]

Dynamic dims default to 1; override with --shape (repeatable). Output notes what
fraction of nodes were counted — non-matmul ops (softmax, norm, resize) carry few
FLOPs and are excluded by convention, matching the pinned mix definition.
Self-check: gemm_selftest.onnx (4096^3 MatMul) must report 137.44 GFLOPs.
"""
import sys, argparse
import onnx
from onnx import shape_inference

def dims_of(vi, overrides, default=1):
    name = vi.name
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if d.dim_value > 0:
            dims.append(d.dim_value)
        else:
            dims.append(overrides.get(name, {}).get(len(dims), default))
    return dims

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--shape', action='append', default=[],
                    help='name:AxBxC to pin dynamic input dims')
    a = ap.parse_args()
    overrides = {}
    for spec in a.shape:
        name, dims = spec.rsplit(':', 1)
        overrides[name] = {i: int(v) for i, v in enumerate(dims.lower().split('x'))}

    m = onnx.load(a.model)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception as e:
        print(f'WARN: shape inference partial ({e}); dynamic dims default to 1', file=sys.stderr)

    # tensor name -> dims (inputs, initializers, value_info, outputs)
    shapes = {}
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        shapes[vi.name] = dims_of(vi, overrides)
    for init in m.graph.initializer:
        shapes[init.name] = list(init.dims)

    total = 0.0
    counted, skipped = 0, 0
    for node in m.graph.node:
        try:
            if node.op_type == 'Conv':
                w = shapes[node.input[1]]                 # [Cout, Cin/g, kh, kw]
                out = shapes[node.output[0]]              # [N, Cout, Ho, Wo]
                spatial_out = 1
                for d in out[2:]: spatial_out *= d
                kernel = 1
                for d in w[1:]: kernel *= d               # Cin/g * kh * kw
                total += 2.0 * out[0] * w[0] * kernel * spatial_out
                counted += 1
            elif node.op_type in ('Gemm', 'MatMul'):
                A, B = shapes[node.input[0]], shapes[node.input[1]]
                if node.op_type == 'Gemm':
                    M, K = A[-2], A[-1]
                    for at in node.attribute:
                        if at.name == 'transA' and at.i: M, K = K, M
                    N = B[-1]
                    for at in node.attribute:
                        if at.name == 'transB' and at.i: N = B[-2]
                    batch = 1
                else:
                    M, K, N = A[-2], A[-1], B[-1]
                    batch = 1
                    for d in A[:-2]: batch *= d
                total += 2.0 * batch * M * K * N
                counted += 1
        except (KeyError, IndexError):
            skipped += 1
    n_all = len(m.graph.node)
    print(f'arch_gflops: {total/1e9:.2f}')
    print(f'(counted {counted} Conv/Gemm/MatMul nodes of {n_all} total; '
          f'{skipped} matmul-type nodes skipped for missing shapes — '
          f'if skipped > 0, pin shapes with --shape)')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Architectural GFLOPs per inference from an ONNX file - the Score's workload constant.

Counts Conv / Gemm / MatMul only (FMA = 2 ops) at the graph's declared shapes; dynamic dims default
to 1 unless pinned with --shape name:AxBxC (repeatable). Non-matmul ops carry few FLOPs and are
excluded by convention. Prints 'arch_gflops: X.XX' (parsed by the build scripts) then a coverage note.
Self-check: gemm_selftest.onnx (4096^3 MatMul) must report 137.44 GFLOPs.
"""
import argparse
import sys

import onnx
from onnx import shape_inference


def dims_of(value_info, overrides, default=1):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            dims.append(dim.dim_value)
        else:
            dims.append(overrides.get(value_info.name, {}).get(len(dims), default))
    return dims


def tensor_shapes(model, overrides):
    """tensor name -> dims over inputs, value_info, outputs and initializers."""
    shapes = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        shapes[value_info.name] = dims_of(value_info, overrides)
    for initializer in model.graph.initializer:
        shapes[initializer.name] = list(initializer.dims)
    return shapes


def conv_flops(node, shapes):
    weight = shapes[node.input[1]]      # [Cout, Cin/g, kh, kw]
    output = shapes[node.output[0]]     # [N, Cout, Ho, Wo]
    spatial_out = 1
    for dim in output[2:]:
        spatial_out *= dim
    kernel = 1
    for dim in weight[1:]:
        kernel *= dim
    return 2.0 * output[0] * weight[0] * kernel * spatial_out


def matmul_flops(node, shapes):
    a_shape, b_shape = shapes[node.input[0]], shapes[node.input[1]]
    if node.op_type == 'Gemm':
        m, k = a_shape[-2], a_shape[-1]
        for attribute in node.attribute:
            if attribute.name == 'transA' and attribute.i:
                m, k = k, m
        n = b_shape[-1]
        for attribute in node.attribute:
            if attribute.name == 'transB' and attribute.i:
                n = b_shape[-2]
        batch = 1
    else:
        m, k, n = a_shape[-2], a_shape[-1], b_shape[-1]
        batch = 1
        for dim in a_shape[:-2]:
            batch *= dim
    return 2.0 * batch * m * k * n


def parse_shape_overrides(specs):
    overrides = {}
    for spec in specs:
        name, dims = spec.rsplit(':', 1)
        overrides[name] = {i: int(value) for i, value in enumerate(dims.lower().split('x'))}
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--shape', action='append', default=[], help='name:AxBxC to pin dynamic input dims')
    args = parser.parse_args()
    overrides = parse_shape_overrides(args.shape)

    model = onnx.load(args.model)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f'WARN: shape inference partial ({exc}); dynamic dims default to 1', file=sys.stderr)
    shapes = tensor_shapes(model, overrides)

    total = 0.0
    counted, skipped = 0, 0
    for node in model.graph.node:
        try:
            if node.op_type == 'Conv':
                total += conv_flops(node, shapes)
                counted += 1
            elif node.op_type in ('Gemm', 'MatMul'):
                total += matmul_flops(node, shapes)
                counted += 1
        except (KeyError, IndexError):
            skipped += 1
    print(f'arch_gflops: {total / 1e9:.2f}')
    print(f'(counted {counted} Conv/Gemm/MatMul nodes of {len(model.graph.node)} total; '
          f'{skipped} matmul-type nodes skipped for missing shapes — '
          f'if skipped > 0, pin shapes with --shape)')


if __name__ == '__main__':
    main()

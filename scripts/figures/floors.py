"""Roofline latency floors and the N they imply, from a stage-4 results.json.

A measured N moves as an engine is optimized for the device, but never past
physics: optimization deletes overhead, not bytes or FLOPs. Latency therefore
has a floor and N a ceiling, both computable from the run's own numbers:

    t_floor   = max(bytes_per_frame / bw_eff, arch_gflops / compute_ceiling)
    N_ceiling = N recomputed with p99 replaced by t_floor

The compute ceiling is the burst ceiling of the row's precision, except when
the ceilings run carries a sustained median for that precision below it: a
continuous-rate row cannot ride burst recovery, so the floor uses the
sustained value (a floor from an unattainable rate would overstate N_ceiling).
t_floor is a pure roofline bound - no launch or elementwise overhead - so
N_ceiling is an upper bound, never a target.
"""
import json

CEIL_KEY = {
    'fp16': 'tensor_ceiling_fp16_tflops',
    'int8': 'tensor_ceiling_int8_tops',
    'fp8': 'tensor_ceiling_fp8_tflops',
    'fp32': 'cudacore_fp32_tflops',
}


def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def load_device(path):
    """The rows of a stage-4 results.json with the floor inputs resolved per row."""
    d = json.load(open(path))
    ceil = d.get('ceilings') or {}
    bw = num(ceil.get('bw_eff_gbps')) or num(ceil.get('dram_gbps_idle')) or num(ceil.get('dram_gbps_cpu_loaded'))
    vram_cap = num(d.get('vram_capacity_mb'))
    models = []
    for m in d.get('models', []):
        prec = (m.get('precision') or 'fp16').lower()
        comp = num(ceil.get(CEIL_KEY.get(prec, '')))
        if comp is not None and (ceil.get('sustained_prec') or '').lower() == prec:
            sust = num(ceil.get('sustained_tflops_median'))
            if sust is not None and sust < comp:
                comp = sust
        models.append({
            'name': m.get('name'),
            'p99_ms': num(m.get('p99_ms')),
            'hz': num(m.get('hz')) or 0.0,
            'deadline_ms': num(m.get('deadline_ms')),
            'arch_gflops': num(m.get('arch_gflops')) or 0.0,
            'bytes_mb': num(m.get('bytes_per_frame_MB')),
            'bytes_source': m.get('bytes_source'),
            'vram_mb': num(m.get('vram_budget_mb')) or num(m.get('vram_mb')) or 0.0,
            'precision': prec,
            'comp_ceiling': comp,
        })
    return {'path': path, 'bw': bw, 'vram_cap': vram_cap, 'models': models}


def utilization(t_ms, m, dev):
    """Budget vector for one row at latency t_ms on the device."""
    u = {'time': (t_ms / 1000.0) * m['hz']}
    if m['bytes_mb'] is not None and dev['bw']:
        u['bw'] = (m['bytes_mb'] / 1000.0) * m['hz'] / dev['bw']
    if dev['vram_cap']:
        u['vram'] = m['vram_mb'] / dev['vram_cap']
    return u


def n_at(t_ms, m, dev):
    """(L, C, N) for the row alone at latency t_ms."""
    if not t_ms or not m['deadline_ms']:
        return None, None, None
    L = m['deadline_ms'] / t_ms
    u = utilization(t_ms, m, dev)
    umax = max(u.values()) if u else 0.0
    C = (1.0 / umax) if umax > 0 else float('inf')
    return L, C, min(L, C)


def t_floor(m, dev):
    """Roofline latency floor; returns (t_ms, note)."""
    t_mem = (m['bytes_mb'] / dev['bw']) if (m['bytes_mb'] is not None and dev['bw']) else None
    t_comp = (m['arch_gflops'] / m['comp_ceiling']) if (m['arch_gflops'] and m['comp_ceiling']) else None
    if t_mem is None and t_comp is None:
        return None, 'no bytes and no compute ceiling - floor unknown'
    if t_mem is None:
        return t_comp, 'compute floor only (bytes/frame not measured)'
    if t_comp is None:
        return t_mem, 'memory floor only (no compute ceiling for this precision)'
    return max(t_mem, t_comp), ('memory-bound floor' if t_mem >= t_comp else 'compute-bound floor')

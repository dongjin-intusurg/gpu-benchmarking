#!/usr/bin/env python3
"""Stage 7.1: render the fixed figure set from data.json.

    render.py data.json --out <dir>

One function per figure; each returns None (skipped) when the stage it draws
is absent from data.json. Rows, mixes, arms, points and precisions are read
from the data - nothing here names a model, a mix or a power point. The
rendering is pinned by style.py so that the same data.json gives the same
pixels on any machine with the same matplotlib.
"""
import argparse
import json
import math
import os
import sys

import style  # noqa: F401  (selects the backend before pyplot is imported)
from style import (ACCENT, ACCENT_GHOST, ARM_COLOR, BOUND_COLOR, INK, INK2, LINESTYLES, MUT, OK, PIPE_COLOR,
                   PRECISION_COLOR, SERIES, TEAL, TRACK, WARN, apply_rc, color_for, row_height, save, style_ax,
                   subplots, top_legend)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

FIGURES = []


def figure(name):
    def wrap(fn):
        FIGURES.append((name, fn))
        return fn
    return wrap


def val(x):
    return x is not None and isinstance(x, (int, float)) and not math.isnan(x)


def hz_label(row):
    hz = row.get('hz')
    return f"{row['name']}  ({hz:g} Hz)" if val(hz) and hz > 0 else row['name']


def ms(x):
    return f'{x:.0f} ms' if x >= 100 else f'{x:.3g} ms'


def log_axis(axis):
    """Log scale with plain-number major labels and no minor labels (they collide on narrow ranges)."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(NullFormatter())


def rows_axis(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.4, -0.6)


def note(fig, text):
    fig.supxlabel(text, fontsize=7, color=INK2, ha='left', x=0.005)


def barh_values(ax, xs, ys, fmt='{:.3g}', offset=1.05, color=INK2, fontsize=7):
    for x, y in zip(xs, ys):
        if val(x):
            ax.text(x * offset if ax.get_xscale() == 'log' else x + offset, y, fmt.format(x), va='center',
                    ha='left', fontsize=fontsize, color=color)


# ----------------------------------------------------------------------------- ceilings
@figure('fig_ceilings_attain')
def fig_ceilings_attain(d):
    c = d.get('ceilings')
    if not c:
        return None
    precs = [p for p in ('fp16', 'int8', 'fp8', 'fp32') if p in c['precisions'] and val(c['precisions'][p].get('datasheet'))]
    shaped = c.get('shaped_gemm_tflops') or {}
    fp16_ds = (c['precisions'].get('fp16') or {}).get('datasheet')
    ncols = 2 if shaped and val(fp16_ds) else 1
    fig, axes = subplots(1, ncols, 10 if ncols == 2 else 6.5, 3.8, width_ratios=[3, 2] if ncols == 2 else None)
    ax = axes[0] if ncols == 2 else axes
    series = [('torch_burst_same_n', 'torch GEMM burst (n=4096)', ACCENT),
              ('trt_best', 'TensorRT kernel-isolated GEMM (best n)', TEAL),
              ('torch_sustained_median', 'torch sustained (median over the soak)', INK2)]
    width = 0.26
    x = np.arange(len(precs))
    for j, (key, label, color) in enumerate(series):
        vals = [c['precisions'][p].get(key) for p in precs]
        ds = [c['precisions'][p]['datasheet'] for p in precs]
        pcts = [100.0 * v / s if val(v) and val(s) and s else 0.0 for v, s in zip(vals, ds)]
        bars = ax.bar(x + (j - 1) * width, pcts, width, color=color, label=label)
        for bar, v, p in zip(bars, vals, precs):
            if val(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                        f"{v:.4g}\n{c['precisions'][p]['unit']}", ha='center', va='bottom', fontsize=6.5, color=INK)
    ax.set_xticks(x, [f"{p}\n{c['precisions'][p]['datasheet']:g} {c['precisions'][p]['unit']}" for p in precs])
    ax.set_ylim(0, 115)
    ax.axhline(100, color=MUT, lw=0.8, ls='--')
    style_ax(ax, ylabel='% of datasheet dense peak', title='Measured compute ceilings vs the datasheet')
    top_legend(ax, ncol=1)
    if ncols == 2:
        ax2 = axes[1]
        names = sorted(shaped)
        vals = [shaped[n] for n in names]
        ax2.bar(range(len(names)), [100.0 * v / fp16_ds for v in vals], color=ACCENT_GHOST, edgecolor=ACCENT, linewidth=0.8)
        for i, v in enumerate(vals):
            ax2.text(i, 100.0 * v / fp16_ds + 1.2, f'{v:.4g}', ha='center', va='bottom', fontsize=7, color=INK)
        ax2.set_xticks(range(len(names)), ['×'.join(n.split('x')[:2]) + '\n×' + n.split('x')[2] if n.count('x') == 2 else n for n in names], fontsize=7)
        ax2.set_ylim(0, 115)
        ax2.axhline(100, color=MUT, lw=0.8, ls='--')
        style_ax(ax2, ylabel='% of fp16 datasheet', title='Shaped fp16 GEMMs (M×N×K)')
    note(fig, 'Burst = best of 5 window-normalized reps at one size; TensorRT = GEMM-layer profile of a built engine; '
              'the plan is against sustained.')
    return fig


@figure('fig_bandwidth')
def fig_bandwidth(d):
    c = d.get('ceilings')
    if not c or not c.get('bandwidth'):
        return None
    haircut = c.get('haircut_gbps_by_workers') or {}
    ncols = 2 if len(haircut) > 1 else 1
    fig, axes = subplots(1, ncols, 10 if ncols == 2 else 6, 3.6)
    ax = axes[0] if ncols == 2 else axes
    kernels = list(c['bandwidth'])
    vals = [c['bandwidth'][k]['best_gbps'] for k in kernels]
    bars = ax.bar(range(len(kernels)), vals, color=[PRECISION_COLOR['copy'] if k == 'copy_RW' else ACCENT_GHOST for k in kernels],
                  edgecolor=[PRECISION_COLOR['copy'] if k == 'copy_RW' else ACCENT for k in kernels], linewidth=0.8)
    for bar, k in zip(bars, kernels):
        b = c['bandwidth'][k]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{b['best_gbps']:.4g}\n@{b['at_buffer']}", ha='center', va='bottom', fontsize=7, color=INK)
    ds = d['device']['datasheet'].get('bw_gbps')
    if val(ds):
        ax.axhline(ds, color=MUT, lw=0.8, ls='--', label=f'datasheet {ds:g} GB/s')
    if val(c.get('bw_eff_gbps')):
        ax.axhline(c['bw_eff_gbps'], color=PRECISION_COLOR['copy'], lw=0.8, ls=':',
                   label=f"bw_eff {c['bw_eff_gbps']:.4g} GB/s = best copy_RW (budget basis)")
    ax.set_xticks(range(len(kernels)), kernels)
    ax.set_ylim(0, (ds if val(ds) else max(vals)) * 1.18)
    style_ax(ax, ylabel='GB/s', title='DRAM bandwidth, best over buffer sizes (idle GPU)')
    top_legend(ax, ncol=1)
    if ncols == 2:
        ax2 = axes[1]
        workers = sorted(haircut, key=int)
        ax2.plot([int(w) for w in workers], [haircut[w] for w in workers], marker='o', color=ACCENT, lw=1.2, ms=4)
        for w in workers:
            ax2.text(int(w), haircut[w] + 1.5, f'{haircut[w]:.4g}', ha='center', va='bottom', fontsize=7, color=INK)
        ax2.set_ylim(0, (ds if val(ds) else max(vals)) * 1.18)
        style_ax(ax2, xlabel='CPU memory-traffic workers', ylabel='GPU copy GB/s',
                 title='CPU-load haircut (informational; budgets use idle)')
    return fig


@figure('fig_sustained')
def fig_sustained(d):
    c = d.get('ceilings')
    s = (c or {}).get('sustained')
    if not s or not s.get('trace'):
        return None
    tr = s['trace']
    t = [p['t'] for p in tr]
    fig, (ax, ax2) = subplots(2, 1, 9, 5.2, sharex=True, height_ratios=[3, 2])
    ax.plot(t, [p['tflops'] for p in tr], color=ACCENT, lw=1.0, label=f"{s['precision']} GEMM throughput")
    ax.axhline(s['median'], color=INK, lw=0.9, ls='-', label=f"sustained median {s['median']:.4g}")
    if val(s.get('burst_same_n')):
        ax.axhline(s['burst_same_n'], color=MUT, lw=0.9, ls='--',
                   label=f"burst at the same size {s['burst_same_n']:.4g} (ratio {s['sustained_over_burst']:.2f})")
    ax.set_ylim(s['min'] * 0.7, max(s['max'], s.get('burst_same_n') or 0) * 1.12)
    style_ax(ax, ylabel=s['unit'], title=f"Sustained {s['precision']} throughput over a {s['duration_s']:.0f} s soak")
    top_legend(ax, ncol=3)
    mhz = [p['gpu_mhz'] for p in tr]
    if any(val(m) for m in mhz):
        ax2.plot(t, mhz, color=INK2, lw=1.0, label='GPU MHz')
        ax2.set_ylim(0, max(m for m in mhz if val(m)) * 1.2)
    style_ax(ax2, xlabel='s', ylabel='GPU MHz')
    temps = [p['temp_c'] for p in tr]
    if any(val(x) for x in temps):
        ax3 = ax2.twinx()
        ax3.plot(t, temps, color=WARN, lw=0.9, label='max temperature')
        ax3.set_ylabel('°C', color=WARN)
        ax3.grid(False)
        ax3.spines['top'].set_visible(False)
        handles = ax2.get_legend_handles_labels()[0] + ax3.get_legend_handles_labels()[0]
        ax2.legend(handles=handles, loc='lower left', ncol=2)
    note(fig, 'A throughput dip while the clock still reads maximum is a power/thermal clamp; sustained is the planning ceiling.')
    return fig


# ----------------------------------------------------------------------------- solo
def solo_rows(d):
    s = d.get('solo')
    return (s or {}).get('rows') or []


@figure('fig_solo_latency')
def fig_solo_latency(d):
    rows = [r for r in solo_rows(d) if val(r.get('p99_ms'))]
    if not rows:
        return None
    n = len(rows)
    fig, ax = subplots(1, 1, 9.5, row_height(n))
    ax.set_xscale('log')
    log_axis(ax.xaxis)
    for i, r in enumerate(rows):
        kind = r['deadline_kind']
        if kind == 'spec':
            color = ACCENT if r.get('meets_deadline') else WARN
        else:
            color = MUT
        ax.barh(i, r['p99_ms'], height=0.62, color=color, alpha=0.5 if r.get('valid') is False else 1.0)
        if val(r.get('mean_ms')):
            ax.plot(r['mean_ms'], i, marker='|', color='white', ms=8, mew=1.3)
        if kind != 'none' and val(r.get('deadline_ms')):
            ax.plot(r['deadline_ms'], i, marker='v', ms=5, color=INK, mfc=INK if kind == 'spec' else 'white', mew=0.9)
        tag = ms(r['p99_ms'])
        if r.get('valid') is False:
            tag += '  (excluded: clock integrity)'
        anchor = max(r['p99_ms'], r.get('deadline_ms') or 0 if kind != 'none' else 0)
        ax.text(anchor * 1.1, i, tag, va='center', ha='left', fontsize=7, color=INK2)
    rows_axis(ax, [hz_label(r) for r in rows])
    ax.set_xlim(right=max(r['p99_ms'] for r in rows) * 4)
    style_ax(ax, xlabel='ms (log)', title='Solo latency of record per row: p99 bar, mean tick, deadline marker')
    handles = [Patch(color=ACCENT, label='p99 within the deadline'), Patch(color=WARN, label='p99 misses the deadline'),
               Patch(color=MUT, label='deadline is a placeholder (rate open) or no rate'),
               Line2D([], [], marker='v', color=INK, ls='', label='deadline'),
               Line2D([], [], marker='|', color=INK2, ls='', label='mean')]
    top_legend(ax, ncol=3, handles=handles)
    note(fig, 'Rows without a rate carry no deadline marker: their latency is the verdict (modal use).')
    return fig


@figure('fig_solo_N')
def fig_solo_N(d):
    rows = [r for r in solo_rows(d) if val(r.get('N'))]
    if not rows:
        return None
    n = len(rows)
    headroom = d.get('headroom') or 0.65
    fig, ax = subplots(1, 1, 9.5, row_height(n))
    ax.set_xscale('log')
    log_axis(ax.xaxis)
    xmax = 1.0
    for i, r in enumerate(rows):
        L, C, N = r.get('L'), r.get('C'), r['N']
        pts = [x for x in (L, C) if val(x)]
        if len(pts) == 2:
            ax.plot(sorted(pts), [i, i], color=TRACK, lw=3, zorder=1)
        if val(L):
            ax.plot(L, i, marker='>', ms=8, color=ACCENT, mfc='white', mew=1.0, ls='', zorder=2)
        if val(C):
            ax.plot(C, i, marker='s', ms=7.5, color=INK2, mfc='white', mew=1.0, ls='', zorder=2)
        color = ACCENT if r.get('cause') == 'latency-limited' else TEAL
        ax.plot(N, i, marker='o', ms=4.5, color=color, ls='', zorder=3)
        label = f'N {N:.3g}' + ('' if val(C) else '  (C unbounded: no budget measured)')
        if r['deadline_kind'] == 'placeholder':
            label += '  *'
        ax.text(max(pts + [N]) * 1.3, i, label, va='center', ha='left', fontsize=7, color=INK2)
        xmax = max(xmax, max(pts + [N]))
    ax.axvline(1.0, color=WARN, lw=0.9, ls='--', label='N = 1: one device')
    ax.axvline(1.0 / headroom, color=MUT, lw=0.9, ls=':', label=f'C at {headroom:.0%} headroom (1/{headroom:g})')
    rows_axis(ax, [hz_label(r) for r in rows])
    ax.set_xlim(right=xmax * 4)
    style_ax(ax, xlabel='units of this device (log)', title='Solo N = min(L, C) per row')
    handles = [Line2D([], [], marker='>', color=ACCENT, mfc='white', ls='', label='L = deadline / p99'),
               Line2D([], [], marker='s', color=INK2, mfc='white', ls='', label='C = 1 / U_max'),
               Line2D([], [], marker='o', color=ACCENT, ls='', label='N, latency-limited'),
               Line2D([], [], marker='o', color=TEAL, ls='', label='N, throughput-limited')]
    handles += ax.get_legend_handles_labels()[0]
    top_legend(ax, ncol=3, handles=handles)
    note(fig, '* rate not fixed by the mix: N at the placeholder rate/deadline; see the rate sweep.')
    return fig


@figure('fig_solo_budgets')
def fig_solo_budgets(d):
    rows = [r for r in solo_rows(d) if r.get('budgets')]
    if not rows:
        return None
    n = len(rows)
    headroom = d.get('headroom') or 0.65
    fig, axes = subplots(1, 3, 11, row_height(n), sharey=True)
    xmax = 1.6
    titles = {'time': 'time: p99 × Hz', 'bw': 'bandwidth: bytes × Hz / bw_eff', 'vram': 'VRAM footprint'}
    for ax, key in zip(axes, ('time', 'bw', 'vram')):
        for i, r in enumerate(rows):
            share = r['budgets'].get(key)
            shares = {k: v for k, v in r['budgets'].items() if val(v)}
            binding = shares and key == max(shares, key=shares.get)
            ax.add_patch(Rectangle((0, i - 0.31), xmax, 0.62, color=TRACK, lw=0))
            if not val(share):
                text = 'not measured' if key != 'time' else 'not scored'
                ax.text(0.02, i, text, va='center', ha='left', fontsize=6.5, color=MUT)
                continue
            if key == 'time' and (not val(r.get('hz')) or r['hz'] == 0):
                ax.text(0.02, i, 'no rate (modal: no sustained bill)', va='center', ha='left', fontsize=6.5, color=MUT)
                continue
            color = WARN if share > 1.0 else (ACCENT if binding else ACCENT_GHOST)
            ax.barh(i, min(share, xmax), height=0.62, color=color)
            label = f'{share:.3g}' + ('  ▶' if share > xmax else '')
            ax.text(min(share, xmax) + 0.02, i, label, va='center', ha='left', fontsize=7,
                    color=INK if binding else INK2, fontweight='bold' if binding else 'normal')
        ax.axvline(headroom, color=MUT, lw=0.8, ls=':')
        ax.axvline(1.0, color=WARN, lw=0.8, ls='--')
        ax.set_xlim(0, xmax + 0.35)
        ax.set_xticks([0, 0.5, 1.0, 1.5], ['0', '0.5', '1', '1.5'])
        style_ax(ax, xlabel='share of one device', title=titles[key])
    rows_axis(axes[0], [hz_label(r) for r in rows])
    fig.suptitle('Solo budgets per row: the binding gauge (bold) sets U_max and C = 1 / U_max', x=0.01, ha='left', fontsize=10)
    note(fig, f'Dotted = {headroom:.0%} headroom line; dashed = one device full. A share beyond the axis is marked ▶.')
    return fig


@figure('fig_bound_mix')
def fig_bound_mix(d):
    rows = [r for r in solo_rows(d) if (r.get('roofline') or {}).get('time_pct_by_bound')]
    if not rows:
        return None
    n = len(rows)
    fig, ax = subplots(1, 1, 9, row_height(n, per_row=0.3))
    keys = ['memory', 'compute', 'latency', 'unclassified']
    left = np.zeros(n)
    for key in keys:
        vals = np.array([float(r['roofline']['time_pct_by_bound'].get(key) or 0.0) for r in rows])
        ax.barh(range(n), vals, left=left, height=0.62, color=BOUND_COLOR[key], label=f'{key}-bound')
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 8:
                ax.text(l + v / 2, i, f'{v:.0f}%', va='center', ha='center', fontsize=7,
                        color='white' if key in ('memory', 'compute') else INK)
        left = left + vals
    rows_axis(ax, [r['name'] for r in rows])
    ax.set_xlim(0, 100)
    style_ax(ax, xlabel='% of kernel time', title='What bounds the kernel time of each engine row (against the measured ceilings)')
    top_legend(ax, ncol=4)
    sources = sorted({r['roofline'].get('byte_source') for r in rows if r['roofline'].get('byte_source')})
    note(fig, 'Byte source: ' + ', '.join(sources) + '. Latency-bound = far from both roofs (launch/serialization dominated).')
    return fig


@figure('fig_roofline')
def fig_roofline(d):
    s = d.get('solo') or {}
    kernels = s.get('kernels') or {}
    if not kernels:
        return None
    names = sorted(kernels)
    ceil = s.get('ceilings_used') or {}
    bw = ceil.get('dram_gbps_idle') or s.get('bw_eff_gbps')
    pipes_ceiling = {'int8': ceil.get('tensor_ceiling_int8_tops'), 'fp16': ceil.get('tensor_ceiling_fp16_tflops'),
                     'fp8': ceil.get('tensor_ceiling_fp8_tflops'), 'cuda': ceil.get('cudacore_fp32_tflops')}
    ncols = min(3, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = subplots(nrows, ncols, 3.6 * ncols, 2.9 * nrows + 0.5, squeeze=False)
    rows = {r['name']: r for r in solo_rows(d)}
    for k, name in enumerate(names):
        ax = axes[k // ncols][k % ncols]
        ks = [x for x in kernels[name] if val(x.get('intensity')) and val(x.get('achieved')) and x['intensity'] > 0 and x['achieved'] > 0]
        us_max = max((x['us'] for x in ks if val(x.get('us'))), default=1.0) or 1.0
        for x in ks:
            size = 6 + 70 * (x.get('us') or 0) / us_max
            ax.scatter(x['intensity'], x['achieved'], s=size, color=color_for(x.get('pipe'), PIPE_COLOR), alpha=0.65,
                       edgecolor='white', linewidth=0.4, zorder=3)
        xs = np.logspace(-1, 4, 60)
        if val(bw):
            ax.plot(xs, xs * bw / 1e3, color=INK2, lw=0.9, ls='--', zorder=2)
        for pipe, c in pipes_ceiling.items():
            if val(c) and any(x.get('pipe') == pipe for x in ks):
                ax.axhline(c, color=PIPE_COLOR.get(pipe, MUT), lw=0.8, ls=':', zorder=2)
        ax.set_xscale('log')
        log_axis(ax.xaxis)
        ax.set_yscale('log')
        ax.set_xlim(0.1, 1e4)
        ax.set_ylim(1e-3, 1e3)
        bounds = ((rows.get(name) or {}).get('roofline') or {}).get('time_pct_by_bound') or {}
        title = name + (f"  mem {bounds.get('memory', 0):.0f}% / comp {bounds.get('compute', 0):.0f}% / lat {bounds.get('latency', 0):.0f}%"
                        if bounds else '')
        style_ax(ax, title=title)
        ax.set_title(title, fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel('achieved TOPS')
        if k + ncols >= len(names):
            ax.set_xlabel('ops / DRAM byte')
    for k in range(len(names), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    handles = [Line2D([], [], marker='o', ls='', color=c, label=f'{p} pipe') for p, c in PIPE_COLOR.items()
               if any(x.get('pipe') == p for ks in kernels.values() for x in ks)]
    handles += [Line2D([], [], color=INK2, ls='--', label='DRAM roof (bw_eff)'),
                Line2D([], [], color=MUT, ls=':', label='pipe ceiling')]
    fig.legend(handles=handles, loc='outside lower center', ncol=len(handles))
    fig.suptitle('Kernel roofline per engine row: marker area ∝ kernel time', x=0.01, ha='left', fontsize=10)
    return fig


@figure('fig_floors')
def fig_floors(d):
    rows = [r for r in solo_rows(d) if r.get('floor') and val(r.get('p99_ms'))]
    if not rows:
        return None
    n = len(rows)
    fig, (ax, ax2) = subplots(1, 2, 11, row_height(n), sharey=True)
    ax.set_xscale('log')
    log_axis(ax.xaxis)
    ax2.set_xscale('log')
    log_axis(ax2.xaxis)
    for i, r in enumerate(rows):
        f = r['floor']
        t, p99 = f.get('t_floor_ms'), r['p99_ms']
        memory_only = 'memory-only' in (f.get('note') or '')
        if not val(t):
            ax.text(p99 * 1.15, i, 'no floor: ' + (f.get('note') or '').split(':')[-1].strip(), va='center', fontsize=6.5, color=MUT)
            ax.plot(p99, i, marker='o', ms=5, color=INK, ls='')
            continue
        ax.plot([t, p99], [i, i], color=MUT if memory_only else ACCENT_GHOST, lw=2.2, ls=':' if memory_only else '-', zorder=1)
        ax.plot(t, i, marker='s' if f.get('bound_by') == 'compute' else 'o', ms=5.5, color=ACCENT if not memory_only else MUT,
                mfc='white', mew=1.0, ls='', zorder=2)
        ax.plot(p99, i, marker='o', ms=5, color=INK, ls='', zorder=3)
        ax.text(p99 * 1.15, i, f'×{p99 / t:.3g}', va='center', fontsize=7, color=INK2)
        N, Nc = r.get('N'), f.get('N_ceiling')
        if val(N) and val(Nc):
            ax2.plot([N, Nc], [i, i], color=MUT if memory_only else ACCENT_GHOST, lw=2.2, ls=':' if memory_only else '-', zorder=1)
            ax2.plot(N, i, marker='o', ms=5, color=INK, ls='', zorder=3)
            ax2.plot(Nc, i, marker='o', ms=5.5, color=ACCENT if not memory_only else MUT, mfc='white', mew=1.0, ls='', zorder=2)
            ax2.text(max(N, Nc) * 1.2, i, f'{N:.3g} → {Nc:.3g}', va='center', fontsize=7, color=INK2)
    rows_axis(ax, [hz_label(r) for r in rows])
    ax2.axvline(1.0, color=WARN, lw=0.9, ls='--')
    ax.set_xlim(right=max(r['p99_ms'] for r in rows) * 30)
    ax2.set_xlim(right=max(max(r.get('N') or 0, r['floor'].get('N_ceiling') or 0) for r in rows) * 6)
    style_ax(ax, xlabel='ms (log)', title='Latency: floor (hollow) → measured p99 (filled)')
    style_ax(ax2, xlabel='units of this device (log)', title='N: measured (filled) → at the floor (hollow)')
    handles = [Line2D([], [], marker='o', color=ACCENT, mfc='white', ls='', label='memory-bound floor  bytes / bw_eff'),
               Line2D([], [], marker='s', color=ACCENT, mfc='white', ls='', label='compute-bound floor  arch GFLOPs / ceiling'),
               Line2D([], [], marker='o', color=MUT, mfc='white', ls=':', label='memory-only floor (no compute term measured)'),
               Line2D([], [], marker='o', color=INK, ls='', label='measured')]
    top_legend(ax, ncol=2, handles=handles)
    note(fig, 'The floor is a roofline bound: an engine cannot beat max(bytes/bw_eff, arch/ceiling); N at the floor is an upper bound, not a promise.')
    return fig


@figure('fig_rate_sweep')
def fig_rate_sweep(d):
    rows = [r for r in solo_rows(d) if r.get('N_vs_hz')]
    if not rows:
        return None
    fig, ax = subplots(1, 1, 9.5, 5.4)
    for i, r in enumerate(rows):
        pts = [(hz, N) for hz, N in r['N_vs_hz'] if val(hz) and val(N) and hz > 0 and N > 0]
        if not pts:
            continue
        color = SERIES[i % len(SERIES)]
        ls = LINESTYLES[(i // len(SERIES)) % len(LINESTYLES)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', ms=3.5, lw=1.1, color=color, ls=ls, label=r['name'])
        if val(r.get('max_hz_at_N1')) and r['max_hz_at_N1'] > 0:
            ax.plot(r['max_hz_at_N1'], 1.0, marker='|', ms=9, mew=1.4, color=color, ls='')
    ax.axhline(1.0, color=WARN, lw=0.9, ls='--')
    ax.set_xscale('log')
    log_axis(ax.xaxis)
    ax.set_yscale('log')
    style_ax(ax, xlabel='request rate (Hz, log)', ylabel='N with the period as the deadline (log)',
             title='Rate sweep for rows whose rate is not fixed by the mix; tick = highest rate at N ≥ 1')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=7)
    return fig


# ----------------------------------------------------------------------------- co-location
def coloc(d):
    return d.get('coloc') or None


@figure('fig_coloc_matrix')
def fig_coloc_matrix(d):
    c = coloc(d)
    if not c or not c.get('cells'):
        return None
    mixes, arms = c['mixes'], c['arms']
    cells = {(x['mix'], x['arm']): x for x in c['cells']}
    fig, ax = subplots(1, 1, 1.7 * len(arms) + 3.2, 0.5 * len(mixes) + 1.6)
    vals = [x['N_measured'] for x in c['cells'] if val(x.get('N_measured'))]
    vmax = max(vals + [1.0])
    cmap = plt.get_cmap('Blues')
    for i, mix in enumerate(mixes):
        for j, arm in enumerate(arms):
            x = cells.get((mix, arm))
            if not x or not x.get('valid'):
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=TRACK, edgecolor='white', hatch='///', lw=1.5))
                ax.text(j + 0.5, i + 0.5, (x or {}).get('status') or 'missing', ha='center', va='center', fontsize=6.5, color=INK2, wrap=True)
                continue
            N = x.get('N_measured')
            shade = cmap(0.15 + 0.7 * min(N, vmax) / vmax) if val(N) else TRACK
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=shade, edgecolor='white', lw=1.5))
            if x.get('fits') is False:
                ax.add_patch(Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, facecolor='none', edgecolor=WARN, lw=1.6))
            dark = val(N) and N / vmax > 0.55
            ax.text(j + 0.5, i + 0.42, f'{N:.2f}' if val(N) else '-', ha='center', va='center', fontsize=9,
                    color='white' if dark else INK, fontweight='bold')
            ax.text(j + 0.5, i + 0.78, 'fits' if x.get('fits') else 'does not fit', ha='center', va='center', fontsize=6.5,
                    color='white' if dark else (OK if x.get('fits') else WARN))
    ax.set_xlim(0, len(arms))
    ax.set_ylim(len(mixes), 0)
    ax.set_xticks([j + 0.5 for j in range(len(arms))], arms)
    ax.set_yticks([i + 0.5 for i in range(len(mixes))], mixes)
    ax.xaxis.tick_top()
    ax.grid(False)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.set_title('Co-location N_measured per mix × arm (red outline = a row misses its deadline rule; hatched = no valid measurement)',
                 fontsize=9, pad=24)
    return fig


@figure('fig_coloc_contention')
def fig_coloc_contention(d):
    c = coloc(d)
    if not c or not c.get('cells'):
        return None
    mixes, arms = c['mixes'], c['arms']
    by_mix = {}
    for x in c['cells']:
        by_mix.setdefault(x['mix'], {})[x['arm']] = x
    ncols = 2 if len(mixes) > 1 else 1
    nrows = math.ceil(len(mixes) / ncols)
    fig, axes = subplots(nrows, ncols, 5.4 * ncols, 2.3 * nrows + 0.8, squeeze=False)
    for k, mix in enumerate(mixes):
        ax = axes[k // ncols][k % ncols]
        ax.set_xscale('log')
        log_axis(ax.xaxis)
        cells = by_mix.get(mix, {})
        names = sorted({rn for x in cells.values() for rn in (x.get('rows') or {})})
        valid_arms = [a for a in arms if cells.get(a, {}).get('valid')]
        na = max(len(valid_arms), 1)
        xmax = 1.0
        for i, rn in enumerate(names):
            first = next((cells[a]['rows'][rn] for a in valid_arms if rn in cells[a].get('rows', {})), None)
            if first:
                if val(first.get('paced_solo_p99_ms')):
                    ax.plot(first['paced_solo_p99_ms'], i, marker='o', ms=5, color=INK2, mfc='white', mew=1.0, ls='', zorder=2)
                if val(first.get('deadline_ms')):
                    ax.plot(first['deadline_ms'], i, marker='v', ms=5, color=INK, ls='', zorder=2)
                    xmax = max(xmax, first['deadline_ms'])
            for j, a in enumerate(valid_arms):
                r = cells[a].get('rows', {}).get(rn)
                if not r or not val(r.get('p99_ms')):
                    continue
                y = i + (j - (na - 1) / 2) * 0.22
                miss = val(r.get('miss_frac')) and r['miss_frac'] > (c.get('rules') or {}).get('miss_frac_max', 0.01)
                ax.plot(r['p99_ms'], y, marker='o', ms=5, color=color_for(a, ARM_COLOR, j), ls='',
                        mec=WARN if miss else color_for(a, ARM_COLOR, j), mew=1.4 if miss else 0.8, zorder=3)
                if val(r.get('p99_contended_pred')):
                    ax.plot(r['p99_contended_pred'], y, marker='x', ms=4.5, color=color_for(a, ARM_COLOR, j), ls='', zorder=2)
                xmax = max(xmax, r['p99_ms'])
        ax.set_yticks(range(len(names)), names, fontsize=7.5)
        ax.set_ylim(len(names) - 0.4, -0.6)
        ax.set_xlim(right=xmax * 2.2)
        status = ', '.join(f"{a}: N {cells[a]['N_measured']:.2f}" + ('' if cells[a].get('fits') else ' (unfit)')
                           for a in valid_arms if val(cells[a].get('N_measured')))
        style_ax(ax, xlabel='p99 ms (log)' if k + ncols >= len(mixes) else None)
        ax.set_title(f'{mix}   {status}', fontsize=8)
    for k in range(len(mixes), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    used = {x['arm'] for x in c['cells'] if x.get('valid')}
    handles = [Line2D([], [], marker='o', color=color_for(a, ARM_COLOR, j), ls='', label=f'{a}: contended p99')
               for j, a in enumerate(arms) if a in used]
    handles += [Line2D([], [], marker='o', color=INK2, mfc='white', ls='', label='paced solo p99'),
                Line2D([], [], marker='x', color=INK2, ls='', label='predicted contended p99'),
                Line2D([], [], marker='v', color=INK, ls='', label='deadline'),
                Line2D([], [], marker='o', color='white', mec=WARN, mew=1.4, ls='', label='misses > rule')]
    fig.legend(handles=handles, loc='outside lower center', ncol=min(len(handles), 5))
    fig.suptitle('Contention per frame row: paced solo → contended p99 per arm', x=0.01, ha='left', fontsize=10)
    return fig


@figure('fig_coloc_bounds')
def fig_coloc_bounds(d):
    c = coloc(d)
    cells = [x for x in (c or {}).get('cells') or [] if x.get('valid') and val(x.get('N_measured'))]
    if not cells:
        return None
    n = len(cells)
    labels = [f"{x['mix']} / {x['arm']}" for x in cells]
    fig, (ax, ax2) = subplots(1, 2, 11, row_height(n), sharey=True, width_ratios=[3, 2])
    ax.set_xscale('log')
    log_axis(ax.xaxis)
    xmax = 1.0
    for i, x in enumerate(cells):
        L, C, N = x.get('L_contended'), x.get('C_composed'), x['N_measured']
        pts = [v for v in (L, C) if val(v)]
        if len(pts) == 2:
            ax.plot(sorted(pts), [i, i], color=TRACK, lw=3, zorder=1)
        if val(L):
            ax.plot(L, i, marker='>', ms=6, color=ACCENT, mfc='white', mew=1.0, ls='', zorder=2)
        if val(C):
            ax.plot(C, i, marker='s', ms=5.5, color=INK2, mfc='white', mew=1.0, ls='', zorder=2)
        ax.plot(N, i, marker='o', ms=6, color=OK if x.get('fits') else WARN, ls='', zorder=3)
        ax.text(max(pts + [N]) * 1.25, i, f'N {N:.2f}' + (f"  worst: {x['worst_row']}" if x.get('worst_row') else ''),
                va='center', ha='left', fontsize=7, color=INK2)
        xmax = max(xmax, max(pts + [N]))
        m = x.get('makespan_p99_over_period')
        if val(m):
            ax2.barh(i, m, height=0.62, color=OK if m <= 1.0 else WARN)
            ax2.text(m + 0.02, i, f'{m:.2f}', va='center', fontsize=7, color=INK2)
    ax.axvline(1.0, color=WARN, lw=0.9, ls='--')
    ax2.axvline(1.0, color=WARN, lw=0.9, ls='--')
    rows_axis(ax, labels)
    ax.set_xlim(right=xmax * 6)
    ax2.set_xlim(0, max(1.15, max((x.get('makespan_p99_over_period') or 0) for x in cells) * 1.2))
    style_ax(ax, xlabel='units of this device (log)', title='N_measured = min(L contended, C composed)')
    style_ax(ax2, xlabel='makespan p99 / period', title='Period fill: all rows done within the period')
    handles = [Line2D([], [], marker='>', color=ACCENT, mfc='white', ls='', label='L contended = deadline / contended p99 (worst row)'),
               Line2D([], [], marker='s', color=INK2, mfc='white', ls='', label='C composed = 1 / summed budgets'),
               Line2D([], [], marker='o', color=OK, ls='', label='N, all rows fit'),
               Line2D([], [], marker='o', color=WARN, ls='', label='N, a row misses its rule')]
    handles.append(Line2D([], [], color=WARN, lw=0.9, ls='--', label='1.0: one device / one period'))
    top_legend(ax, ncol=2, handles=handles, fontsize=7)
    ax2.set_title(ax2.get_title(loc='left'), loc='left', pad=8 + 13 * 3)
    return fig


# ----------------------------------------------------------------------------- power
def power(d):
    return d.get('power') or None


def grouped_barh(ax, labels, series, colors, fmt='{:.3g}', hatch=None):
    """series: list of (name, values aligned with labels); hatch: same shape, True where a bar is hatched."""
    n, k = len(labels), len(series)
    height = 0.8 / max(k, 1)
    for j, (name, values) in enumerate(series):
        ys = [i + (j - (k - 1) / 2) * height for i in range(n)]
        vals = [v if val(v) else 0.0 for v in values]
        bars = ax.barh(ys, vals, height=height * 0.92, color=colors[j % len(colors)], label=name)
        for i, (bar, v) in enumerate(zip(bars, values)):
            if hatch and hatch[j][i]:
                bar.set_hatch('////')
                bar.set_edgecolor('white')
            if val(v):
                ax.text(v, bar.get_y() + bar.get_height() / 2, ' ' + fmt.format(v), va='center', ha='left', fontsize=6.5, color=INK2)
            else:
                ax.text(0, bar.get_y() + bar.get_height() / 2, ' n/a', va='center', ha='left', fontsize=6.5, color=MUT)
    rows_axis(ax, labels)


@figure('fig_power_solo')
def fig_power_solo(d):
    p = power(d)
    if not p:
        return None
    points = p['points_order']
    keys = sorted({k for pt in points for k in (p['points'][pt].get('paced_solo') or {})})
    if not keys:
        return None
    fig, axes = subplots(1, 3, 11.5, row_height(len(keys), per_row=0.22 * max(len(points), 1) + 0.12), sharey=True)
    colors = SERIES
    for ax, (metric, title, fmt) in zip(axes, (('p99_ms', 'paced-solo p99 (ms)', '{:.3g}'),
                                               ('mean_w', 'module power (W)', '{:.3g}'),
                                               ('j_per_frame_marginal', 'energy above idle per frame (J, log)', '{:.3g}'))):
        series = [(pt, [(p['points'][pt].get('paced_solo') or {}).get(k, {}).get(metric) for k in keys]) for pt in points]
        hatch = [[bool(((p['points'][pt].get('paced_solo') or {}).get(k) or {}).get('miss_frac') or 0) for k in keys] for pt in points]
        grouped_barh(ax, keys, series, colors, fmt, hatch)
        values = [v for _, vs in series for v in vs if val(v) and v > 0]
        if metric.startswith('j_per_frame'):
            ax.set_xscale('log')
            log_axis(ax.xaxis)
            ax.set_xlim(min(values, default=0.1) / 2, max(values, default=1.0) * 4)
        else:
            ax.set_xlim(0, max(values, default=1.0) * 1.3)
        style_ax(ax, title=title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=len(points), title='power operating point', fontsize=7, title_fontsize=7)
    fig.suptitle('Paced solo rows across power operating points (hatched = deadline misses)', x=0.01, ha='left', fontsize=10)
    return fig


@figure('fig_power_cells')
def fig_power_cells(d):
    p = power(d)
    if not p:
        return None
    points = p['points_order']
    keys = sorted({k for pt in points for k, v in (p['points'][pt].get('cells') or {}).items() if v.get('valid')})
    if not keys:
        return None
    fig, axes = subplots(1, 2, 10, row_height(len(keys), per_row=0.22 * max(len(points), 1) + 0.12), sharey=True)
    for ax, (metric, title) in zip(axes, (('N_measured', 'co-location N_measured'), ('j_per_period', 'energy per period (J)'))):
        series = [(pt, [((p['points'][pt].get('cells') or {}).get(k) or {}).get(metric) for k in keys]) for pt in points]
        hatch = [[((p['points'][pt].get('cells') or {}).get(k) or {}).get('fits') is False for k in keys] for pt in points]
        grouped_barh(ax, keys, series, SERIES, '{:.3g}', hatch)
        ax.set_xlim(0, max((v for _, vs in series for v in vs if val(v)), default=1.0) * 1.3)
        style_ax(ax, title=title)
    axes[0].axvline(1.0, color=WARN, lw=0.9, ls='--')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=len(points), title='power operating point', fontsize=7, title_fontsize=7)
    fig.suptitle('Co-location cells across power operating points (hatched = a row misses its rule)', x=0.01, ha='left', fontsize=10)
    return fig


@figure('fig_power_per_watt')
def fig_power_per_watt(d):
    p = power(d)
    if not p or not any(p.get('curves', {}).values()):
        return None
    points = [pt for pt in p['points_order'] if p['curves'].get(pt)]
    fig, (ax, ax2, ax3) = subplots(1, 3, 12.5, 3.9)
    idle = None
    for j, pt in enumerate(points):
        ls = LINESTYLES[j % len(LINESTYLES)]
        curves = p['curves'][pt]['precisions']
        idle = idle if idle is not None else p['curves'][pt].get('idle_module_w')
        for i, prec in enumerate(sorted(curves)):
            unit = curves[prec].get('unit') or ''
            pts = [q for q in curves[prec]['points'] if val(q.get('module_w'))]
            color = color_for(prec, PRECISION_COLOR, i)
            label = f'{prec} @ {pt}'
            if unit.upper() in ('TOPS', 'TFLOPS'):
                ax.plot([q['delivered'] for q in pts], [q['module_w'] for q in pts], marker='o', ms=3, lw=1.0, ls=ls, color=color, label=label)
                fit = ((p['points'][pt].get('per_watt') or {}).get('precisions') or {}).get(prec, {}).get('fit') or {}
                if val(fit.get('slope')) and val(fit.get('intercept')) and pts:
                    xs = np.array([0, max(q['delivered'] for q in pts)])
                    ax.plot(xs, fit['intercept'] + fit['slope'] * xs, lw=0.7, ls=ls, color=color, alpha=0.5)
                ax2.plot([q['busy_pct'] for q in pts], [q['per_w'] for q in pts], marker='o', ms=3, lw=1.0, ls=ls, color=color, label=label)
            ax3.plot([q['busy_pct'] for q in pts], [q['gpu_mhz'] for q in pts], marker='o', ms=3, lw=1.0, ls=ls, color=color, label=label)
    if val(idle):
        ax.axhline(idle, color=MUT, lw=0.8, ls=':')
        ax.text(0.99, idle, f'idle {idle:.3g} W', fontsize=7, color=INK2, va='bottom', ha='right', transform=ax.get_yaxis_transform())
    style_ax(ax, xlabel='delivered TOPS / TFLOPS', ylabel='module W', title='Power vs delivered throughput (thin = linear fit)')
    style_ax(ax2, xlabel='busy %', ylabel='TOPS per module W', title='Efficiency vs duty')
    style_ax(ax3, xlabel='busy %', ylabel='GPU MHz', title='Clock vs duty (copy shown too)')
    ax.legend(fontsize=6.5, loc='upper left')
    ax3.legend(fontsize=6.5, loc='lower right')
    return fig


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', help='comma-separated figure names to render')
    args = ap.parse_args()
    d = json.load(open(args.data))
    os.makedirs(args.out, exist_ok=True)
    only = set(args.only.split(',')) if args.only else None
    apply_rc()
    written, skipped = [], []
    for name, fn in FIGURES:
        if only and name not in only:
            continue
        fig = fn(d)
        if fig is None:
            skipped.append(name)
            continue
        save(fig, os.path.join(args.out, name + '.png'))
        written.append(name)
    for name in written:
        print(f'  wrote   {name}.png')
    for name in skipped:
        print(f'  skipped {name} (stage not present)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Shared look of the stage-7 figures: palette, rc settings and the save routine.

Everything that could make two renders of the same data.json differ is pinned
here: backend, fonts, hinting, layout engine, DPI and the PNG metadata. Import
this module before pyplot anywhere in the stage.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK = '#0A2540'
INK2 = '#5A6B7B'
ACCENT = '#0067B9'
ACCENT_GHOST = '#CFE0F0'
WARN = '#C0392B'
MUT = '#9AA5B1'
OK = '#1E8E3E'
TEAL = '#00A3B4'
TRACK = '#EDF1F6'

PRECISION_COLOR = {'fp16': '#0067B9', 'int8': '#E8772E', 'fp8': '#2E8B57', 'fp32': '#7B2D8E',
                   'copy': '#7B2D8E', 'nvfp4': '#B03A8C'}
ARM_COLOR = {'plain': '#0067B9', 'mps': '#E8772E', 'streams': '#2E8B57', 'mig': '#7B2D8E'}
BOUND_COLOR = {'memory': '#0067B9', 'compute': '#E8772E', 'latency': '#9AA5B1', 'unclassified': '#D9DEE5'}
PIPE_COLOR = {'int8': '#E8772E', 'fp16': '#0067B9', 'fp8': '#2E8B57', 'cuda': '#7B2D8E', 'tensor_unknown': '#9AA5B1'}
SERIES = ['#0067B9', '#E8772E', '#2E8B57', '#7B2D8E', '#C0392B', '#00A3B4', '#8C6D1F', '#5A6B7B']
LINESTYLES = ['-', '--', ':', '-.']

DPI = 200
RC = {
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'mathtext.fontset': 'dejavusans',
    'text.hinting': 'auto',
    'text.hinting_factor': 8,
    'text.antialiased': True,
    'path.simplify': False,
    'agg.path.chunksize': 0,
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#C6D2E0',
    'axes.linewidth': 0.9,
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.titlesize': 10,
    'axes.titlelocation': 'left',
    'axes.labelsize': 9,
    'axes.labelcolor': '#31445A',
    'axes.formatter.useoffset': False,
    'axes.formatter.use_locale': False,
    'axes.unicode_minus': False,
    'grid.color': '#E8EDF3',
    'grid.linewidth': 0.7,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'legend.frameon': False,
}


def apply_rc():
    matplotlib.rcdefaults()
    plt.rcParams.update(RC)


def color_for(name, table, index=0):
    return table.get(name) or SERIES[index % len(SERIES)]


def style_ax(ax, xlabel=None, ylabel=None, title=None):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def top_legend(ax, ncol, **kw):
    """Legend above the axes, below the title: the title pad grows with the legend's row count."""
    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=ncol, handlelength=1.6,
                       columnspacing=1.2, borderaxespad=0.2, **kw)
    rows = -(-len(legend.get_texts()) // max(ncol, 1))
    for loc in ('left', 'center', 'right'):
        title = ax.get_title(loc=loc)
        if title:
            ax.set_title(title, loc=loc, pad=8 + 13 * rows)
    return legend


def figure(width, height, **kw):
    return plt.figure(figsize=(width, height), constrained_layout=True, **kw)


def subplots(nrows, ncols, width, height, **kw):
    return plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True, **kw)


def save(fig, path):
    fig.savefig(path, format='png', dpi=DPI, facecolor='white', metadata={'Software': None})
    plt.close(fig)
    return path


def row_height(n, per_row=0.26, base=1.4, minimum=2.6):
    return max(minimum, base + per_row * n)

"""Markdown table helpers shared by the stage-7 report."""


def fmt(value, pattern='%.2f', na='-'):
    """Format a number, '-' for None / NaN / non-numeric."""
    if value is None or isinstance(value, bool):
        return na if value is None else str(value)
    if isinstance(value, (int, float)):
        if value != value:
            return na
        return pattern % value
    return str(value)


def pct(value, decimals=0, na='-'):
    if value is None or not isinstance(value, (int, float)) or value != value:
        return na
    return f'{value * 100:.{decimals}f}%'


def table(header, rows, align=None):
    """Rows of already-formatted cells -> markdown table lines. align: 'l'/'r' per column."""
    align = align or ['l'] + ['r'] * (len(header) - 1)
    sep = ['---:' if a == 'r' else ':---' for a in align]
    lines = ['| ' + ' | '.join(str(h) for h in header) + ' |',
             '| ' + ' | '.join(sep) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return lines


def flag(value, yes='yes', no='no', na='-'):
    if value is None:
        return na
    return yes if value else no


def num(value, na='-'):
    """Plain number: no exponent, 3-4 significant digits, integers past 1000."""
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool) or value != value:
        return na
    if abs(value) >= 1000:
        return f'{value:,.0f}'.replace(',', ' ')
    return f'{value:.3g}' if abs(value) < 100 else f'{value:.0f}'

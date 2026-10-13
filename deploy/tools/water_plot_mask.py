"""Plot-only hypothesis: concurrent brief departures from normal indicate disturbance.

Marks recovered events, not vehicle identity; no detector votes are changed.
"""
import html

MAX_SECONDS = 5.
ONSET_SECONDS = 1.
MIN_BLOCKS = 2


def find_windows(history, blocks):
    events = []
    for k, block in enumerate(blocks):
        start = None
        previous = None
        for t, values in history:
            status = values[k][3]
            if previous == 'normal' and status != 'normal':
                start = t
            if status == 'normal' and start is not None:
                if 0 < t-start <= MAX_SECONDS:
                    events.append(dict(label=block['label'],row=block['row'],start=start,end=t))
                start = None
            previous = status
    result = {b['label']: [] for b in blocks}
    for event in events:
        peers = [e for e in events if e['row']==event['row']
                 and abs(e['start']-event['start'])<=ONSET_SECONDS]
        if len({e['label'] for e in peers}) >= MIN_BLOCKS:
            result[event['label']].append(dict(start_s=event['start'],end_s=event['end']))
    return result


def bands(windows, duration):
    body = ''
    for window in windows:
        start, end = window['start_s'], window['end_s']
        title = html.escape(f'mask共同短时扰动 {start:.2f}–{end:.2f}s；仅标记，未排除检测')
        body += (f'<rect x="{40+840*start/duration:.2f}" y="10" '
                 f'width="{max(.5,840*(end-start)/duration):.2f}" height="115" '
                 f'fill="#888" fill-opacity="0.20"><title>{title}</title></rect>'
                 f'<path d="M{40+840*start/duration:.2f} 10 V125" '
                 f'stroke="#777" stroke-dasharray="3 3"/>')
    return body

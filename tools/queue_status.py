"""Read-only queue inspection, including CPU-only work and process identity."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def process(pid, expected_identity):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, same_process=bool(expected_identity) and fields[19] == expected_identity,
                    process_state=fields[0], session_id=int(fields[3]),
                    terminal_attached=int(fields[4]) != 0)
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, same_process=False)


def inspect(root):
    jobs = json.loads((root / 'jobs.json').read_text())
    state = json.loads((root / 'state.json').read_text())
    records = state.get('jobs', {})
    counts = Counter(records.get(job['id'], {}).get('status', 'pending') for job in jobs)
    running = []
    for key, record in records.items():
        if record.get('status') != 'running':
            continue
        running.append(dict(id=key, elapsed_seconds=round(time.time() - record['started'], 1),
            process=process(record['pid'], record.get('process_identity')),
            command=record.get('command'), log=str(root / (key + '.log'))))
    return dict(total=len(jobs), counts=dict(counts),
                heartbeat_age_seconds=round(time.time() - state['heartbeat'], 1)
                if 'heartbeat' in state else None,
                queue_finished=state.get('queue_finished', False),
                all_complete=state.get('all_complete', False), running=running,
                note='A live process/heartbeat is not proof of useful progress. GPU idle can mean CPU mapping or disk IO.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Queue lane containing jobs.json/state.json')
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), indent=2))

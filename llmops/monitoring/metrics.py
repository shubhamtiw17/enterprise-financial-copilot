from collections import defaultdict
from typing import Dict

_counters: Dict[str, int] = defaultdict(int)
_latencies: Dict[str, list] = defaultdict(list)


def increment(metric: str) -> None:
    _counters[metric] += 1


def record_latency(metric: str, ms: float) -> None:
    _latencies[metric].append(ms)


def get_summary() -> dict:
    return {
        "counters": dict(_counters),
        "avg_latencies": {
            k: round(sum(v) / len(v), 2)
            for k, v in _latencies.items() if v
        },
    }

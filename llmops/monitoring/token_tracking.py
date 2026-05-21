from collections import defaultdict
from typing import Dict

_token_counts: Dict[str, int] = defaultdict(int)


def record_tokens(model: str, input_tokens: int, output_tokens: int) -> None:
    _token_counts[f"{model}.input"] += input_tokens
    _token_counts[f"{model}.output"] += output_tokens


def get_token_summary() -> dict:
    return dict(_token_counts)

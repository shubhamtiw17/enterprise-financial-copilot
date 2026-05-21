from typing import List, Dict


class MemoryManager:
    def __init__(self, max_turns: int = 10):
        self._history: List[Dict] = []
        self._max_turns = max_turns

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        if len(self._history) > self._max_turns * 2:
            self._history = self._history[-self._max_turns * 2:]

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def clear(self) -> None:
        self._history = []

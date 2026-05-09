"""
memory.py — E-mboni Temporal Filtering & Intelligence Engine
Implements the "Memory" logic to prevent audio spam in high-density
environments like Nyabugogo by tracking known objects and state changes.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


# How close (in normalized coords) two positions must be to count as "same object"
POSITION_TOLERANCE = 0.08

# How long (seconds) an object stays in memory before it's considered "new" again
MEMORY_TTL = 2.5


@dataclass
class CachedObject:
    label: str
    norm_x: float
    norm_y: float
    priority: str
    last_seen: float = field(default_factory=time.time)


class MemoryEngine:
    """
    Tracks recently reported objects to suppress repeated alerts.
    Also monitors obstacle state to trigger "Path clear" only on
    transition from obstacles → no obstacles.
    """

    def __init__(self):
        self._cache: dict[str, CachedObject] = {}
        self._prev_had_obstacles: bool = False

    def _is_same_position(self, cached: CachedObject, norm_x: float, norm_y: float) -> bool:
        """Returns True if the new detection is close enough to the cached one."""
        return (
            abs(cached.norm_x - norm_x) < POSITION_TOLERANCE and
            abs(cached.norm_y - norm_y) < POSITION_TOLERANCE
        )

    def _evict_expired(self):
        """Remove objects not seen within MEMORY_TTL seconds."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now - v.last_seen > MEMORY_TTL]
        for k in expired:
            del self._cache[k]

    def is_known(self, label: str, norm_x: float, norm_y: float) -> bool:
        """
        Returns True if this object was already reported recently
        at nearly the same position — suppress the alert.
        """
        self._evict_expired()
        if label in self._cache:
            cached = self._cache[label]
            if self._is_same_position(cached, norm_x, norm_y):
                cached.last_seen = time.time()  # refresh TTL
                return True
        return False

    def remember(self, label: str, norm_x: float, norm_y: float, priority: str):
        """Store or update an object in the cache after reporting it."""
        self._cache[label] = CachedObject(
            label=label,
            norm_x=norm_x,
            norm_y=norm_y,
            priority=priority,
        )

    def check_path_clear(self, has_obstacles: bool) -> Optional[str]:
        """
        State-change monitor.
        Returns "Path clear." ONLY when transitioning from obstacles → no obstacles.
        Returns None in all other cases (prevents constant chatter in empty hallways).
        """
        msg = None
        if self._prev_had_obstacles and not has_obstacles:
            msg = "Path clear."
        self._prev_had_obstacles = has_obstacles
        return msg

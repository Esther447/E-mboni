 """
memory.py — E-mboni Temporal Filtering & Intelligence Engine
Implements the "Memory" logic to prevent audio spam in high-density
environments like Nyabugogo by tracking known objects and state changes.
Also detects approaching vs. stationary objects via box_area history.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# How close (in normalized coords) two positions must be to count as "same object"
POSITION_TOLERANCE = 0.08

# How long (seconds) an object stays in memory before it's considered "new" again
MEMORY_TTL = 2.5

# Mute period (seconds) for stationary objects after first report
STATIONARY_MUTE = 20.0

# Number of frames to track for motion detection
MOTION_WINDOW = 5

# Minimum % growth in box_area over MOTION_WINDOW frames to classify as "approaching"
APPROACHING_THRESHOLD = 0.25  # 25% area growth = approaching

# Minimum % shrink in box_area over MOTION_WINDOW frames to classify as "retreating"
RETREATING_THRESHOLD = -0.20  # 20% area shrink = moving away

# Crowd mode: number of persons within ~2m to trigger crowd summary
CROWD_THRESHOLD = 3
CROWD_NEAR_AREA = 60000  # box_area proxy for ~2 metres


class MotionState:
    APPROACHING = "APPROACHING"   # box_area growing fast — escalate to HIGH
    STATIONARY  = "STATIONARY"    # box_area stable — parked car, static obstacle
    RETREATING  = "RETREATING"    # box_area shrinking — moving away, lower priority
    UNKNOWN     = "UNKNOWN"       # not enough frames yet


@dataclass
class CachedObject:
    label: str
    norm_x: float
    norm_y: float
    priority: str
    last_seen: float = field(default_factory=time.time)
    first_reported: Optional[float] = None        # time of first report
    stationary_muted_at: Optional[float] = None   # time mute period started
    area_history: deque = field(default_factory=lambda: deque(maxlen=MOTION_WINDOW))

    def update_area(self, box_area: float):
        self.area_history.append(box_area)

    def get_motion_state(self) -> str:
        """Compares oldest vs newest area over MOTION_WINDOW frames."""
        if len(self.area_history) < MOTION_WINDOW:
            return MotionState.UNKNOWN
        oldest = self.area_history[0]
        newest = self.area_history[-1]
        if oldest == 0:
            return MotionState.UNKNOWN
        change = (newest - oldest) / oldest
        if change >= APPROACHING_THRESHOLD:
            return MotionState.APPROACHING
        if change <= RETREATING_THRESHOLD:
            return MotionState.RETREATING
        return MotionState.STATIONARY


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
        Stationary objects enter a 20s mute period after first report.
        Approaching objects always pass through.
        """
        self._evict_expired()
        if label in self._cache:
            cached = self._cache[label]
            if self._is_same_position(cached, norm_x, norm_y):
                cached.last_seen = time.time()
                # Approaching objects always re-alert
                if cached.get_motion_state() == MotionState.APPROACHING:
                    return False
                # Stationary: enforce 20s mute period
                if cached.get_motion_state() == MotionState.STATIONARY:
                    if cached.stationary_muted_at is not None:
                        if time.time() - cached.stationary_muted_at < STATIONARY_MUTE:
                            return True  # still in mute period
                        else:
                            cached.stationary_muted_at = None  # mute expired
                return True
        return False

    def track(self, label: str, norm_x: float, norm_y: float,
              box_area: float, priority: str) -> str:
        """
        Updates area history for an object and returns its current MotionState.
        Call this every frame for every detected object.
        """
        if label not in self._cache:
            self._cache[label] = CachedObject(
                label=label, norm_x=norm_x, norm_y=norm_y, priority=priority
            )
        cached = self._cache[label]
        cached.norm_x = norm_x
        cached.norm_y = norm_y
        cached.last_seen = time.time()
        cached.update_area(box_area)
        return cached.get_motion_state()

    def remember(self, label: str, norm_x: float, norm_y: float, priority: str):
        """Mark object as reported. Starts stationary mute period if applicable."""
        now = time.time()
        if label in self._cache:
            cached = self._cache[label]
            cached.priority = priority
            if cached.first_reported is None:
                cached.first_reported = now
            # Start mute period when object is confirmed stationary
            if (cached.get_motion_state() == MotionState.STATIONARY
                    and cached.stationary_muted_at is None):
                cached.stationary_muted_at = now
        else:
            self._cache[label] = CachedObject(
                label=label, norm_x=norm_x, norm_y=norm_y,
                priority=priority, first_reported=now
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


# ---------------------------------------------------------------------------
# Crowd Detector
# ---------------------------------------------------------------------------

class CrowdDetector:
    """
    Monitors person density to switch from individual alerts to crowd summaries.
    In busy environments like Nyabugogo or Kimironko market, individual
    "person ahead" alerts become noise. This collapses them into one summary.

    Priority bypass: HIGH hazards (stair, car) always interrupt crowd mode.
    """

    def __init__(self):
        self._in_crowd_mode: bool = False

    def _count_nearby_persons(self, raw_detections: list) -> tuple[int, list[float]]:
        """Count persons within ~2m and collect their norm_x positions."""
        nearby = [
            r for r in raw_detections
            if r.label == "person" and r.box_area > CROWD_NEAR_AREA
        ]
        return len(nearby), [r.norm_x for r in nearby]

    def _crowd_direction(self, positions: list[float]) -> str:
        """Suggest navigation direction away from crowd centre."""
        if not positions:
            return "straight ahead"
        avg_x = sum(positions) / len(positions)
        if avg_x < 0.4:
            return "navigate right"
        elif avg_x > 0.6:
            return "navigate left"
        return "slow down"

    def evaluate(self, raw_detections: list, payload: list) -> tuple[bool, Optional[str]]:
        """
        Returns (is_crowd_mode, crowd_message).

        - is_crowd_mode: True if density threshold exceeded
        - crowd_message: summary string or None

        HIGH priority hazards always bypass crowd mode and return is_crowd_mode=False
        so the main loop handles them normally.
        """
        # HIGH hazards always bypass crowd mode
        high_hazards = [p for p in payload if p.priority == "HIGH"]
        if high_hazards:
            self._in_crowd_mode = False
            return False, None

        count, positions = self._count_nearby_persons(raw_detections)

        if count >= CROWD_THRESHOLD:
            self._in_crowd_mode = True
            direction = self._crowd_direction(positions)
            msg = f"Crowd ahead, {direction}. {count} people within 2 meters."
            return True, msg

        self._in_crowd_mode = False
        return False, None

"""
memory.py — E-mboni Temporal Filtering & Intelligence Engine
Implements the "Memory" logic to prevent audio spam in high-density
environments like Nyabugogo by tracking known objects and state changes.
Also detects approaching vs. stationary objects via box_area history.
"""

import time
from collections import deque, Counter
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

# Ghosting: frames to keep object in memory after it disappears
GHOST_FRAMES = 3

# Path clear: consecutive clean frames required before announcing
PATH_CLEAR_FRAMES = 10

# Bottom zone: norm_y threshold for ground-level priority (Rwanda terrain)
BOTTOM_ZONE_Y = 0.80

# Time-to-collision: box_area growth % over 2 frames to trigger STOP
TTC_THRESHOLD = 0.20  # 20% growth in 2 frames = imminent collision


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
    first_reported: Optional[float] = None
    stationary_muted_at: Optional[float] = None
    ghost_frames_remaining: int = 0          # frames to keep alive after disappearing
    area_history: deque = field(default_factory=lambda: deque(maxlen=MOTION_WINDOW))
    last_two_areas: deque = field(default_factory=lambda: deque(maxlen=2))  # for TTC

    def update_area(self, box_area: float):
        self.area_history.append(box_area)
        self.last_two_areas.append(box_area)

    def get_ttc_critical(self) -> bool:
        """
        Time-to-Collision check over last 2 frames.
        Returns True if box grew >= 20% — imminent collision, skip all polite logic.
        """
        if len(self.last_two_areas) < 2:
            return False
        old, new = self.last_two_areas[0], self.last_two_areas[1]
        if old == 0:
            return False
        return (new - old) / old >= TTC_THRESHOLD

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
    Full intelligence engine:
    - Temporal filtering (suppress known objects)
    - Ghosting timer (keep objects alive for GHOST_FRAMES after disappearing)
    - TTC (Time-to-Collision) fast escalation
    - 10-frame path clear certainty
    - Vibration lock for bottom zone (Rwanda terrain)
    - Stationary 20s mute period
    """

    def __init__(self):
        self._cache: dict[str, CachedObject] = {}
        self._prev_had_obstacles: bool = False
        self._clean_frame_count: int = 0      # consecutive frames with no HIGH/MEDIUM
        self._vibe_lock: bool = False          # True = continuous vibration active
        self._vibe_lock_label: Optional[str] = None

    @property
    def vibe_locked(self) -> bool:
        return self._vibe_lock

    @property
    def vibe_lock_label(self) -> Optional[str]:
        return self._vibe_lock_label

    def _is_same_position(self, cached: CachedObject, norm_x: float, norm_y: float) -> bool:
        """Returns True if the new detection is close enough to the cached one."""
        return (
            abs(cached.norm_x - norm_x) < POSITION_TOLERANCE and
            abs(cached.norm_y - norm_y) < POSITION_TOLERANCE
        )

    def _evict_expired(self):
        """Remove objects not seen within MEMORY_TTL seconds AND ghost frames exhausted."""
        now = time.time()
        to_remove = []
        for k, v in self._cache.items():
            if now - v.last_seen > MEMORY_TTL and v.ghost_frames_remaining <= 0:
                to_remove.append(k)
            elif now - v.last_seen > MEMORY_TTL:
                v.ghost_frames_remaining -= 1  # count down ghost frames
        for k in to_remove:
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
        Updates area history and returns MotionState.
        Also resets ghost frames when object is seen again.
        """
        if label not in self._cache:
            self._cache[label] = CachedObject(
                label=label, norm_x=norm_x, norm_y=norm_y, priority=priority
            )
        cached = self._cache[label]
        cached.norm_x = norm_x
        cached.norm_y = norm_y
        cached.last_seen = time.time()
        cached.ghost_frames_remaining = GHOST_FRAMES  # reset ghost on detection
        cached.update_area(box_area)
        return cached.get_motion_state()

    def is_ttc_critical(self, label: str) -> bool:
        """Returns True if object grew 20%+ in last 2 frames — imminent collision."""
        cached = self._cache.get(label)
        return cached.get_ttc_critical() if cached else False

    def update_vibe_lock(self, raw_detections: list):
        """
        Vibration lock for bottom zone (Rwanda terrain).
        If stair/ground hazard detected in bottom 20% of frame,
        lock continuous vibration until zone is clear.
        """
        bottom_hazards = [
            r for r in raw_detections
            if r.norm_y > BOTTOM_ZONE_Y and r.label in ["stair", "fire hydrant", "parking meter"]
        ]
        if bottom_hazards:
            self._vibe_lock = True
            self._vibe_lock_label = bottom_hazards[0].label
        else:
            self._vibe_lock = False
            self._vibe_lock_label = None

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
        Requires PATH_CLEAR_FRAMES (10) consecutive clean frames before
        announcing path clear. Immediately resets if obstacle reappears.
        """
        if has_obstacles:
            self._clean_frame_count = 0
            self._prev_had_obstacles = True
            return None

        if self._prev_had_obstacles:
            self._clean_frame_count += 1
            if self._clean_frame_count >= PATH_CLEAR_FRAMES:
                self._clean_frame_count = 0
                self._prev_had_obstacles = False
                return "Path clear."
        return None


# ---------------------------------------------------------------------------
# Consistency Filter — object must appear in N consecutive frames to be reported
# ---------------------------------------------------------------------------

# Consistency: frames an object must appear in to be confirmed
CONSISTENCY_REQUIRED = 3   # raised from 2 — object must appear in 3 of last 5 frames
CONSISTENCY_WINDOW   = 5   # sliding window size

# Person motion: position change threshold to classify as moving vs stationary
PERSON_MOTION_THRESHOLD = 0.04  # normalised coords — if centre moves > 4% of frame width, moving


class PersonMotionState:
    MOVING   = "moving"
    SITTING  = "sitting"
    STANDING = "standing"
    UNKNOWN  = "unknown"


class ConsistencyFilter:
    """
    Tracks the last CONSISTENCY_WINDOW frames for each label.
    An object is confirmed if it appears in >= CONSISTENCY_REQUIRED of those frames.
    This is more robust than a simple streak counter — a one-frame dropout
    (e.g. YOLO misses frame 3 but sees frames 1,2,4,5) still confirms.

    Also tracks person position across frames to classify:
    - person moving   (centre x/y shifts > threshold between frames)
    - person sitting  (bbox aspect ratio tall+narrow AND stationary)
    - person standing (bbox aspect ratio tall, stationary, centre-mid frame)
    """

    def __init__(self):
        self._history: dict[str, deque] = {}       # label → deque of 0/1 (seen/not seen)
        self._person_positions: deque = deque(maxlen=CONSISTENCY_WINDOW)  # (norm_x, norm_y, box_h, box_w)

    def update(self, detected_labels: list[str]):
        """Call once per frame with the list of detected labels."""
        current = set(detected_labels)

        # Update sliding window for every tracked label
        all_known = set(self._history.keys()) | current
        for label in all_known:
            if label not in self._history:
                self._history[label] = deque(maxlen=CONSISTENCY_WINDOW)
            self._history[label].append(1 if label in current else 0)

    def update_person_position(self, norm_x: float, norm_y: float, box_w: float, box_h: float):
        """Call when a person is detected, to track their position over frames."""
        self._person_positions.append((norm_x, norm_y, box_w, box_h))

    def is_confirmed(self, label: str) -> bool:
        """Returns True if object appeared in >= CONSISTENCY_REQUIRED of last CONSISTENCY_WINDOW frames."""
        history = self._history.get(label)
        if not history:
            return False
        return sum(history) >= CONSISTENCY_REQUIRED

    def get_person_motion_state(self) -> str:
        """
        Analyses recent person position history to classify motion state.

        - MOVING:   centre position changed > PERSON_MOTION_THRESHOLD between frames
        - SITTING:  stationary + bbox wider relative to height (aspect < 1.2)
        - STANDING: stationary + bbox tall (aspect >= 1.2)
        - UNKNOWN:  not enough frames yet
        """
        if len(self._person_positions) < 2:
            return PersonMotionState.UNKNOWN

        # Check position change across all stored frames
        positions = list(self._person_positions)
        max_movement = 0.0
        for i in range(1, len(positions)):
            dx = abs(positions[i][0] - positions[i-1][0])
            dy = abs(positions[i][1] - positions[i-1][1])
            max_movement = max(max_movement, dx, dy)

        if max_movement > PERSON_MOTION_THRESHOLD:
            return PersonMotionState.MOVING

        # Stationary — classify by aspect ratio of most recent bbox
        _, _, box_w, box_h = positions[-1]
        aspect = box_h / box_w if box_w > 0 else 1.0
        if aspect < 1.2:
            return PersonMotionState.SITTING
        return PersonMotionState.STANDING


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

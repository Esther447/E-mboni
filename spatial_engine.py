"""
spatial_engine.py — E-mboni Spatial Logic Engine
Translates YOLOv8 bounding box data into natural language directions,
distance estimates, and structured JSON payloads for the frontend.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Object Priority Tiers
# ---------------------------------------------------------------------------

DANGER_OBJECTS = [
    "person", "car", "bus", "truck", "motorcycle", "bicycle",
    "stair", "traffic light",
]

NAVIGATION_OBJECTS = [
    "chair", "dining table", "door", "sofa", "bed", "toilet",
]

UTILITY_OBJECTS = [
    "backpack", "suitcase", "dog", "cat", "fire hydrant",
]

ALL_OBJECTS = DANGER_OBJECTS + NAVIGATION_OBJECTS + UTILITY_OBJECTS


# ---------------------------------------------------------------------------
# Input / Output Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RawDetection:
    label: str
    box_area: float   # pixel area: (x2-x1) * (y2-y1)
    norm_x: float     # normalized center x: 0.0 (left) → 1.0 (right)
    norm_y: float     # normalized center y: 0.0 (top)  → 1.0 (bottom)

@dataclass
class DetectionResult:
    object: str
    direction: str          # e.g. "straight ahead, low"
    distance: Optional[str] # e.g. "1 meter" or None if too far
    priority: str           # HIGH / MEDIUM / LOW
    vibe: Optional[str]     # STRONG / LIGHT / PULSE / None
    vibe_pattern: Optional[str] # e.g. "THREE_SHORT" for stairs, "LONG" for overhangs
    speech: str             # full natural language alert
    vertical_zone: Optional[str] # GROUND_LEVEL / OVERHANG / None


# ---------------------------------------------------------------------------
# Direction Engine
# ---------------------------------------------------------------------------

def get_horizontal(norm_x: float) -> str:
    """5-zone horizontal positioning."""
    if norm_x < 0.2:    return "to your far left"
    elif norm_x < 0.4:  return "on your left"
    elif norm_x <= 0.6: return "straight ahead"
    elif norm_x <= 0.8: return "on your right"
    else:               return "to your far right"

def get_vertical(norm_y: float) -> Optional[str]:
    """Above/below detection for head-height vs ground-level objects."""
    if norm_y < 0.3:   return "high"
    elif norm_y > 0.7: return "low"
    return None

def get_direction(norm_x: float, norm_y: float) -> str:
    """
    Combines horizontal and vertical into a natural language direction.

    Examples:
        (0.5, 0.5) → "straight ahead"
        (0.1, 0.5) → "to your far left"
        (0.5, 0.8) → "straight ahead, low"
        (0.9, 0.2) → "to your far right, high"
    """
    horizontal = get_horizontal(norm_x)
    vertical = get_vertical(norm_y)
    if vertical:
        return f"{horizontal}, {vertical}"
    return horizontal


# ---------------------------------------------------------------------------
# Distance Estimator
# ---------------------------------------------------------------------------

# Thresholds calibrated for a standard laptop webcam at 640x480.
# Adjust DISTANCE_THRESHOLDS if using a different resolution or focal length.
DISTANCE_THRESHOLDS = [
    (150000, "50 centimeters"),
    (120000, "1 meter"),
    (60000,  "2 meters"),
    (30000,  "3 to 4 meters"),
]

def get_distance(box_area: float) -> Optional[str]:
    """
    Estimates real-world distance from bounding box pixel area.
    Returns None if object is too far to be actionable.

    Formula basis: Distance ∝ 1 / sqrt(box_area)
    Thresholds derived from: Distance = (KnownWidth × FocalLength) / PixelWidth
    """
    for threshold, label in DISTANCE_THRESHOLDS:
        if box_area > threshold:
            return label
    return None


# ---------------------------------------------------------------------------
# Vertical Awareness — Rwanda Terrain
# ---------------------------------------------------------------------------

# Vertical zone labels
GROUND_LEVEL = "GROUND_LEVEL"  # norm_y > 0.7 — stairs, curbs, open drains
OVERHANG     = "OVERHANG"      # norm_y < 0.3 — branches, signs, low ceilings

# Objects that are specifically dangerous at ground level
GROUND_HAZARDS = ["stair", "fire hydrant", "parking meter"]

# Objects that are specifically dangerous as overhangs
OVERHANG_HAZARDS = ["potted plant", "traffic light", "stop sign"]

# Vibration patterns (sent to frontend/Android Vibrator API)
class VibePattern:
    THREE_SHORT = "THREE_SHORT"  # ···  stair / ground drop — change in level
    TWO_SHORT   = "TWO_SHORT"    # ··   general ground hazard
    LONG        = "LONG"         # ———  overhang — sustained warning
    STRONG      = "STRONG"       # ████ immediate danger
    LIGHT       = "LIGHT"        # ·    navigation hint


def get_vertical_zone(norm_y: float) -> Optional[str]:
    """Classify object into vertical zone based on y position."""
    if norm_y > 0.7: return GROUND_LEVEL
    if norm_y < 0.3: return OVERHANG
    return None

def get_vibe_pattern(label: str, vertical_zone: Optional[str], priority: str) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (vibe, vibe_pattern) based on label, vertical zone and priority.

    Stair at ground level  → PULSE + THREE_SHORT (change in ground level)
    Any object as overhang → PULSE + LONG        (head/body hazard)
    HIGH priority          → STRONG              (immediate danger)
    MEDIUM priority        → LIGHT               (navigation hint)
    """
    # Stair specifically at ground level — three short pulses
    if label == "stair" and vertical_zone == GROUND_LEVEL:
        return "PULSE", VibePattern.THREE_SHORT

    # Any ground hazard at ground level
    if label in GROUND_HAZARDS and vertical_zone == GROUND_LEVEL:
        return "PULSE", VibePattern.TWO_SHORT

    # Overhang — long sustained vibration
    if vertical_zone == OVERHANG:
        return "PULSE", VibePattern.LONG

    # Default priority-based vibe
    if priority == "HIGH":   return VibePattern.STRONG, None
    if priority == "MEDIUM": return VibePattern.LIGHT, None
    return None, None


def build_vertical_speech(label: str, horizontal: str, vertical_zone: Optional[str],
                          distance: Optional[str], priority: str) -> str:
    """Builds terrain-aware speech with specific warnings for stairs and overhangs."""
    dist_str = f", {distance}" if distance else ""

    if label == "stair" and vertical_zone == GROUND_LEVEL:
        return f"Warning! Stairs below {horizontal}{dist_str}. Watch your step."

    if vertical_zone == GROUND_LEVEL and label in GROUND_HAZARDS:
        return f"Ground hazard {horizontal}{dist_str}. {label} below."

    if vertical_zone == OVERHANG:
        return f"Overhead obstacle {horizontal}{dist_str}. Duck or move aside."

    direction = f"{horizontal}, {'low' if vertical_zone == GROUND_LEVEL else 'high'}" if vertical_zone else horizontal
    if priority == "HIGH":
        return f"STOP. {label} {direction}{dist_str}"
    return f"{label} {direction}{dist_str}"



def get_priority(label: str) -> str:
    if label in DANGER_OBJECTS:     return "HIGH"
    if label in NAVIGATION_OBJECTS: return "MEDIUM"
    if label in UTILITY_OBJECTS:    return "LOW"
    return "NONE"

def get_vibe(priority: str) -> Optional[str]:
    return {"HIGH": VibePattern.STRONG, "MEDIUM": VibePattern.LIGHT}.get(priority)

def is_actionable(priority: str, box_area: float) -> bool:
    """
    Distance gate per priority tier:
    - HIGH:   any distance (immediate danger)
    - MEDIUM: < 2m (box_area > 60000)
    - LOW:    < 1m (box_area > 120000)
    """
    if priority == "HIGH":   return True
    if priority == "MEDIUM": return box_area > 60000
    if priority == "LOW":    return box_area > 120000
    return False


# ---------------------------------------------------------------------------
# Speech Builder
# ---------------------------------------------------------------------------

def build_speech(label: str, direction: str, distance: Optional[str], priority: str) -> str:
    dist_str = f", {distance}" if distance else ""
    if priority == "HIGH":
        return f"STOP. {label} {direction}{dist_str}"
    return f"{label} {direction}{dist_str}"


# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------

def process_detections(raw_detections: list[RawDetection]) -> list[DetectionResult]:
    """
    Takes a list of RawDetection objects and returns a sorted list of
    DetectionResult objects ready for the API response or voice engine.

    Sorted by priority: HIGH → MEDIUM → LOW
    """
    results = []

    for det in raw_detections:
        if det.label not in ALL_OBJECTS:
            continue

        priority = get_priority(det.label)
        if not is_actionable(priority, det.box_area):
            continue

        direction     = get_direction(det.norm_x, det.norm_y)
        horizontal    = get_horizontal(det.norm_x)
        vertical_zone = get_vertical_zone(det.norm_y)
        distance      = get_distance(det.box_area)
        vibe, vibe_pattern = get_vibe_pattern(det.label, vertical_zone, priority)
        speech        = build_vertical_speech(det.label, horizontal, vertical_zone, distance, priority)

        results.append(DetectionResult(
            object=det.label,
            direction=direction,
            distance=distance,
            priority=priority,
            vibe=vibe,
            vibe_pattern=vibe_pattern,
            speech=speech,
            vertical_zone=vertical_zone,
        ))

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    results.sort(key=lambda r: priority_order.get(r.priority, 3))
    return results

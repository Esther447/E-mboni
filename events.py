"""
events.py — E-mboni Privacy-Safe Event System

Implements the data flow table:

Scenario         | Local Device  | Backend           | Guardian          | Admin
Normal Walking   | Audio/Vibe    | Frame deleted     | No data           | No data
High-Speed Hazard| STOP alert    | Logged as event   | Safety Alert text | System: Online
Lost Device      | Silent        | Wait for ping     | Last Seen request | Device ID only
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class EventType(str, Enum):
    SAFETY_EVENT  = "SAFETY_EVENT"   # HIGH priority hazard detected
    DEVICE_PING   = "DEVICE_PING"    # Heartbeat — device is online
    DEVICE_LOST   = "DEVICE_LOST"    # No ping received within threshold


@dataclass
class SafetyEvent:
    """
    Logged ONLY when a HIGH priority hazard is detected.
    Contains NO camera data, NO location, NO personal identifiers.
    Only the hazard type, direction, and timestamp.
    """
    event_type: str = EventType.SAFETY_EVENT
    hazard: str = ""                  # e.g. "stair", "car"
    direction: str = ""               # e.g. "straight ahead, low"
    distance: Optional[str] = None    # e.g. "1 meter"
    timestamp: float = field(default_factory=time.time)

    def to_guardian_alert(self) -> dict:
        """
        What the guardian receives — text only, no camera, no location.
        """
        return {
            "type": "SAFETY_ALERT",
            "message": f"Safety alert: {self.hazard} detected {self.direction}.",
            "distance": self.distance,
            "time": self.timestamp,
            "note": "No camera or location data included."
        }

    def to_admin_view(self) -> dict:
        """
        What admin sees — system status only, no hazard details, no personal data.
        """
        return {
            "status": "online",
            "event": "SAFETY_EVENT",
            "note": "No user data accessible."
        }


@dataclass
class DevicePing:
    """
    Heartbeat sent periodically to confirm device is online.
    Contains only device_id and timestamp — no user data.
    """
    device_id: str
    timestamp: float = field(default_factory=time.time)
    status: str = "online"


# ---------------------------------------------------------------------------
# In-memory event store (replace with DB in production)
# ---------------------------------------------------------------------------

class EventStore:
    """
    Minimal in-memory store for safety events and device pings.
    Frames are NEVER stored — only structured event metadata.
    """

    def __init__(self):
        self._safety_events: list[SafetyEvent] = []
        self._device_pings: dict[str, DevicePing] = {}
        self._location_sharing: dict[str, bool] = {}   # user_id → sharing enabled
        self._last_known_location: dict[str, dict] = {} # user_id → {lat, lng, time}

    # --- Safety Events ---

    def log_safety_event(self, hazard: str, direction: str,
                         distance: Optional[str] = None) -> SafetyEvent:
        """
        Called when HIGH priority hazard detected.
        Frame is NOT stored — only the structured event.
        """
        event = SafetyEvent(hazard=hazard, direction=direction, distance=distance)
        self._safety_events.append(event)
        # Keep only last 50 events in memory
        if len(self._safety_events) > 50:
            self._safety_events = self._safety_events[-50:]
        return event

    def get_guardian_alerts(self, since: float = 0.0) -> list[dict]:
        """
        Returns text-only alerts for guardian — no camera, no location.
        """
        return [
            e.to_guardian_alert()
            for e in self._safety_events
            if e.timestamp > since
        ]

    # --- Device Ping / Lost Device ---

    def ping(self, device_id: str) -> DevicePing:
        """Record a heartbeat from a device."""
        p = DevicePing(device_id=device_id)
        self._device_pings[device_id] = p
        return p

    def get_device_status(self, device_id: str, timeout: float = 30.0) -> str:
        """
        Returns 'online' or 'offline' based on last ping time.
        Admin sees this only — no user data.
        """
        ping = self._device_pings.get(device_id)
        if not ping:
            return "offline"
        return "online" if (time.time() - ping.timestamp) < timeout else "offline"

    # --- Location (User-Controlled) ---

    def enable_location_sharing(self, user_id: str, enabled: bool):
        """User controls whether guardian can see last known location."""
        self._location_sharing[user_id] = enabled

    def update_last_known_location(self, user_id: str, lat: float, lng: float):
        """Only stored if user has enabled sharing."""
        if self._location_sharing.get(user_id, False):
            self._last_known_location[user_id] = {
                "lat": lat, "lng": lng, "time": time.time()
            }

    def get_last_known_location(self, user_id: str) -> Optional[dict]:
        """
        Guardian can request this — only if user enabled sharing.
        Returns None if sharing is disabled.
        """
        if not self._location_sharing.get(user_id, False):
            return None
        return self._last_known_location.get(user_id)


# Singleton store — shared across all requests
event_store = EventStore()

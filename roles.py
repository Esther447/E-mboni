from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class Role(str, Enum):
    USER = "user"           # Visually impaired — primary user
    GUARDIAN = "guardian"   # Caregiver — limited alerts only
    ADMIN = "admin"         # System maintainer — no personal data access


@dataclass
class Permissions:
    realtime_ai: bool = False        # Live camera detection + voice/vibration
    local_settings: bool = False     # Change app settings
    last_known_location: bool = False # View user's last location (guardian only, if user enables)
    emergency_alerts: bool = False   # Receive fall/danger alerts
    device_management: bool = False  # Register/update hardware + licenses
    see_camera: bool = False         # View live camera feed — ALWAYS False for guardian/admin
    track_actions: bool = False      # Log user movements/actions — ALWAYS False for guardian/admin
    see_location: bool = False       # View real-time location — ALWAYS False for admin


ROLE_PERMISSIONS = {
    Role.USER: Permissions(
        realtime_ai=True,
        local_settings=True,
        last_known_location=True,   # Controls their own sharing
        emergency_alerts=True,
        see_camera=True,
        track_actions=False,        # No server-side action logging
        see_location=True,
    ),
    Role.GUARDIAN: Permissions(
        realtime_ai=False,
        local_settings=False,
        last_known_location=True,   # Only if user has enabled sharing
        emergency_alerts=True,
        device_management=False,
        see_camera=False,           # Privacy wall — cannot see camera
        track_actions=False,        # Privacy wall — text alerts only
        see_location=False,         # Only last known, not real-time
    ),
    Role.ADMIN: Permissions(
        realtime_ai=False,
        local_settings=False,
        last_known_location=False,  # Privacy wall — no location access
        emergency_alerts=False,
        device_management=True,
        see_camera=False,           # Privacy wall — cannot see camera
        track_actions=False,        # Privacy wall — no action tracking
        see_location=False,         # Privacy wall — no location access
    ),
}


def get_permissions(role: Role) -> Permissions:
    return ROLE_PERMISSIONS[role]

def can_access(role: Role, permission: str) -> bool:
    return getattr(get_permissions(role), permission, False)

def get_allowed_api_routes(role: Role) -> list:
    routes = {
        Role.USER: [
            "POST /detect",
            "GET /settings",
            "PUT /settings",
            "POST /location/share",
        ],
        Role.GUARDIAN: [
            "GET /alerts",
            "GET /location/last-known",  # Only if user has enabled sharing
        ],
        Role.ADMIN: [
            "GET /devices",
            "POST /devices/register",
            "PUT /devices/update",
            "GET /devices/status",      # Online/Offline only — no personal data
        ],
    }
    return routes.get(role, [])

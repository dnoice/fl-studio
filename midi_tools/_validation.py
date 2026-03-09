"""MIDI validation utilities.

Centralized validation functions for MIDI note, velocity, and channel values.
"""


def validate_pitch(pitch: int) -> int:
    """Clamp a MIDI pitch value to the valid range 0-127."""
    return max(0, min(127, pitch))


def validate_velocity(velocity: int) -> int:
    """Clamp a MIDI velocity value to the valid range 0-127."""
    return max(0, min(127, velocity))


def validate_channel(channel: int) -> int:
    """Clamp a MIDI channel to the valid range 0-15."""
    return max(0, min(15, channel))


def is_valid_pitch(pitch: int) -> bool:
    """Check if a MIDI pitch value is in the valid range 0-127."""
    return 0 <= pitch <= 127


def is_valid_velocity(velocity: int) -> bool:
    """Check if a MIDI velocity value is in the valid range 0-127."""
    return 0 <= velocity <= 127

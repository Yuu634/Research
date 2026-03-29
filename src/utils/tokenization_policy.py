from typing import Literal

VelocityLevel = Literal["Ghost", "Normal", "Accent"]

POSITIONS_PER_BEAT = 24
CP_POS_PAD_VALUE = 24


def classify_velocity_level(velocity: int) -> VelocityLevel:
    if not isinstance(velocity, int):
        raise ValueError(f"velocity must be int, got: {type(velocity).__name__}")
    if velocity < 0 or velocity > 127:
        raise ValueError(f"velocity must be in range 0..127, got: {velocity}")

    if velocity < 15:
        return "Ghost"
    if velocity < 100:
        return "Normal"
    return "Accent"


def quantize_position_in_beat(tick_in_bar: int, beat_resolution: int) -> int:
    if beat_resolution <= 0:
        raise ValueError(f"beat_resolution must be > 0, got: {beat_resolution}")

    return (tick_in_bar % beat_resolution) * POSITIONS_PER_BEAT // beat_resolution


def quantize_step_in_bar(tick_in_bar: int, beat_resolution: int) -> int:
    if beat_resolution <= 0:
        raise ValueError(f"beat_resolution must be > 0, got: {beat_resolution}")

    beat_index = tick_in_bar // beat_resolution
    pos_in_beat = quantize_position_in_beat(tick_in_bar, beat_resolution)
    return beat_index * POSITIONS_PER_BEAT + pos_in_beat

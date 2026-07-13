"""Pure configuration helpers for bounded collection smoke runs."""

from __future__ import annotations

from dataclasses import dataclass


SMOKE_MAX_DECIMATION = 20


@dataclass(frozen=True)
class CollectTiming:
    """Resolved simulator timing for production or smoke collection."""

    decimation: int
    render_interval: int
    smoke_action_steps: int
    smoke_step_timeout_s: float | None

    @property
    def smoke_enabled(self) -> bool:
        return self.smoke_action_steps > 0


def resolve_collect_timing(
    *,
    collect_decimation: int,
    collect_render_interval: int,
    smoke_action_steps: int,
    smoke_decimation: int,
    smoke_step_timeout_s: float,
) -> CollectTiming:
    """Resolve production timing or a bounded, low-decimation smoke mode.

    Smoke intentionally ignores the production timing flags so a copied
    production command cannot accidentally run 300 physics steps per waypoint.
    One render per action step preserves the camera-observation cadence.
    """
    if collect_decimation <= 0:
        raise ValueError(f"collect_decimation must be positive, got {collect_decimation}")
    if collect_render_interval < 0:
        raise ValueError(
            "collect_render_interval must be non-negative, "
            f"got {collect_render_interval}"
        )
    if smoke_action_steps < 0:
        raise ValueError(f"smoke_action_steps must be non-negative, got {smoke_action_steps}")

    if smoke_action_steps == 0:
        return CollectTiming(
            decimation=collect_decimation,
            render_interval=collect_render_interval or collect_decimation,
            smoke_action_steps=0,
            smoke_step_timeout_s=None,
        )

    if not 1 <= smoke_decimation <= SMOKE_MAX_DECIMATION:
        raise ValueError(
            "smoke_decimation must be between 1 and "
            f"{SMOKE_MAX_DECIMATION}, got {smoke_decimation}"
        )
    if smoke_step_timeout_s <= 0:
        raise ValueError(
            "smoke_step_timeout_s must be positive when smoke_action_steps is enabled, "
            f"got {smoke_step_timeout_s}"
        )

    return CollectTiming(
        decimation=smoke_decimation,
        render_interval=smoke_decimation,
        smoke_action_steps=smoke_action_steps,
        smoke_step_timeout_s=smoke_step_timeout_s,
    )

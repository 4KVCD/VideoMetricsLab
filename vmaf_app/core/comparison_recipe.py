"""How the two videos' pictures are prepared before any metric compares
them, and which of those settings decide what a comparison is (for the
cache)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from vmaf_app.core.models import CropMode, ResampleTarget, ScaleDirection


@dataclass(frozen=True, slots=True)
class ComparisonRecipe:
    """The settings that prepare the pictures every metric compares: black
    bars, scaling, the duration limit, and a resolution test."""

    crop_mode: CropMode
    scale_algorithm: str
    scale_direction: ScaleDirection
    duration_limit: float
    resample_test: ResampleTarget | None

    def identity_dict(self) -> dict:
        """Returns the settings that decide what the comparison is, ready for
        JSON. Settings that change only the speed are never part of it.

        The scaling algorithm is left out: a comparison scaled with another
        algorithm, or on the GPU instead of the CPU, counts as the same
        comparison (the user's decision, because the scores differ too
        little to tell apart). Scores saved under one are found under any
        other, and changing it calculates nothing again; the algorithm a
        result was scaled with stays recorded with it."""
        identity = asdict(self)
        del identity["scale_algorithm"]
        # The manual crops are always empty, but are kept so that every saved
        # comparison keeps the name its identity gave it.
        identity.update(manual_source_crop=None, manual_distorted_crop=None)
        return identity

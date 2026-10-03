"""Scientific identity of the common pictures supplied to metric engines."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from vmaf_app.core.models import CropBox, CropMode, ResampleTarget, ScaleDirection


@dataclass(frozen=True, slots=True)
class ComparisonRecipe:
    crop_mode: CropMode
    manual_source_crop: CropBox | None
    manual_distorted_crop: CropBox | None
    scale_algorithm: str
    scale_direction: ScaleDirection
    duration_limit: float
    resample_test: ResampleTarget | None

    def identity_dict(self) -> dict:
        """JSON-ready scientific preprocessing identity, never performance knobs.

        The scaling algorithm is left out: a comparison scaled with another
        algorithm, or on the GPU instead of the CPU, is the same comparison
        (the user's decision, 2026-10-03 -- the scores differ too little to
        tell apart). Scores saved under one are found under any other, and
        changing it calculates nothing again; the algorithm a result was
        scaled with stays recorded with it."""
        identity = asdict(self)
        del identity["scale_algorithm"]
        return identity

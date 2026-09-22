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
        """JSON-ready scientific preprocessing identity, never performance knobs."""
        return asdict(self)

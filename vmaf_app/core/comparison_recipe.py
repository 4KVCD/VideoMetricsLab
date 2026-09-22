"""Scientific identity of the common pictures supplied to metric engines."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from vmaf_app.core.models import CropBox, CropMode, ResampleTarget, ScaleDirection, VmafOptions


@dataclass(frozen=True, slots=True)
class ComparisonRecipe:
    crop_mode: CropMode
    manual_source_crop: CropBox | None
    manual_distorted_crop: CropBox | None
    scale_algorithm: str
    scale_direction: ScaleDirection
    duration_limit: float
    resample_test: ResampleTarget | None

    @classmethod
    def from_vmaf_options(cls, options: VmafOptions) -> ComparisonRecipe:
        return cls(
            crop_mode=options.crop_mode,
            manual_source_crop=options.manual_source_crop,
            manual_distorted_crop=options.manual_distorted_crop,
            scale_algorithm=options.scale_algorithm,
            scale_direction=options.scale_direction,
            duration_limit=options.duration_limit,
            resample_test=options.resample_test,
        )

    def identity_dict(self) -> dict:
        """JSON-ready scientific preprocessing identity, never performance knobs."""
        return asdict(self)

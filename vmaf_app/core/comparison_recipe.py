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


Size = tuple[int, int]


def comparison_sizes(source: Size, distorted: Size, direction: ScaleDirection) -> tuple[Size, Size]:
    """The size each input is compared at, from its content size (after
    crop). Equal sizes need no scaling; otherwise the scale direction says
    which input is brought to the other's size."""
    if source == distorted:
        return source, distorted
    if direction is ScaleDirection.DISTORTED_TO_SOURCE:
        return source, source
    return distorted, distorted


def upscaled_inputs(source: Size, distorted: Size, direction: ScaleDirection) -> tuple[bool, bool]:
    """Whether the reference and the test are each enlarged for comparison.

    Vship scales these itself, on the GPU (see perceptual_vship): FFmpeg then
    pipes the smaller original instead of the enlarged frame. A reduction
    stays in FFmpeg, where it makes the piped frame smaller.
    """
    targets = comparison_sizes(source, distorted, direction)
    return tuple(  # type: ignore[return-value]
        target[0] * target[1] > size[0] * size[1]
        for size, target in zip((source, distorted), targets, strict=True)
    )

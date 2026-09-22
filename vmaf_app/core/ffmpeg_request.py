"""Adapter from the current FFmpeg/UI options into generic analysis requests."""
from __future__ import annotations

from pathlib import Path

from vmaf_app.core.analysis_request import (
    AnalysisRequest,
    ExecutionPreferences,
    FrameCoverage,
    MetricRequestSpec,
)
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.metrics import FRAME_METRICS
from vmaf_app.core.models import CropBox, ResampleTarget, VmafOptions, clone_options


def _copy_crop(crop: CropBox | None) -> CropBox | None:
    return None if crop is None else CropBox(crop.w, crop.h, crop.x, crop.y)


def _copy_resample(target: ResampleTarget | None) -> ResampleTarget | None:
    return None if target is None else ResampleTarget(target.width, target.label)


def comparison_recipe_from_vmaf_options(options: VmafOptions) -> ComparisonRecipe:
    """Snapshot the common scientific preprocessing represented by the UI row."""
    return ComparisonRecipe(
        crop_mode=options.crop_mode,
        manual_source_crop=_copy_crop(options.manual_source_crop),
        manual_distorted_crop=_copy_crop(options.manual_distorted_crop),
        scale_algorithm=options.scale_algorithm,
        scale_direction=options.scale_direction,
        duration_limit=options.duration_limit,
        resample_test=_copy_resample(options.resample_test),
    )


def metric_request_specs(
    options: VmafOptions,
    metric_keys: tuple[str, ...] | None = None,
) -> tuple[MetricRequestSpec, ...]:
    """Translate current FFmpeg metric choices into backend-neutral specs."""
    requested = options.requested_metrics() if metric_keys is None else metric_keys
    libvmaf_requested = any(key != "xpsnr" for key in requested)
    specs: list[MetricRequestSpec] = []
    for key in requested:
        coverage = FrameCoverage(
            "sampled" if libvmaf_requested and options.n_subsample > 1 else "full",
            options.n_subsample if libvmaf_requested and options.n_subsample > 1 else 1,
        )
        parameters: tuple[tuple[str, object], ...] = ()
        if key == "vmaf":
            model = options.model or options.model_choice
            custom_identity = ""
            if options.custom_model_path:
                path = Path(options.custom_model_path).resolve()
                try:
                    stat = path.stat()
                    custom_identity = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
                except OSError:
                    custom_identity = str(path)
            parameters = (
                ("model", model),
                ("model_choice", options.model_choice),
                ("custom_model", custom_identity),
            )
        elif key == "vmaf_neg":
            parameters = (("model", "version=vmaf_v0.6.1neg"),)
        compatibility = "ffmpeg-xpsnr-v1" if key == "xpsnr" else "ffmpeg-libvmaf-v1"
        spec = MetricRequestSpec(
            key=key,
            backend_id="ffmpeg",
            parameters=parameters,
            coverage=coverage,
            implementation_compatibility_id=compatibility,
        )
        specs.append(spec)
    return tuple(specs)


def analysis_request_from_vmaf_options(options: VmafOptions) -> AnalysisRequest:
    """Freeze one mutable UI/backend option object into a generic request."""
    return AnalysisRequest(
        recipe=comparison_recipe_from_vmaf_options(options),
        metrics=metric_request_specs(options),
        execution=ExecutionPreferences(
            gpu_decode=options.gpu_decode,
            gpu_vendor=options.gpu_vendor,
            n_threads=options.n_threads,
        ),
    )


def supplemental_metric_specs(options: VmafOptions) -> tuple[MetricRequestSpec, ...]:
    """Compatible current FFmpeg metrics worth probing in addition to a row's request.

    The UI shows already-cached companion scores when their scientific recipe
    matches. XPSNR-only requests stay isolated because their full-frame coverage
    differs from mixed libvmaf runs when subsampling is enabled.
    """
    if not any(key != "xpsnr" for key in options.requested_metrics()):
        return ()
    fuller = clone_options(options)
    for metric in FRAME_METRICS:
        if metric.ffmpeg_binding is not None:
            fuller.set_metric_enabled(metric.key, True)
    return metric_request_specs(fuller)

"""Adapter from the current FFmpeg/UI options into generic analysis requests."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from vmaf_app.core.analysis_request import (
    AnalysisRequest,
    ExecutionPreferences,
    FrameCoverage,
    MetricRequestSpec,
)
from vmaf_app.core.comparison_recipe import ComparisonRecipe
from vmaf_app.core.cvvdp import CvvdpSettings
from vmaf_app.core.metrics import FRAME_METRICS, METRICS, metric_definition
from vmaf_app.core.model_select import AUTO_MODEL_CHOICE, CUSTOM_MODEL_CHOICE
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
    cvvdp: CvvdpSettings | None = None,
) -> tuple[MetricRequestSpec, ...]:
    """Translate current FFmpeg metric choices into backend-neutral specs."""
    requested = options.requested_metrics() if metric_keys is None else metric_keys
    ffmpeg_requested = tuple(key for key in requested if metric_definition(key).ffmpeg_binding is not None)
    libvmaf_requested = any(key != "xpsnr" for key in ffmpeg_requested)
    specs: list[MetricRequestSpec] = []
    for key in requested:
        definition = metric_definition(key)
        if definition.ffmpeg_binding is None:
            if definition.backend_id is None:
                raise ValueError(f"No executable backend is registered for metric {key!r}")
            if key == "cvvdp":
                # GPU only (Vship), and always every frame: CVVDP models
                # motion over time, so it is not offered with subsampling.
                # The display and resize setting are part of what the score
                # means, so they are part of its identity.
                specs.append(MetricRequestSpec(
                    key=key, backend_id=definition.backend_id,
                    parameters=(cvvdp or CvvdpSettings()).spec_parameters(),
                    coverage=FrameCoverage("full", 1),
                    implementation_compatibility_id="cvvdp-vship-gpu-v1",
                ))
                continue
            specs.append(MetricRequestSpec(
                key=key,
                backend_id=definition.backend_id,
                parameters=(),
                coverage=FrameCoverage("sampled" if options.n_subsample > 1 else "full", options.n_subsample),
                implementation_compatibility_id=f"{key}-auto-or-libjxl-cpu-v1",
            ))
            continue
        coverage = FrameCoverage(
            "sampled" if libvmaf_requested and options.n_subsample > 1 else "full",
            options.n_subsample if libvmaf_requested and options.n_subsample > 1 else 1,
        )
        parameters: tuple[tuple[str, object], ...] = ()
        if key == "vmaf":
            # Only what decides the score. An explicit or bundled choice is
            # the model that runs. Auto's model depends on the size compared
            # at, known for certain only once black bars are detected, and is
            # recorded in the score's provenance instead. This used to be
            # `options.model`, a field that is not the model on a row set to
            # Auto -- empty, a leftover default or a loaded result's -- so the
            # same row was saved and looked up under different keys.
            choice = options.model_choice
            model = "" if choice in (AUTO_MODEL_CHOICE, CUSTOM_MODEL_CHOICE) else choice
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


def analysis_request_from_vmaf_options(
    options: VmafOptions, metric_keys: tuple[str, ...] | None = None,
    perceptual_backends: Mapping[str, str] | None = None,
    cvvdp: CvvdpSettings | None = None,
) -> AnalysisRequest:
    """Freeze one mutable UI/backend option object into a generic request."""
    backend_choices = {"ssimulacra2": "gpu", "butteraugli": "gpu"}
    if perceptual_backends is not None:
        for key, backend in perceptual_backends.items():
            if key not in backend_choices:
                raise ValueError(f"Unknown perceptual metric backend setting: {key!r}")
            if backend not in {"gpu", "cpu"}:
                raise ValueError(f"Unsupported {key} compute backend: {backend!r}")
            backend_choices[key] = backend
    return AnalysisRequest(
        recipe=comparison_recipe_from_vmaf_options(options),
        metrics=metric_request_specs(options, metric_keys, cvvdp),
        execution=ExecutionPreferences(
            gpu_decode=options.gpu_decode,
            gpu_vendor=options.gpu_vendor,
            n_threads=options.n_threads,
            perceptual_backends=tuple(sorted(backend_choices.items())),
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


def displayable_metric_specs(
    options: VmafOptions, cvvdp: CvvdpSettings | None = None,
) -> tuple[MetricRequestSpec, ...]:
    """Every saved score a row can show for its recipe, ticked or not.

    A row shows a saved score whether or not its metric is ticked: the tick
    says what the next run calculates, not what may be displayed. So a
    lookup probes every metric, not only the ticked ones:
    - the FFmpeg metrics as a combined libvmaf run stores them, and XPSNR
      as an XPSNR-only run stores it (full coverage even when libvmaf is
      subsampled);
    - SSIMULACRA2 and Butteraugli at the row's coverage;
    - CVVDP for the row's display.
    Only for finding scores: "Recalculate" still clears just the selected
    metrics, and runs still calculate just the ticked ones.
    """
    fuller = clone_options(options)
    for metric in FRAME_METRICS:
        if metric.ffmpeg_binding is not None:
            fuller.set_metric_enabled(metric.key, True)
    xpsnr_only = clone_options(options)
    for metric in FRAME_METRICS:
        if metric.ffmpeg_binding is not None:
            xpsnr_only.set_metric_enabled(metric.key, metric.key == "xpsnr")
    perceptual = tuple(metric.key for metric in METRICS if metric.backend_id == "perceptual")
    specs: dict[str, MetricRequestSpec] = {}
    for spec in (*metric_request_specs(fuller), *metric_request_specs(xpsnr_only, ("xpsnr",)),
                 *metric_request_specs(options, perceptual, cvvdp)):
        specs.setdefault(repr(spec.identity_dict()), spec)
    return tuple(specs.values())

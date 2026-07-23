"""Speaker diarization via pyannote.audio (optional dependency)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

from local_whisper_transcribe.audio import load_audio_waveform
from local_whisper_transcribe.transcribe import Segment

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-3.1"
# pyannote.audio 4.x also downloads PLDA assets from community-1 when loading 3.1.
HF_MODELS_TO_ACCEPT = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "pyannote/speaker-diarization-community-1",
)

_ACCESS_ERROR_MARKERS = (
    "403",
    "gated",
    "authorized",
    "restricted",
    "could not download",
    "private or gated",
    "not in the authorized",
    "cannot access",
)


class DiarizationNotInstalledError(RuntimeError):
    """Raised when pyannote.audio is not installed."""


class DiarizationTokenError(RuntimeError):
    """Raised when a HuggingFace token is required but missing."""


class DiarizationAccessError(RuntimeError):
    """Raised when HuggingFace model access is denied."""


def _suppress_torchcodec_warning() -> None:
    warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)


def _patch_speechbrain_windows_inspect() -> None:
    """Work around SpeechBrain LazyModule breaking on Windows.

    SpeechBrain only skips lazy imports when the caller path ends with
    ``/inspect.py``. On Windows the path uses backslashes, so ``inspect.stack()``
    (used by Lightning while loading checkpoints) accidentally triggers a real
    import of optional ``k2`` and fails.
    """
    try:
        from speechbrain.utils.importutils import LazyModule
    except ImportError:
        return

    if getattr(LazyModule.ensure_module, "_lwt_windows_patch", False):
        return

    import importlib
    import inspect
    import sys

    def ensure_module(self, stacklevel: int):  # type: ignore[no-untyped-def]
        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except Exception:
            pass

        if importer_frame is not None:
            filename = importer_frame.filename.replace("\\", "/")
            if filename.endswith("/inspect.py"):
                raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as exc:
                raise ImportError(f"Lazy import of {repr(self)} failed") from exc

        return self.lazy_module

    ensure_module._lwt_windows_patch = True  # type: ignore[attr-defined]
    LazyModule.ensure_module = ensure_module  # type: ignore[method-assign]


def _format_access_error(exc: Exception) -> str:
    message = str(exc)
    models = "\n".join(f"  - https://huggingface.co/{model}" for model in HF_MODELS_TO_ACCEPT)
    missing_hint = ""
    for model in HF_MODELS_TO_ACCEPT:
        if model in message:
            missing_hint = (
                f"\nThe error points to {model} — "
                "open the link and click Agree and access repository.\n"
            )
            break
    if "speaker-diarization-community-1" in message and "community-1" not in missing_hint:
        missing_hint = (
            "\npyannote.audio 4.x also needs speaker-diarization-community-1 "
            "(4th model — easy to miss):\n"
            "  - https://huggingface.co/pyannote/speaker-diarization-community-1\n"
        )
    return (
        "HuggingFace denied access to a diarization model.\n"
        f"{missing_hint}"
        "Accept user conditions for ALL required models:\n"
        f"{models}\n"
        "Then create a Read token at https://huggingface.co/settings/tokens and save it:\n"
        "  lwt config set diarization.hf_token <token>\n"
        "Verify before transcribing:\n"
        "  lwt install verify-diarization"
    )


def _require_pipeline():
    try:
        with warnings.catch_warnings():
            _suppress_torchcodec_warning()
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationNotInstalledError(
            "pyannote.audio is not installed.\n"
            "Install the optional diarization dependencies with:\n"
            "  lwt install diarization\n"
            f"Then accept the model license at {HF_MODEL_URL}"
        ) from exc
    return Pipeline


def _load_pipeline(hf_token: str, device: str = "cpu"):
    Pipeline = _require_pipeline()
    _patch_speechbrain_windows_inspect()
    try:
        with warnings.catch_warnings():
            _suppress_torchcodec_warning()
            warnings.filterwarnings(
                "ignore",
                message=".*speechbrain.pretrained.*",
                category=UserWarning,
            )
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
    except Exception as exc:
        message = str(exc).lower()
        if any(marker in message for marker in _ACCESS_ERROR_MARKERS):
            raise DiarizationAccessError(_format_access_error(exc)) from exc
        raise

    import torch

    from local_whisper_transcribe.cuda_runtime import configure_cuda_dll_paths

    if device == "cuda":
        configure_cuda_dll_paths()
        pipeline.to(torch.device("cuda"))
    else:
        pipeline.to(torch.device("cpu"))
    return pipeline


def _annotation_from_output(output: Any):
    """Normalize pyannote 3.x Annotation / 4.x DiarizeOutput to Annotation."""
    if hasattr(output, "speaker_diarization"):
        return output.speaker_diarization
    return output


def verify_hf_model_access(model_id: str, hf_token: str) -> tuple[bool, str]:
    """Return (ok, detail) for a single HuggingFace model."""
    try:
        from huggingface_hub import HfApi, hf_hub_download

        HfApi().model_info(model_id, token=hf_token)

        if model_id == "pyannote/speaker-diarization-community-1":
            hf_hub_download(
                model_id,
                "xvec_transform.npz",
                subfolder="plda",
                token=hf_token,
            )

        return True, "access OK"
    except Exception as exc:
        message = str(exc).lower()
        if any(marker in message for marker in _ACCESS_ERROR_MARKERS):
            return False, "license not accepted or token lacks access"
        return False, str(exc)


def verify_diarization_access(hf_token: str) -> list[tuple[str, bool, str]]:
    """Check HuggingFace access for all diarization models."""
    results: list[tuple[str, bool, str]] = []
    for model_id in HF_MODELS_TO_ACCEPT:
        ok, detail = verify_hf_model_access(model_id, hf_token)
        results.append((model_id, ok, detail))
    return results


def _segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


# Overall progress weights for pyannote pipeline stages.
_DIARIZE_STEP_WEIGHTS: dict[str, float] = {
    "load": 0.05,
    "prepare": 0.05,
    "segmentation": 0.40,
    "embeddings": 0.40,
    "finalize": 0.10,
}

_DIARIZE_STEP_LABELS: dict[str, str] = {
    "load": "Loading model",
    "prepare": "Preparing audio",
    "segmentation": "Segmentation",
    "speaker_counting": "Counting speakers",
    "embeddings": "Speaker embeddings",
    "discrete_diarization": "Clustering",
    "finalize": "Finishing",
}


def _overall_progress(
    step: str,
    *,
    completed: float | None = None,
    total: float | None = None,
) -> tuple[float, str]:
    """Map a pyannote step to overall 0–100 progress and a short label."""
    label = _DIARIZE_STEP_LABELS.get(step, step.replace("_", " ").title())

    # Cumulative start offsets for the main weighted stages.
    load_w = _DIARIZE_STEP_WEIGHTS["load"]
    prep_w = _DIARIZE_STEP_WEIGHTS["prepare"]
    seg_w = _DIARIZE_STEP_WEIGHTS["segmentation"]
    emb_w = _DIARIZE_STEP_WEIGHTS["embeddings"]
    fin_w = _DIARIZE_STEP_WEIGHTS["finalize"]

    starts = {
        "load": 0.0,
        "prepare": load_w,
        "segmentation": load_w + prep_w,
        "speaker_counting": load_w + prep_w + seg_w,
        "embeddings": load_w + prep_w + seg_w,
        "discrete_diarization": load_w + prep_w + seg_w + emb_w,
        "finalize": load_w + prep_w + seg_w + emb_w,
    }
    weights = {
        "load": load_w,
        "prepare": prep_w,
        "segmentation": seg_w,
        "speaker_counting": 0.0,
        "embeddings": emb_w,
        "discrete_diarization": 0.0,
        "finalize": fin_w,
    }

    start = starts.get(step, load_w + prep_w)
    weight = weights.get(step, 0.0)

    if completed is not None and total and total > 0 and weight > 0:
        fraction = min(1.0, max(0.0, float(completed) / float(total)))
        pct = int(fraction * 100)
        return (start + weight * fraction) * 100.0, f"{label} ({pct}%)"

    # Step finished (no intra-step progress) → advance to end of this stage.
    return (start + weight) * 100.0, label


def diarize_audio(
    audio_path: Path,
    hf_token: str | None = None,
    *,
    device: str = "cpu",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_callback: Callable[[float, float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run speaker diarization and return segments with start, end, and speaker labels.

    ``progress_callback(completed, total, description)`` receives overall progress
    where ``total`` is always 100 (percent-style), matching the Whisper UI.
    """
    if not hf_token:
        raise DiarizationTokenError(
            "A HuggingFace token is required for speaker diarization.\n"
            f"1. Accept the model license at {HF_MODEL_URL}\n"
            "2. Create a token at https://huggingface.co/settings/tokens\n"
            "3. Set it via --hf-token, HF_TOKEN env var, or:\n"
            "     lwt config set diarization.hf_token <token>"
        )

    resolved_device = "cuda" if device == "cuda" else "cpu"

    def report(step: str, *, completed: float | None = None, total: float | None = None) -> None:
        if not progress_callback:
            return
        overall, label = _overall_progress(step, completed=completed, total=total)
        progress_callback(overall, 100.0, label)

    report("load", completed=0, total=1)

    with warnings.catch_warnings():
        _suppress_torchcodec_warning()
        pipeline = _load_pipeline(hf_token, device=resolved_device)

    report("load", completed=1, total=1)

    kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    report("prepare", completed=0, total=1)
    audio_input = load_audio_waveform(Path(audio_path))
    report("prepare", completed=1, total=1)

    def hook(step_name: str, _artifact: Any = None, **hook_kwargs: Any) -> None:
        # setup_hook injects file=...; Inference also passes completed/total.
        completed = hook_kwargs.get("completed")
        total = hook_kwargs.get("total")
        report(str(step_name), completed=completed, total=total)

    output = pipeline(audio_input, hook=hook, **kwargs)
    annotation = _annotation_from_output(output)

    segments: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    report("finalize", completed=1, total=1)
    return segments


def merge_transcription_with_diarization(
    whisper_segments: list[Segment],
    diarization_segments: list[dict[str, Any]],
) -> list[Segment]:
    """Assign each Whisper segment to the diarization speaker with maximum overlap."""
    merged: list[Segment] = []

    for whisper_seg in whisper_segments:
        best_speaker: str | None = None
        best_overlap = 0.0

        for dia_seg in diarization_segments:
            overlap = _segment_overlap(
                whisper_seg.start,
                whisper_seg.end,
                dia_seg["start"],
                dia_seg["end"],
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dia_seg["speaker"]

        merged.append(
            Segment(
                start=whisper_seg.start,
                end=whisper_seg.end,
                text=whisper_seg.text,
                speaker=best_speaker,
            )
        )

    return merged


def apply_speaker_names(segments: list[Segment], speaker_names: list[str]) -> list[Segment]:
    """Rename SPEAKER_00, SPEAKER_01, ... to user-provided display names."""
    if not speaker_names:
        return segments

    mapping = {f"SPEAKER_{index:02d}": name for index, name in enumerate(speaker_names)}
    renamed: list[Segment] = []
    for seg in segments:
        speaker = seg.speaker
        if speaker and speaker in mapping:
            speaker = mapping[speaker]
        renamed.append(
            Segment(start=seg.start, end=seg.end, text=seg.text, speaker=speaker)
        )
    return renamed

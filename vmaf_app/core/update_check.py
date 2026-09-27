"""Is there a newer release? Asked once, when the app starts.

The releases are published on GitHub, and GitHub's public API names the
latest one -- drafts and pre-releases left out -- so no server of our own is
needed. No key either: GitHub allows 60 such requests an hour from one
address. Nothing is sent but the request itself; GitHub sees the address it
comes from, as it would for any page.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

RELEASES_URL = "https://api.github.com/repos/4KVCD/VideoMetricsLab/releases/latest"
TIMEOUT_SECONDS = 8.0


class UpdateCheckError(RuntimeError):
    """GitHub could not be asked, or its answer could not be read."""


@dataclass(frozen=True)
class Release:
    version: str  # "1.3", from the tag "v1.3"
    page_url: str
    notes: str


def version_tuple(text: str) -> tuple[int, ...] | None:
    """ "v1.2.1" -> (1, 2, 1); trailing zeros dropped, so 1.3 and 1.3.0 are
    the same version. None for anything that is not a plain version."""
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip())
    if match is None:
        return None
    parts = [int(part) for part in match.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def is_newer(candidate: str, current: str) -> bool:
    """Compared as numbers: 1.10 is newer than 1.9."""
    new, old = version_tuple(candidate), version_tuple(current)
    return new is not None and old is not None and new > old


def latest_release(opener=urllib.request.urlopen, timeout: float = TIMEOUT_SECONDS) -> Release:
    from vmaf_app import APP_NAME, __version__

    request = urllib.request.Request(RELEASES_URL, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"{APP_NAME}/{__version__}",  # GitHub refuses requests without one
    })
    try:
        with opener(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as error:  # network, HTTP and JSON errors alike
        raise UpdateCheckError(str(error)) from error
    tag = str(data.get("tag_name") or "") if isinstance(data, dict) else ""
    if version_tuple(tag) is None:
        raise UpdateCheckError(f"GitHub named no release version (tag {tag!r})")
    return Release(
        version=tag.strip().removeprefix("v"),
        page_url=str(data.get("html_url") or "https://github.com/4KVCD/VideoMetricsLab/releases/latest"),
        notes=str(data.get("body") or ""),
    )

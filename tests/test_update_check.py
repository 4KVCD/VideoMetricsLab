import json
import urllib.error

import pytest

from vmaf_app.core import update_check


@pytest.mark.parametrize(("candidate", "current", "newer"), [
    ("v1.3", "1.2.1", True),
    ("v1.10", "1.9", True),  # as numbers, not text
    ("v1.2.1", "1.2.1", False),
    ("v1.3.0", "1.3", False),  # the same version
    ("v1.2", "1.3", False),
    ("nightly", "1.3", False),
])
def test_versions_are_compared_as_numbers(candidate, current, newer):
    assert update_check.is_newer(candidate, current) is newer


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_the_latest_release_is_read_from_github():
    seen = []

    def opener(request, timeout):
        seen.append((request.full_url, dict(request.header_items()), timeout))
        return _Response(json.dumps({"tag_name": "v1.4", "html_url": "https://github.com/x/releases/tag/v1.4",
                                     "body": "- Added a thing."}).encode())

    release = update_check.latest_release(opener=opener)
    assert release == update_check.Release("1.4", "https://github.com/x/releases/tag/v1.4", "- Added a thing.")
    url, headers, timeout = seen[0]
    assert url == "https://api.github.com/repos/4KVCD/VideoMetricsLab/releases/latest"
    assert headers["User-agent"].startswith("VideoMetricsLab/") and timeout > 0


@pytest.mark.parametrize("failure", [
    urllib.error.URLError("no network"),
    TimeoutError("timed out"),
])
def test_an_unreachable_github_is_an_update_check_error(failure):
    def opener(request, timeout):
        raise failure

    with pytest.raises(update_check.UpdateCheckError):
        update_check.latest_release(opener=opener)


@pytest.mark.parametrize("body", [b"not json", b'{"message": "Not Found"}', b"[]"])
def test_an_unreadable_answer_is_an_update_check_error(body):
    with pytest.raises(update_check.UpdateCheckError):
        update_check.latest_release(opener=lambda request, timeout: _Response(body))

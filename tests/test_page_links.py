import pytest

from what_to_id.page_links import page_url


@pytest.mark.parametrize("target", ["map.html", "../map.html", "https://example.org/map.html"])
def test_valid_navigation_targets(target):
    assert page_url(target) == target


@pytest.mark.parametrize(
    "target",
    [
        "",
        "javascript:alert(1)",
        "data:text/html,x",
        "//evil.example/a",
        "https://user:pass@example.org/map",
        "https:///missing-host",
        r"..\map.html",
        "map\n.html",
    ],
)
def test_reject_executable_or_ambiguous_navigation(target):
    with pytest.raises(ValueError, match="relative path or HTTPS"):
        page_url(target)

"""Validate configured navigation targets before publishing static pages."""

from urllib.parse import urlsplit


def page_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        not value
        or value.startswith("//")
        or "\\" in value
        or any(ord(c) < 33 for c in value)
        or parsed.scheme not in ("", "https")
        or (parsed.scheme and not parsed.netloc)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("page URL must be a relative path or HTTPS URL without credentials")
    return value

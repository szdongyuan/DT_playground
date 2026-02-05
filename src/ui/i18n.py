"""
i18n utilities (gettext-based).

Design goals:
- Keep callsites simple: use tr_("Text") in UI code.
- Support language switching via config and app restart.
- Avoid hard dependency on GNU tools at runtime (only .mo files needed).
"""

from __future__ import annotations

import gettext
from pathlib import Path
from typing import Optional, Union

_translations: gettext.NullTranslations = gettext.NullTranslations()


PathLike = Union[str, Path]


def install(
    language: str,
    *,
    domain: str = "messages",
    locale_dir: Optional[PathLike] = None,
) -> gettext.NullTranslations:
    """
    Install translations for the given language.

    This does not inject builtins; callsites should import `tr_` from this module.
    """
    global _translations

    if locale_dir is None:
        # src/ui/i18n.py -> src/locale
        locale_dir_path = Path(__file__).resolve().parents[1] / "locale"
    else:
        locale_dir_path = Path(locale_dir).expanduser().resolve()

    # `fallback=True` ensures we never crash if .mo is missing; msgid is returned.
    _translations = gettext.translation(
        domain=domain,
        localedir=str(locale_dir_path),
        languages=[language],
        fallback=True,
    )
    return _translations


def tr_(message: str) -> str:
    """Translate a single message."""
    try:
        return _translations.gettext(message)
    except Exception:
        return message


def ngettext(singular: str, plural: str, n: int) -> str:
    """Plural translation helper."""
    try:
        return _translations.ngettext(singular, plural, n)
    except Exception:
        return singular if n == 1 else plural


def pgettext(context: str, message: str) -> str:
    """
    Contextual translation helper.

    Uses the conventional msgctxt encoding: "context\\x04message".
    """
    combined = f"{context}\x04{message}"
    translated = tr_(combined)
    return message if translated == combined else translated


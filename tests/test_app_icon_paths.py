import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.ui import startup_splash


def test_get_app_icon_path_in_dev_mode(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    expected_path = project_root / "assets" / "DTPG_logo.ico"

    monkeypatch.setattr(startup_splash, "__file__", str(project_root / "src" / "ui" / "startup_splash.py"))
    monkeypatch.setattr(startup_splash.sys, "frozen", False, raising=False)

    assert startup_splash.get_app_icon_path() == expected_path


def test_get_app_icon_path_in_frozen_mode(monkeypatch):
    exe_path = Path("D:/dist/AudioTrainingPlatform/AudioTrainingPlatform.exe")
    expected_path = exe_path.parent / "assets" / "DTPG_logo.ico"

    monkeypatch.setattr(startup_splash.sys, "frozen", True, raising=False)
    monkeypatch.setattr(startup_splash.sys, "executable", str(exe_path))

    assert startup_splash.get_app_icon_path() == expected_path

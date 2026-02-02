"""
Restart Manager

Implements a user-triggered, graceful restart flow:
- (If busy) request stop and wait up to stop_timeout_s
- Spawn a new process with a restart token + ready file path
- Wait up to ready_timeout_s for the new process to signal readiness (main window shown)
- On success: gracefully shutdown and quit
- On failure: keep the current app running and report an error
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtWidgets import QApplication

from src.utils.config import config


@dataclass(frozen=True)
class RestartSpec:
    token: str
    ready_file: str


class RestartManager(QObject):
    def __init__(self, parent: QObject, app_window: object):
        super().__init__(parent)
        self._app_window = app_window

        self._stop_timer: Optional[QTimer] = None
        self._ready_timer: Optional[QTimer] = None

        self._stop_deadline: float = 0.0
        self._ready_deadline: float = 0.0
        self._ready_timeout_s: float = 10.0
        # Hard guard to prevent spawning multiple instances.
        self._spawn_started: bool = False

        self._spec: Optional[RestartSpec] = None
        self._on_finished: Optional[Callable[[bool, str], None]] = None

        self._stop_requested = False

    def restart(
        self,
        *,
        stop_timeout_s: float,
        ready_timeout_s: float,
        on_finished: Callable[[bool, str], None],
    ) -> None:
        self._on_finished = on_finished
        self._ready_timeout_s = float(ready_timeout_s)

        # Phase 1: stop if busy
        if self._is_busy():
            self._stop_deadline = time.monotonic() + float(stop_timeout_s)
            self._request_stop()
            self._start_stop_poll()
            return

        # Phase 2: spawn + wait for ready
        self._spawn_started = True
        self._start_spawn_and_wait(ready_timeout_s=self._ready_timeout_s)

    def _finish(self, success: bool, message: str) -> None:
        if self._stop_timer:
            self._stop_timer.stop()
            self._stop_timer.deleteLater()
            self._stop_timer = None
        if self._ready_timer:
            self._ready_timer.stop()
            self._ready_timer.deleteLater()
            self._ready_timer = None

        cb = self._on_finished
        self._on_finished = None
        if cb:
            cb(success, message)

    def _is_busy(self) -> bool:
        # Both workflow and training are considered busy.
        try:
            mw = getattr(self._app_window, "main_window", None)
            if mw is None:
                return False

            workflow_running = False
            training_running = False

            try:
                workflow_running = bool(mw._workflow_controller.is_running())
            except Exception:
                workflow_running = False

            try:
                training_running = bool(mw._training_controller.is_training)
            except Exception:
                training_running = False

            return workflow_running or training_running
        except Exception:
            return False

    def _request_stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True

        try:
            mw = getattr(self._app_window, "main_window", None)
            if mw is None:
                return
            try:
                mw._workflow_controller.stop()
            except Exception:
                pass
            try:
                mw._training_controller.stop_training()
            except Exception:
                pass
        except Exception:
            pass

    def _start_stop_poll(self) -> None:
        if self._stop_timer is not None:
            return
        self._stop_timer = QTimer(self)
        self._stop_timer.setInterval(100)
        self._stop_timer.timeout.connect(self._on_stop_poll)
        self._stop_timer.start()

    def _on_stop_poll(self) -> None:
        # If we've already moved to spawn phase, do nothing.
        if self._spawn_started:
            return

        if not self._is_busy():
            # proceed to phase 2
            self._spawn_started = True
            if self._stop_timer is not None:
                self._stop_timer.stop()
            self._start_spawn_and_wait(ready_timeout_s=self._ready_timeout_s)
            return

        if time.monotonic() >= self._stop_deadline:
            self._finish(
                False,
                "停止当前工作流/训练超时（10 秒）。已取消重启，应用将继续运行。",
            )

    def _start_spawn_and_wait(self, *, ready_timeout_s: float) -> None:
        # Defensive: never spawn twice
        if self._ready_timer is not None or self._spec is not None:
            return

        self._spec = self._build_restart_spec()

        ok, err = self._spawn_new_process(self._spec)
        if not ok:
            self._finish(False, f"无法启动新进程：{err}")
            return

        self._ready_deadline = time.monotonic() + float(ready_timeout_s)
        self._start_ready_poll()

    def _start_ready_poll(self) -> None:
        self._ready_timer = QTimer(self)
        self._ready_timer.setInterval(100)
        self._ready_timer.timeout.connect(self._on_ready_poll)
        self._ready_timer.start()

    def _on_ready_poll(self) -> None:
        assert self._spec is not None
        path = self._spec.ready_file

        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("token") == self._spec.token:
                    # Best effort cleanup
                    try:
                        os.remove(path)
                    except Exception:
                        pass

                    self._graceful_shutdown_and_quit()
                    return
        except Exception:
            # Ignore parse/race errors; keep waiting until timeout.
            pass

        if time.monotonic() >= self._ready_deadline:
            self._finish(
                False,
                "新进程未在 10 秒内完成启动（未收到 ready 信号）。已取消重启，应用将继续运行。",
            )

    def _build_restart_spec(self) -> RestartSpec:
        token = str(uuid.uuid4())
        ready_file = os.path.join(
            tempfile.gettempdir(),
            f"dt_playground_restart_ready_{token}.json",
        )
        return RestartSpec(token=token, ready_file=ready_file)

    def _spawn_new_process(self, spec: RestartSpec) -> tuple[bool, str]:
        try:
            cmd, cwd = self._build_restart_command(spec)

            creationflags = 0
            if os.name == "nt":
                # Detach so the new instance survives when the old exits.
                creationflags |= subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

            subprocess.Popen(
                cmd,
                cwd=cwd,
                close_fds=False,
                creationflags=creationflags,
            )
            return True, ""
        except Exception as e:
            return False, str(e)

    def _build_restart_command(self, spec: RestartSpec) -> tuple[list[str], str]:
        # Use a stable cwd to keep relative paths/log paths predictable.
        if getattr(sys, "frozen", False):
            project_root = Path(sys.executable).resolve().parent
        else:
            project_root = Path(__file__).resolve().parents[2]

        args = [
            "--restart-token",
            spec.token,
            "--restart-ready-file",
            spec.ready_file,
        ]

        if getattr(sys, "frozen", False):
            # PyInstaller packaged app: relaunch the executable itself.
            cmd = [sys.executable, *args]
            return cmd, str(project_root)

        # Dev mode: python main.py ...
        main_py = project_root / "main.py"
        cmd = [sys.executable, str(main_py), *args]
        return cmd, str(project_root)

    def _graceful_shutdown_and_quit(self) -> None:
        # Save config/state best-effort
        try:
            config.save()
        except Exception:
            pass

        # Stop timers/bgtasks best-effort
        try:
            mw = getattr(self._app_window, "main_window", None)
            if mw is not None:
                try:
                    if getattr(mw, "resource_timer", None) is not None:
                        mw.resource_timer.stop()
                except Exception:
                    pass
                try:
                    mw._workflow_controller.stop()
                except Exception:
                    pass
                try:
                    mw._training_controller.stop_training()
                except Exception:
                    pass
        except Exception:
            pass

        # Close main window and quit Qt.
        try:
            if hasattr(self._app_window, "close"):
                self._app_window.close()
        except Exception:
            pass

        QTimer.singleShot(0, QApplication.quit)


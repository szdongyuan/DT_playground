"""Regression tests for workflow worker lifecycle handling."""

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QApplication

from src.workflow.engine import EngineState, ExecutionResult, WorkflowEngine
from src.workflow.node_base import BaseNode
from src.workflow.workflow import Workflow


def _application() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout: float = 2.0) -> None:
    app = _application()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    app.processEvents()
    assert predicate(), "Timed out waiting for the workflow lifecycle condition"


class _ResultCollector(QObject):
    def __init__(self, engine: WorkflowEngine):
        super().__init__()
        self.engine = engine
        self.results: list[ExecutionResult] = []
        self.snapshots: list[tuple[EngineState, object]] = []

    @Slot(object)
    def collect(self, result: ExecutionResult) -> None:
        self.snapshots.append((self.engine.state, self.engine._worker))
        self.results.append(result)


class _RequiredParameterNode(BaseNode):
    node_type = "required_parameter_test"
    display_name = "Required Parameter Test"

    def _setup_ports(self):
        pass

    def _setup_parameters(self):
        self.add_parameter("required_value", "float", None)

    def execute(self) -> bool:
        return True


class _FailOnceNode(BaseNode):
    node_type = "fail_once_test"
    display_name = "Fail Once Test"

    def __init__(self, node_id=None):
        self.call_count = 0
        super().__init__(node_id)

    def _setup_ports(self):
        pass

    def execute(self) -> bool:
        self.call_count += 1
        if self.call_count == 1:
            self.error_message = "Expected first-run failure"
            return False
        return True


class _CooperativeStopNode(BaseNode):
    node_type = "cooperative_stop_test"
    display_name = "Cooperative Stop Test"

    def __init__(self, node_id=None):
        self.started = threading.Event()
        self.stop_requested = threading.Event()
        super().__init__(node_id)

    def _setup_ports(self):
        pass

    def request_stop(self) -> None:
        self.stop_requested.set()

    def execute(self) -> bool:
        self.started.set()
        if self.stop_requested.wait(0.3):
            self.error_message = "Stopped"
            return False
        return True


class _RaiseNode(BaseNode):
    node_type = "raise_test"
    display_name = "Raise Test"

    def _setup_ports(self):
        pass

    def execute(self) -> bool:
        raise RuntimeError("Expected node exception")


def _workflow_with(node: BaseNode) -> Workflow:
    workflow = Workflow("Lifecycle Test")
    assert workflow.add_node(node)
    return workflow


def test_validation_failure_does_not_create_worker_and_can_retry():
    node = _RequiredParameterNode("required")
    engine = WorkflowEngine()
    engine.set_workflow(_workflow_with(node))

    assert engine.execute() is False
    assert engine.state == EngineState.IDLE
    assert engine._worker is None

    node.set_parameter("required_value", 1.0)
    collector = _ResultCollector(engine)
    engine.workflow_finished.connect(collector.collect)

    assert engine.execute() is True
    _wait_until(lambda: len(collector.results) == 1)

    assert collector.results[0].success is True
    assert collector.snapshots == [(EngineState.IDLE, None)]


def test_terminal_signal_is_emitted_after_worker_cleanup():
    engine = WorkflowEngine()
    engine.set_workflow(_workflow_with(_FailOnceNode("fail")))
    collector = _ResultCollector(engine)
    engine.workflow_finished.connect(collector.collect)

    assert engine.execute() is True
    _wait_until(lambda: len(collector.results) == 1)

    assert collector.results[0].success is False
    assert collector.snapshots == [(EngineState.IDLE, None)]


def test_node_exception_finishes_with_clean_idle_state():
    engine = WorkflowEngine()
    engine.set_workflow(_workflow_with(_RaiseNode("raise")))
    collector = _ResultCollector(engine)
    engine.workflow_finished.connect(collector.collect)

    assert engine.execute() is True
    _wait_until(lambda: len(collector.results) == 1)

    assert collector.results[0].success is False
    assert "Expected node exception" in collector.results[0].message
    assert collector.snapshots == [(EngineState.IDLE, None)]


def test_synchronous_execution_finalizes_before_signal():
    engine = WorkflowEngine()
    engine.set_workflow(_workflow_with(_FailOnceNode("sync")))
    collector = _ResultCollector(engine)
    engine.workflow_finished.connect(collector.collect)

    result = engine.execute_sync()

    assert result.success is False
    assert collector.results == [result]
    assert collector.snapshots == [(EngineState.IDLE, None)]


def test_immediate_retry_from_terminal_handler_keeps_new_worker_owned():
    node = _FailOnceNode("retry")
    engine = WorkflowEngine()
    engine.set_workflow(_workflow_with(node))

    class _RetryCollector(_ResultCollector):
        def __init__(self, target_engine: WorkflowEngine):
            super().__init__(target_engine)
            self.retry_started = False
            self.retry_worker = None

        @Slot(object)
        def collect(self, result: ExecutionResult) -> None:
            super().collect(result)
            if len(self.results) == 1:
                self.retry_started = self.engine.execute()
                self.retry_worker = self.engine._worker

    collector = _RetryCollector(engine)
    engine.workflow_finished.connect(collector.collect)

    assert engine.execute() is True
    _wait_until(lambda: len(collector.results) == 2)

    assert collector.retry_started is True
    assert collector.retry_worker is not None
    assert collector.results[0].success is False
    assert collector.results[1].success is True
    assert collector.snapshots == [
        (EngineState.IDLE, None),
        (EngineState.IDLE, None),
    ]
    assert engine.state == EngineState.IDLE
    assert engine._worker is None


def test_stop_requests_active_node_and_finishes_before_reenable():
    node = _CooperativeStopNode("stop")
    engine = WorkflowEngine()
    engine.set_workflow(_workflow_with(node))
    collector = _ResultCollector(engine)
    engine.workflow_finished.connect(collector.collect)

    assert engine.execute() is True
    _wait_until(node.started.is_set)

    engine.stop()
    assert engine.state == EngineState.STOPPING
    assert node.stop_requested.is_set()

    _wait_until(lambda: len(collector.results) == 1)

    assert collector.results[0].success is False
    assert "stopped" in collector.results[0].message.lower()
    assert collector.snapshots == [(EngineState.IDLE, None)]

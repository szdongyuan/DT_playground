"""Typed anomaly file nodes shared by GUI and CLI workflows."""

from pathlib import Path

from src.ui.i18n import tr_
from ..anomaly_io import export_results, load_model, output_target, save_model
from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType
from .anomaly import AnomalyResultData, AnomalyScoresData


class _AnomalyFileNode(BaseNode):
    """Common validation and error reporting without interactive prompts."""

    def _validate_configuration(self):
        return None

    def validate(self):
        valid, message = super().validate()
        if not valid:
            return valid, message
        try:
            self._validate_configuration()
            return True, ""
        except (ValueError, OSError) as exc:
            return False, str(exc)

    def execute(self):
        for port in self.outputs.values():
            port.clear()
        self.error_message = ""
        try:
            self._validate_configuration()
            self._execute_file_operation()
            self.report_status(tr_("Anomaly file operation completed"))
            return True
        except Exception as exc:
            self.error_message = tr_("Anomaly file operation failed: {error}").format(error=str(exc))
            return False


@register_node
class SaveAnomalyModelNode(_AnomalyFileNode):
    node_type = "save_anomaly_model"
    display_name = tr_("Save anomaly model")
    category = NodeCategory.TRAINING
    subcategory = tr_("Model management")
    subcategory_order = 50
    palette_order = 40
    description = tr_("Save scoring state without overwriting; keep the decision workflow for reproduction")
    icon = "📤"

    def _setup_ports(self):
        self.add_input("anomaly_model", DataType.ANOMALY_MODEL, tr_("Anomaly model"))
        self.add_output("model_path", DataType.ANY, tr_("Saved path"))

    def _setup_parameters(self):
        self.add_parameter("output_folder", "folder", ".", display_name=tr_("Output directory"))
        self.add_parameter("file_name", "str", "anomaly_model.anomaly.zip",
                           display_name=tr_("Filename"))

    def _validate_configuration(self):
        target = output_target(self.get_parameter("output_folder"), self.get_parameter("file_name"))
        if not target.name.endswith(".anomaly.zip"):
            raise ValueError(tr_("Anomaly model filename must end with .anomaly.zip"))

    def _execute_file_operation(self):
        target = output_target(self.get_parameter("output_folder"), self.get_parameter("file_name"))
        self.set_output_data("model_path", save_model(self.get_input_data("anomaly_model"), target))


@register_node
class LoadAnomalyModelNode(_AnomalyFileNode):
    node_type = "load_anomaly_model"
    display_name = tr_("Load anomaly model")
    category = NodeCategory.TRAINING
    subcategory = tr_("Model management")
    subcategory_order = 50
    palette_order = 30
    description = tr_("Load explicitly trusted anomaly scoring state; joblib can execute code")
    icon = "📥"

    def _setup_ports(self):
        self.add_output("anomaly_model", DataType.ANOMALY_MODEL, tr_("Anomaly model"))

    def _setup_parameters(self):
        self.add_parameter("model_path", "file", "", display_name=tr_("Model path"),
                           file_filter=tr_("Anomaly models (*.anomaly.zip *.joblib);;All files (*)"))
        self.add_parameter("format", "choice", "anomaly_zip", display_name=tr_("Model format"),
                           choices=["anomaly_zip", "legacy_joblib"])
        self.add_parameter("trusted", "bool", False, display_name=tr_("Trust this model (allows code execution)"),
                           description=tr_("Saved in the workflow. Enable only for a trusted model; hashes and type checks do not make joblib safe."))

    def _validate_configuration(self):
        if self.get_parameter("trusted") is not True:
            raise PermissionError(tr_("Loading joblib can execute code; explicitly trust the model before loading"))
        value = self.get_parameter("model_path")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(tr_("Model file does not exist: {path}").format(path=value))
        if self.get_parameter("format") not in ("anomaly_zip", "legacy_joblib"):
            raise ValueError(tr_("Unsupported anomaly model format"))

    def _execute_file_operation(self):
        if self.get_parameter("format") == "legacy_joblib":
            self.report_status(tr_("Legacy joblib model: runtime version compatibility cannot be verified"))
        model = load_model(Path(self.get_parameter("model_path")).expanduser(),
                           trusted=self.get_parameter("trusted"), format=self.get_parameter("format"))
        self.set_output_data("anomaly_model", model)


@register_node
class ExportAnomalyResultsNode(_AnomalyFileNode):
    node_type = "export_anomaly_results"
    display_name = tr_("Export anomaly results")
    category = NodeCategory.OUTPUT
    subcategory = tr_("Result export")
    subcategory_order = 40
    palette_order = 10
    description = tr_("Export every aligned score or decision to CSV and JSON in a new directory")
    icon = "📄"

    def _setup_ports(self):
        self.add_input("anomaly_scores", DataType.ANOMALY_SCORES, tr_("Anomaly scores"), required=False)
        self.add_input("anomaly_result", DataType.ANOMALY_RESULT, tr_("Anomaly result"), required=False)
        self.add_output("export_summary", DataType.METRICS, tr_("Export summary"))

    def _setup_parameters(self):
        self.add_parameter("output_folder", "folder", ".", display_name=tr_("Output directory"))
        self.add_parameter("directory_name", "str", "anomaly_results", display_name=tr_("Result directory name"))
        self.add_parameter("spreadsheet_safe", "bool", False, display_name=tr_("Spreadsheet-safe sample IDs"),
                           description=tr_("Prefix every sample ID with an apostrophe; remove exactly one prefix to recover the original ID"))

    def _validate_configuration(self):
        output_target(self.get_parameter("output_folder"), self.get_parameter("directory_name"))

    def validate(self):
        valid, message = super().validate()
        if valid and sum(port.is_connected for port in self.inputs.values()) != 1:
            return False, tr_("Connect exactly one anomaly scores or result input")
        return valid, message

    def _execute_file_operation(self):
        values = [self.get_input_data(name) for name in self.inputs]
        if sum(value is not None for value in values) != 1:
            raise ValueError(tr_("Connect exactly one anomaly scores or result input"))
        if ((values[0] is not None and not isinstance(values[0], AnomalyScoresData))
                or (values[1] is not None and not isinstance(values[1], AnomalyResultData))):
            raise TypeError(tr_("Export requires anomaly scores or an anomaly result"))
        target = output_target(self.get_parameter("output_folder"), self.get_parameter("directory_name"))
        summary = export_results(next(value for value in values if value is not None), target,
                                 spreadsheet_safe=self.get_parameter("spreadsheet_safe"))
        self.set_output_data("export_summary", summary)

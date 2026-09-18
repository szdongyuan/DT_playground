"""Fitted preprocessing for temporal features and vector matrices."""

from src.ui.i18n import tr_
from ..feature_contract import FeatureStandardizationState, standardize
from ..node_base import BaseNode, NodeCategory, register_node
from ..port import DataType


@register_node
class FeatureStandardizationNode(BaseNode):
    node_type = "feature_standardization"
    display_name = tr_("Feature standardization")
    category = NodeCategory.FEATURE
    subcategory = tr_("Feature post-processing")
    subcategory_order = 40
    palette_order = 20
    description = tr_("Fit normalization on training features or apply saved statistics")
    icon = "📏"

    def _setup_ports(self):
        self.add_input("features", DataType.ANY, tr_("Features"))
        self.add_input("state", DataType.ANY, tr_("Preprocessing state"), required=False)
        self.add_output("features", DataType.ANY, tr_("Features"))
        self.add_output("state", DataType.ANY, tr_("Preprocessing state"))

    def _setup_parameters(self):
        self.add_parameter("mode", "choice", "fit_transform", display_name=tr_("Mode"),
                           choices=["fit_transform", "transform"])
        self.add_parameter("epsilon", "float", 1e-6, display_name=tr_("Minimum scale"), min_value=1e-12)

    def validate(self):
        valid, message = super().validate()
        if valid and self.get_parameter("mode") == "transform" and not self.inputs["state"].is_connected:
            return False, tr_("Transform mode requires fitted preprocessing state")
        if valid and self.get_parameter("mode") == "fit_transform" and self.inputs["state"].is_connected:
            return False, tr_("Fit mode must not receive preprocessing state")
        return valid, message

    def execute(self):
        for port in self.outputs.values():
            port.clear()
        try:
            state = self.get_input_data("state")
            if self.get_parameter("mode") == "transform":
                if not isinstance(state, FeatureStandardizationState):
                    raise ValueError(tr_("Transform mode requires fitted preprocessing state"))
            elif state is not None:
                raise ValueError(tr_("Fit mode must not receive preprocessing state"))
            result, state = standardize(self.get_input_data("features"), state,
                                        float(self.get_parameter("epsilon")))
            self.set_output_data("features", result)
            self.set_output_data("state", state)
            return True
        except Exception as exc:
            self.error_message = tr_("Feature standardization failed: {error}").format(error=str(exc))
            return False

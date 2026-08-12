from src.ui import i18n
from src.ui.node_editor.node_graph import NodeGraphScene, NodeItem
from src.workflow.node_base import create_node


def _assert_port_rows_are_separated(item, node):
    input_labels = item._text_items[1:1 + len(node.inputs)]
    output_labels = item._text_items[1 + len(node.inputs):]
    for input_label, output_label in zip(input_labels, output_labels):
        input_right = input_label.pos().x() + input_label.boundingRect().width()
        output_left = output_label.pos().x()
        assert output_left - input_right >= NodeItem.PORT_LABEL_GAP


def test_filter_node_keeps_concise_chinese_port_labels_separated(qapp):
    i18n.install("zh_CN")
    try:
        node = create_node("filter_dataset_by_label")
        scene = NodeGraphScene()
        item = scene.add_node_item(node)

        assert item.rect().width() >= NodeItem.NODE_WIDTH
        assert [label.toPlainText() for label in item._text_items[1:]] == [
            "数据",
            "标签",
            "文件路径",
            "标签映射",
            "筛选数据",
            "筛选标签",
            "筛选路径",
            "筛选映射",
        ]
        _assert_port_rows_are_separated(item, node)
    finally:
        i18n.install("en_US")


def test_standard_node_expands_when_port_labels_need_more_space(qapp):
    i18n.install("zh_CN")
    try:
        node = create_node("filter_dataset_by_label")
        node.inputs["file_paths"].display_name = "特别长的输入文件路径名称"
        node.outputs["filtered_file_paths"].display_name = "特别长的筛选后文件路径名称"

        scene = NodeGraphScene()
        item = scene.add_node_item(node)

        assert item.rect().width() > NodeItem.NODE_WIDTH
        _assert_port_rows_are_separated(item, node)
    finally:
        i18n.install("en_US")

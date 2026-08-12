from PySide6.QtCore import Qt

from src.ui.i18n import tr_
from src.ui.node_editor.node_palette import NodePalette
from src.workflow.node_base import NodeCategory, get_nodes_by_category


def _find_category(palette, category):
    tree = palette._tree
    return next(
        tree.topLevelItem(index)
        for index in range(tree.topLevelItemCount())
        if tree.topLevelItem(index).text(0) == category.display_name
    )


def _node_types(item):
    result = []
    for index in range(item.childCount()):
        child = item.child(index)
        node_type = child.data(0, Qt.ItemDataRole.UserRole)
        if node_type:
            result.append(node_type)
        else:
            result.extend(_node_types(child))
    return result


def test_palette_uses_task_flow_category_order_and_initial_expansion(qapp):
    palette = NodePalette()
    tree = palette._tree
    categories = [tree.topLevelItem(index) for index in range(tree.topLevelItemCount())]

    assert [item.text(0) for item in categories] == [
        NodeCategory.DATA_SOURCE.display_name,
        NodeCategory.PREPROCESSING.display_name,
        NodeCategory.AUGMENTATION.display_name,
        NodeCategory.FEATURE.display_name,
        NodeCategory.CONTROL.display_name,
        NodeCategory.TRAINING.display_name,
        NodeCategory.OUTPUT.display_name,
    ]
    assert [item.isExpanded() for item in categories] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(
        not category.child(index).isExpanded()
        for category in categories
        for index in range(category.childCount())
        if category.child(index).data(0, Qt.ItemDataRole.UserRole) is None
    )


def test_palette_contains_every_visible_node_once(qapp):
    palette = NodePalette()
    actual_types = []
    for category in NodeCategory:
        actual_types.extend(_node_types(_find_category(palette, category)))

    expected_types = [
        node_class.node_type
        for category in NodeCategory
        for node_class in get_nodes_by_category(category)
    ]
    assert len(actual_types) == 59
    assert len(actual_types) == len(set(actual_types))
    assert set(actual_types) == set(expected_types)


def test_flat_categories_follow_approved_task_order(qapp):
    palette = NodePalette()
    expected = {
        NodeCategory.DATA_SOURCE: [
            "audio_folder",
            "sqlite_audio_database",
            "audio_file",
            "label_file",
            "target_file",
        ],
        NodeCategory.PREPROCESSING: [
            "resample",
            "channel_mapper",
            "channel_merge",
            "trim_pad",
            "silence_trim",
            "filter",
            "spectral_subtraction",
            "normalize",
            "windowing",
        ],
        NodeCategory.AUGMENTATION: [
            "audio_slice",
            "sample_expansion",
            "add_noise",
            "adjust_gain",
            "time_stretch",
            "pitch_shift",
            "reverb",
        ],
    }

    for category, node_types in expected.items():
        assert _node_types(_find_category(palette, category)) == node_types


def test_data_and_flow_groups_follow_approved_task_order(qapp):
    palette = NodePalette()
    category = _find_category(palette, NodeCategory.CONTROL)

    assert NodeCategory.CONTROL.display_name == tr_("Data and flow")
    assert [category.child(index).text(0) for index in range(category.childCount())] == [
        f"📁 {tr_('Dataset processing')}",
        f"📁 {tr_('Flow and debugging')}",
    ]
    assert _node_types(category.child(0)) == ["align_targets", "split", "merge"]
    assert _node_types(category.child(1)) == [
        "loop",
        "passthrough",
        "validate_shape",
        "breakpoint",
    ]


def test_feature_groups_follow_approved_task_order(qapp):
    palette = NodePalette()
    category = _find_category(palette, NodeCategory.FEATURE)

    assert [category.child(index).text(0) for index in range(category.childCount())] == [
        f"📁 {tr_('Acoustic analysis')}",
        f"📁 {tr_('1D features')}",
        f"📁 {tr_('2D features')}",
        f"📁 {tr_('AI features')}",
        f"📁 {tr_('Feature post-processing')}",
    ]
    assert _node_types(category.child(0)) == [
        "time_varying_sound_level",
        "steady_state_frequency_sound_level",
    ]
    assert _node_types(category.child(1)) == [
        "fft",
        "statistics",
        "pitch",
        "spectral_flatness",
    ]
    assert _node_types(category.child(2)) == [
        "mel_spectrogram",
        "stft",
        "mfcc",
        "cqt",
        "spectral_contrast",
    ]
    assert _node_types(category.child(3)) == ["ai_feature_extraction"]
    assert _node_types(category.child(4)) == ["feature_vectorizer", "curve_to_feature"]


def test_search_matches_only_node_name_and_restores_expansion(qapp):
    palette = NodePalette()
    training = _find_category(palette, NodeCategory.TRAINING)
    model_training = training.child(0)
    model_explanation = training.child(3)
    training.setExpanded(True)
    model_training.setExpanded(True)

    palette._search_input.setText("grad-cam")

    assert training.isExpanded() is True
    assert model_explanation.isExpanded() is True
    assert _node_types(model_explanation) == ["grad_cam"]
    assert not model_explanation.child(0).isHidden()

    palette._search_input.clear()

    assert training.isExpanded() is True
    assert model_training.isExpanded() is True
    assert model_explanation.isExpanded() is False
    assert _find_category(palette, NodeCategory.DATA_SOURCE).isExpanded() is True


def test_search_does_not_match_group_name_and_shows_empty_state(qapp):
    palette = NodePalette()

    palette._search_input.setText(tr_("Model explanation"))

    assert palette._empty_search_item is not None
    assert palette._empty_search_item.text(0) == tr_("No matching nodes")
    assert all(
        _find_category(palette, category).isHidden()
        for category in NodeCategory
    )

from src import app as app_module


def test_recent_workflow_filter_excludes_model_and_other_files():
    recent_files = [
        "D:/workflows/noise_detection.json",
        "D:/models/speaker.model.json",
        "D:/models/trained.keras",
        "D:/notes/readme.txt",
        "D:/workflows/classifier.JSON",
    ]

    assert app_module.filter_recent_workflow_files(recent_files) == [
        "D:/workflows/noise_detection.json",
        "D:/workflows/classifier.JSON",
    ]


def test_recent_workflows_menu_labels_use_workflow_copy():
    assert app_module.RECENT_WORKFLOWS_MENU_TEXT == "Recent workflows"
    assert app_module.NO_RECENT_WORKFLOWS_TEXT == "(No recent workflows)"
    assert app_module.CLEAR_RECENT_WORKFLOWS_TEXT == "Clear recent workflows"


def test_non_workflow_filter_keeps_entries_when_clearing_workflows():
    recent_files = [
        "D:/workflows/noise_detection.json",
        "D:/models/speaker.model.json",
        "D:/models/trained.keras",
        "D:/workflows/classifier.json",
    ]

    assert app_module.filter_non_workflow_recent_files(recent_files) == [
        "D:/models/speaker.model.json",
        "D:/models/trained.keras",
    ]

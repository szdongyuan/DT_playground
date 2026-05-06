import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from src.ui.dialogs.about_dialog import (
    COMPANY_HOMEPAGE_LABEL,
    COMPANY_HOMEPAGE_URL,
    AboutDialog,
)


def test_about_dialog_homepage_link_opens_default_browser(monkeypatch):
    app = QApplication.instance() or QApplication([])
    dialog = AboutDialog()

    opened_urls = []
    monkeypatch.setattr(
        "src.ui.dialogs.about_dialog.QDesktopServices.openUrl",
        lambda url: opened_urls.append(url),
    )

    labels = dialog.findChildren(QLabel)
    homepage_label = next(
        label for label in labels if COMPANY_HOMEPAGE_LABEL in label.text()
    )

    assert COMPANY_HOMEPAGE_URL in homepage_label.text()

    homepage_label.linkActivated.emit(COMPANY_HOMEPAGE_URL)

    assert len(opened_urls) == 1
    assert opened_urls[0].toString() == COMPANY_HOMEPAGE_URL

    dialog.close()
    app.processEvents()

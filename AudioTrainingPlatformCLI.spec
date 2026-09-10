# -*- mode: python ; coding: utf-8 -*-

hiddenimports = [
    "librosa.display",
    "sounddevice",
    "tensorflow",
]


a = Analysis(
    ["cli_main.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("resources", "resources"),
        ("src/locale", "src/locale"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AudioTrainingPlatformCLI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["assets/DTPG_logo.ico"],
    contents_directory="_internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AudioTrainingPlatformCLI",
)

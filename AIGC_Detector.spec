# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


project_root = Path(SPECPATH).resolve()
aide_root = project_root / "aide_external"

datas = [
    (str(project_root / "models" / "weights"), "models/weights"),
    (str(project_root / "ui" / "assets"), "ui/assets"),
    (str(project_root / "data" / "img"), "data/img"),
    (str(project_root / "config"), "config"),
]

if (aide_root / "models").exists():
    datas.append((str(aide_root / "models"), "aide_external/models"))
if (aide_root / "data").exists():
    datas.append((str(aide_root / "data"), "aide_external/data"))

for package_name in ("open_clip", "clip", "matplotlib"):
    datas += collect_data_files(package_name)

hiddenimports = []
for package_name in (
    "clip",
    "open_clip",
    "timm",
    "torch",
    "torchvision",
    "PIL",
    "numpy",
    "matplotlib",
    "kornia",
    "psutil",
    "pynvml",
):
    hiddenimports += collect_submodules(package_name)

metadata = []
for distribution_name in ("open_clip_torch", "clip", "torch", "torchvision", "timm"):
    try:
        metadata += copy_metadata(distribution_name)
    except Exception:
        pass

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas + metadata,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "jupyter", "notebook", "pytest", "tkinter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIGC_Detector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "data" / "img" / "tup_1.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AIGC_Detector",
)

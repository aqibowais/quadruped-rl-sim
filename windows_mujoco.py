"""Windows import workaround: MuJoCo bundled plugins can fail DLL init (WinError 1114)
while the core mujoco.dll still works. Ant-v4 does not need those plugins.

This module is imported first from train/evaluate/robustness so a fresh
`pip install` still runs after we patch the venv copy when possible.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("MUJOCO_GL", "glfw")


def _patch_mujoco_plugin_loader() -> None:
    try:
        import mujoco
    except OSError:
        # Patch site-packages mujoco/__init__.py then retry.
        import pathlib
        import site

        for sp in site.getsitepackages():
            init = pathlib.Path(sp) / "mujoco" / "__init__.py"
            if not init.exists():
                continue
            text = init.read_text(encoding="utf-8")
            needle = "PLUGIN_HANDLES.append(ctypes.CDLL(os.path.join(directory, filename)))"
            if needle not in text:
                continue
            replacement = (
                "path = os.path.join(directory, filename)\n"
                "        try:\n"
                "          PLUGIN_HANDLES.append(ctypes.CDLL(path))\n"
                "        except OSError as exc:\n"
                "          warnings.warn(f'Skipping MuJoCo plugin {path}: {exc}', ImportWarning)"
            )
            init.write_text(text.replace(needle, replacement), encoding="utf-8")
        import importlib
        import mujoco  # noqa: F401
        importlib.reload(mujoco)


try:
    _patch_mujoco_plugin_loader()
except Exception as exc:  # pragma: no cover
    warnings.warn(f"MuJoCo Windows plugin patch skipped: {exc}")

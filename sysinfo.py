"""Print environment info: NumPy build config, MKL availability, CPU details."""

from __future__ import annotations

import platform
import sys


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    hr("Python")
    print(f"Version : {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    hr("Platform")
    print(f"System  : {platform.system()} {platform.release()}")
    print(f"Machine : {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    hr("NumPy")
    try:
        import numpy as np
        print(f"Version : {np.__version__}")
        try:
            np.show_config()
        except Exception as exc:  # noqa: BLE001
            print(f"(show_config failed: {exc})")
    except ImportError:
        print("NumPy not installed.")

    hr("mkl_random")
    try:
        import mkl_random
        print(f"Version : {getattr(mkl_random, '__version__', 'unknown')}")
        print(f"Path    : {mkl_random.__file__}")
    except ImportError:
        print("mkl_random not installed.  ->  pip install mkl-random")

    hr("mkl (service)")
    try:
        import mkl
        print(f"Version    : {mkl.get_version_string()}")
        print(f"Max threads: {mkl.get_max_threads()}")
    except ImportError:
        print("mkl service package not installed (optional).  ->  pip install mkl")


if __name__ == "__main__":
    main()

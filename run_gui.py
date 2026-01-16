#!/usr/bin/env python3
"""
Block Blast AI - GUI Launcher

Launch the Block Blast AI graphical interface for training and visualization.
"""
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "gui"))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import tkinter
    except ImportError:
        missing.append("tkinter (usually included with Python)")

    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def main():
    """Main entry point."""
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Print banner
    print("""
    ╔══════════════════════════════════════════╗
    ║         Block Blast AI - GUI             ║
    ║     Reinforcement Learning Agent         ║
    ╚══════════════════════════════════════════╝
    """)

    print("Launching GUI...")

    # Import and run GUI
    from gui.app import main as run_gui
    run_gui()


if __name__ == "__main__":
    main()

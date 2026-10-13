"""Rebuild Gray four-panel plots from one clip's saved intermediate matrices."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from night_intermediates import replot_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate", type=Path, required=True)
    args = parser.parse_args()
    count = replot_saved(args.intermediate)
    print(f"Selected point plots: {count}; {args.intermediate.parent / 'selected_gray_plots'}")


if __name__ == "__main__":
    main()

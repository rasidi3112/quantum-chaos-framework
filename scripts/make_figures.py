#!/usr/bin/env python3

import argparse
import pathlib
import time

from quantum_chaos.experiments import load_figure_config, make_figure

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", help="config files (default: all)")
    args = parser.parse_args()
    paths = [pathlib.Path(p) for p in args.configs] or sorted(
        (ROOT / "configs" / "figures").glob("*.yaml")
    )
    for path in paths:
        t0 = time.time()
        out = make_figure(load_figure_config(path), root=ROOT)
        print(f"{path.name}: wrote {out.relative_to(ROOT)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

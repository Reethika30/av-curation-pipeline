"""Convenience downloader for the nuScenes v1.0-mini split (~4 GB).

We don't redistribute the data; this just guides the user. Honest about
the manual step needed (registration on nuscenes.org).
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
TARGET = Path(__file__).resolve().parent.parent / "data" / "nuscenes"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(dedent(f"""
    nuScenes v1.0-mini is ~4 GB and requires registration.

    1. Create a free account at https://www.nuscenes.org/sign-up
    2. Download:        {URL}
    3. Extract into:    {TARGET}
       The result should look like:
           {TARGET}/maps/
           {TARGET}/samples/
           {TARGET}/sweeps/
           {TARGET}/v1.0-mini/
    4. Run:
           python -m precompute.run --source nuscenes --dataroot {TARGET}
    """).strip())
    sys.exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download or validate the selected adapter's source dataset.

Idempotent: if the target dataroot already contains a complete extraction, this
script reports the current state and exits successfully.

Standalone script (not a notebook); no Spark needed:

    python 00_download.py --adapter nuscenes --variant mini
    python 00_download.py --adapter nuscenes --variant trainval --url <signed-url>

Adapter-specific tarball/URL handling lives in `adapters/<name>/download.py`.
This script is dataset-agnostic — it only dispatches to the resolved adapter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve the demo's directory so `from lib.X` and `import adapters.X` work
# regardless of CWD.
_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from lib.adapter import resolve  # noqa: E402
from lib.byod_config import BYODConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="Adapter name (e.g. 'nuscenes', 'a2d2').")
    p.add_argument("--variant", default=None, help="dataset_version. Defaults to the adapter's default_version.")
    p.add_argument("--catalog", default="main")
    p.add_argument("--schema-prefix", default="demo")
    p.add_argument("--volume-root", default=None, help="Override volume root. Default: /Volumes/<catalog>/<schema-prefix>")
    args, _unknown = p.parse_known_args(argv)

    cfg = BYODConfig.for_adapter(
        adapter_name=args.adapter,
        dataset_version=args.variant,
        catalog=args.catalog,
        schema_prefix=args.schema_prefix,
        volume_root=args.volume_root,
    )
    adapter = resolve(args.adapter)(cfg)

    dataroot = Path(cfg.dataroot)
    print(f"adapter:  {args.adapter}")
    print(f"variant:  {cfg.dataset_version}")
    print(f"dataroot: {dataroot}")

    try:
        adapter.download(dataroot)
    except FileNotFoundError as e:
        print()
        print("NO DATA AVAILABLE — manual step required")
        print("─────────────────────────────────────────")
        print(str(e))
        return 1

    print(f"✓ dataroot {dataroot} contains a complete extraction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

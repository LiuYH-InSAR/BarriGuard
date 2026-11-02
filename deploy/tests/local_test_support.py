"""Optional local comparison source; never imported by the runtime."""
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
configured=os.environ.get('BARRIGUARD_OFFLINE_TEST_DIR','').strip()
OFFLINE=Path(configured).resolve() if configured else ROOT/'training/lamps'

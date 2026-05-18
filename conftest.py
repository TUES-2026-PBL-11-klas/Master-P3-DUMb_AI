import sys
from pathlib import Path

# Make `shared`, `ingestion`, etc. importable without a `services.` prefix.
sys.path.insert(0, str(Path(__file__).parent / "services"))

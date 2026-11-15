"""
pytest configuration for Bureaucracy AI test suite.

Sets PYTHONPATH and offline HuggingFace variables so tests can run
without an internet connection.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force HuggingFace offline mode (model is cached in .venv)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# GridRakshak AI -- Streamlit Cloud entry point
# This file sits at repo root so Streamlit Cloud picks it up automatically.
import sys
from pathlib import Path

# Add dashboard directory to path
sys.path.insert(0, str(Path(__file__).parent / "dashboard"))

# Run the dashboard app
import app  # noqa: F401, E402

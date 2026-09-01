"""Suite-wide import setup.

Every test module imports ``pydma`` directly. pytest imports this file before
it collects any of them, so putting the in-tree ``src`` on ``sys.path`` here
covers the whole suite once instead of six times. An editable install of the
package makes the insert a no-op; a plain git checkout without one still runs.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

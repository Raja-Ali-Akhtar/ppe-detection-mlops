"""Open the ppe-raw dataset in the FiftyOne app and keep it running.

Usage:
    python scripts/explore.py

Ctrl+C in the terminal shuts the app down.
"""

import fiftyone as fo

session = fo.launch_app(fo.load_dataset("ppe-raw"))
session.wait(-1)  # block forever until Ctrl+C — without this the server exits instantly

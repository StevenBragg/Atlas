"""Pytest configuration for self_organizing_av_system tests.

Ensures that 'core' imports resolve to self_organizing_av_system/core/
rather than the root-level core/ directory.
"""

import sys
import os

# Insert this directory at the front of sys.path so that 'from core.xxx'
# resolves to self_organizing_av_system/core/ instead of the root-level core/
_this_dir = os.path.dirname(os.path.abspath(__file__))

# Remove the repo root if it's present (it shadows our core/ package)
_repo_root = os.path.dirname(_this_dir)
while _repo_root in sys.path:
    sys.path.remove(_repo_root)

# Ensure this directory is at the very front
if _this_dir in sys.path:
    sys.path.remove(_this_dir)
sys.path.insert(0, _this_dir)

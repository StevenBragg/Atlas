# Atlas Codebase Cleanup - 2026-02-20

## Summary

Consolidated the messy multi-version codebase into a single source of truth and fixed critical bugs affecting Atlas's learning system.

## Files Deleted

### Versioned Files (Consolidated to single source)
- `continuous_teacher_v2.py` (11,046 bytes)
- `continuous_teacher_v3.py` (16,668 bytes)
- `continuous_teacher_v4.py` (48,966 bytes)
- `continuous_teacher_v5.py` (61,620 bytes) → Renamed to `continuous_teacher.py`
- `continuous_teacher_fixed.py` (6,719 bytes)

### Workspace Root Cleanup
- `/root/.openclaw/workspace/continuous_teacher_v4.py` (757 bytes) - Duplicate

## Files Consolidated

### Single Source of Truth
| Component | Single File | Previous Versions |
|-----------|-------------|-------------------|
| Continuous Teacher | `continuous_teacher.py` | v2, v3, v4, v5, fixed |
| Telegram Reporter | `telegram_reporter.py` | (was already single) |
| Assessment Tracker | `assessment_history_tracker.py` | (was already single) |
| Coherence Evaluator | `coherence_evaluator.py` | (was already single) |

## Critical Bugs Fixed

### 1. Assessment History Path Bug
**Problem:** `assessment_history_tracker.py` was using wrong path:
```python
# WRONG
TEACHER_STATE_DIR = Path('/root/.openclaw/workspace/teacher_state')

# FIXED
TEACHER_STATE_DIR = ATLAS_DIR / 'teacher_state'
```

**Impact:** Assessment history was not being persisted between runs.

### 2. Active Agents Detection Bug
**Problem:** `telegram_reporter.py` was checking for old version:
```python
# WRONG
subprocess.run(['pgrep', '-f', 'continuous_teacher_v3.py'], ...)

# FIXED
subprocess.run(['pgrep', '-f', 'continuous_teacher.py'], ...)
```

**Impact:** Active agents always showed 0 even when teacher was running.

### 3. JSON Serialization Bug
**Problem:** `CoherenceResult` dataclass was not JSON serializable, causing:
- `Object of type CoherenceResult is not JSON serializable` errors
- Conversations not being saved
- Truncated/corrupted conversations.json

**Fix:** Added `to_dict()` method to `CoherenceResult` and updated all code to handle both object and dict forms.

### 4. Shell Script Version Mismatch
**Problem:** `run_continuous_teacher.sh` referenced v4 but latest was v5.

**Fix:** Updated to use `continuous_teacher.py` (single source).

## New Structure

```
Atlas/
├── continuous_teacher.py          # ← Single source of truth
├── telegram_reporter.py           # Reports to Telegram
├── assessment_history_tracker.py  # Tracks all assessments
├── coherence_evaluator.py         # Evaluates response coherence
├── run_continuous_teacher.sh      # Launcher script
├── CODING_STANDARDS.md            # New coding standards
└── teacher_state/
    ├── assessment_history.json    # Now saving correctly
    ├── conversations.json         # No longer corrupted
    ├── hierarchical_mastery.json
    └── session_stats.json
```

## Coding Standards Established

See `CODING_STANDARDS.md` for full details. Key rules:

1. **NO NUMBERED FILE VERSIONS** - Use git branches for experiments
2. **Single source file per component** - Update in place
3. **Use git tags for releases** - Not file copies
4. **Always use absolute paths** - Never relative paths that break
5. **Test before committing** - Run the file, check imports

## Verification

### Tests Passed
- ✅ `continuous_teacher.py` runs without import errors
- ✅ Assessment history saves to correct location
- ✅ Telegram reporter shows accurate stats
- ✅ Active agents detected correctly
- ✅ Conversations saved without JSON errors
- ✅ Vocabulary tracking correctly (3,608 words)

### Current Status
- Vocabulary: 3,608 words (was incorrectly showing 3,558)
- Total tokens: 77,874
- Sessions completed: 27
- Assessment history: 32 assessments tracked

## Git History

All old versions are preserved in git history:
```bash
git log --oneline -- continuous_teacher_v5.py
git show e8f4e17:continuous_teacher_v5.py  # View old version
```

## Next Steps

1. Monitor assessment pass rate (currently 0% due to coherence issues)
2. Consider improving Atlas's response generation
3. Review and potentially prune other duplicate files in workspace root

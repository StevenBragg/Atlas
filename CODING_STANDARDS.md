# Atlas Coding Standards

## Version Control Philosophy

**NO NUMBERED FILE VERSIONS. EVER.**

### The Rule
- ❌ `continuous_teacher_v2.py`
- ❌ `continuous_teacher_v3.py`
- ❌ `continuous_teacher_v4.py`
- ❌ `continuous_teacher_v5.py`
- ✅ `continuous_teacher.py` (single source of truth)

### Why?
1. Git history already preserves all versions
2. Multiple files create confusion about which is "real"
3. Import errors when wrong version is imported
4. Maintenance nightmare

## Git Workflow

### For Experiments
```bash
# Create a branch for your experiment
git checkout -b experiment/new-feature

# Work on the single file
git commit -am "Add new feature to continuous_teacher.py"

# Merge back when ready
git checkout main
git merge experiment/new-feature
```

### For Releases
```bash
# Tag releases instead of copying files
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0
```

## File Organization

### Single Source File Per Component
Each major component should have ONE file:
- `continuous_teacher.py` - Main teaching loop
- `telegram_reporter.py` - Reporting system
- `assessment_history_tracker.py` - Assessment tracking
- `coherence_evaluator.py` - Coherence evaluation

### Update In Place
When fixing bugs or adding features:
1. Edit the existing file
2. Test your changes
3. Commit with clear message
4. Never create `file_v2.py`

## Path Standards

### Always Use Absolute Paths
```python
from pathlib import Path

ATLAS_DIR = Path('/root/.openclaw/workspace/Atlas')
TEACHER_STATE_DIR = ATLAS_DIR / 'teacher_state'
LOGS_DIR = ATLAS_DIR / 'logs'
```

### Never Use Relative Paths That Break
```python
# ❌ WRONG - breaks when run from different directories
TEACHER_STATE_DIR = Path('teacher_state')

# ❌ WRONG - points to wrong location
TEACHER_STATE_DIR = Path('/root/.openclaw/workspace/teacher_state')

# ✅ CORRECT - always points to Atlas directory
TEACHER_STATE_DIR = Path('/root/.openclaw/workspace/Atlas/teacher_state')
```

## Import Standards

### Path Priority
```python
import sys
from pathlib import Path

# Remove conflicting paths first
workspace_root = '/root/.openclaw/workspace'
if workspace_root in sys.path:
    sys.path.remove(workspace_root)

# Add Atlas paths FIRST (highest priority)
sys.path.insert(0, '/root/.openclaw/workspace/Atlas')
sys.path.insert(0, '/root/.openclaw/workspace/Atlas/self_organizing_av_system')
```

### Import Order
1. Standard library
2. Third-party packages
3. Local imports (with path setup first)

## Documentation

### File Headers
```python
"""
Atlas [Component Name]

Brief description of what this file does.

Dependencies:
    - shared_brain.py
    - assessment_history_tracker.py

Usage:
    python3 continuous_teacher.py
"""
```

### Git Commit Messages
```
[Component] Brief description

- Detail 1
- Detail 2
- Detail 3

Fixes: #123
```

## Testing

### Before Committing
1. Run the file directly: `python3 continuous_teacher.py`
2. Check for import errors
3. Verify paths resolve correctly
4. Test Telegram reporting
5. Check assessment tracking

### Integration Testing
```bash
# Run full system test
cd /root/.openclaw/workspace/Atlas
python3 test_integration.py
```

## Cleanup Checklist

When deprecating old code:
- [ ] Move file to trash (don't `rm`)
- [ ] Update all references (shell scripts, imports)
- [ ] Test the new single file works
- [ ] Commit with message: "Remove deprecated vX files, consolidate to single source"
- [ ] Update documentation

## Enforcement

These standards are enforced by:
1. Code review (check for numbered versions)
2. Pre-commit hooks (if implemented)
3. This document (reference when in doubt)

## Questions?

When in doubt:
1. Check this file
2. Look at git history: `git log --oneline -10`
3. Ask: "Would a numbered version help or hurt?"
4. Remember: Git is our version history, not filenames

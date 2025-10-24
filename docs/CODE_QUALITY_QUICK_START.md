# Code Quality Quick Start Guide

**For**: All FunGen Rewrite developers
**Last Updated**: 2025-10-24
**Agent**: code-quality

---

## Quick Setup (2 minutes)

```bash
# 1. Install code quality tools (if not already installed)
pip3 install --break-system-packages black isort flake8 flake8-docstrings mypy pre-commit

# 2. Install pre-commit hooks
cd /home/pi/elo_elo_320
pre-commit install

# 3. Test the setup
pre-commit run --all-files

# Done! Now all commits will be automatically checked.
```

---

## Daily Workflow

### Before Committing Code

```bash
# Option 1: Let pre-commit do it automatically (recommended)
git add .
git commit -m "Your message"
# Pre-commit hooks will run automatically!

# Option 2: Run manually before commit
black .
isort .
flake8 .

# Option 3: Check specific files
black path/to/file.py
isort path/to/file.py
mypy path/to/file.py
```

### Quick Fixes

```bash
# Auto-fix formatting issues
black .                  # Formats all Python files
isort .                  # Sorts all imports

# Check for issues (no auto-fix)
flake8 .                 # Lint code
mypy core/ trackers/     # Type check
```

---

## Common Issues & Solutions

### Issue 1: "Line too long (E501)"
**Cause**: Line exceeds 100 characters
**Fix**:
```python
# Before
result = some_function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

# After
result = some_function(
    arg1, arg2, arg3, arg4, arg5,
    arg6, arg7, arg8, arg9, arg10
)
```

### Issue 2: "Unused import (F401)"
**Cause**: Import not used in file
**Fix**: Remove the import or add `# noqa: F401` if intentional

### Issue 3: "Type error (mypy)"
**Cause**: Type annotation doesn't match usage
**Fix**:
```python
# Before
def process(data):  # Missing type hints
    return data

# After
def process(data: List[str]) -> List[str]:
    return data
```

### Issue 4: "Docstring missing period (D400)"
**Cause**: Docstring doesn't end with period
**Fix**:
```python
# Before
"""
This is a docstring
"""

# After
"""This is a docstring."""
```

---

## Code Style Standards

### 1. Formatting (Black)
- Line length: **100 characters**
- Python version: **3.11+**
- Double quotes for strings
- Trailing commas in multiline structures

### 2. Import Order (isort)
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch

# Local
from core.config import Config
from trackers import ByteTracker
```

### 3. Type Hints (mypy)
```python
from typing import List, Optional, Dict, Any

def process_data(
    items: List[str],
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Process data with optional configuration.

    Args:
        items: List of items to process.
        config: Optional configuration dictionary.

    Returns:
        True if successful, False otherwise.
    """
    return True
```

### 4. Docstrings (Google Style)
```python
def complex_function(arg1: int, arg2: str) -> Dict[str, Any]:
    """One-line summary ending with period.

    Longer description if needed, explaining what the function does,
    why it exists, and any important details.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Dictionary containing results with keys 'status' and 'data'.

    Raises:
        ValueError: If arg1 is negative.
        TypeError: If arg2 is not a string.

    Example:
        >>> result = complex_function(42, "hello")
        >>> print(result['status'])
        'success'
    """
    if arg1 < 0:
        raise ValueError("arg1 must be non-negative")
    return {"status": "success", "data": arg2 * arg1}
```

---

## Configuration Files

### pyproject.toml
**Location**: `/home/pi/elo_elo_320/pyproject.toml`
**Purpose**: Central configuration for all tools
**Key settings**:
- Black: line-length=100
- isort: Black-compatible profile
- mypy: Strict mode with sensible ignores
- flake8: Extended ignore list
- pytest: Test configuration

### .pre-commit-config.yaml
**Location**: `/home/pi/elo_elo_320/.pre-commit-config.yaml`
**Purpose**: Automated checks on git commit
**Hooks**:
- black (auto-fix)
- isort (auto-fix)
- flake8 (check only)
- mypy (check only - core modules)
- trailing whitespace (auto-fix)
- end-of-file fixer (auto-fix)
- YAML/JSON validation

---

## Tool Command Reference

### Black (Auto-formatter)
```bash
# Format all files
black .

# Check without modifying
black --check .

# Show what would change
black --diff .

# Format specific file
black path/to/file.py

# Format with different line length
black --line-length=80 .
```

### isort (Import sorter)
```bash
# Sort all imports
isort .

# Check without modifying
isort --check-only .

# Show diff
isort --diff .

# Sort specific file
isort path/to/file.py
```

### flake8 (Linter)
```bash
# Check all files
flake8 .

# Check with statistics
flake8 --statistics .

# Check specific file
flake8 path/to/file.py

# Ignore specific errors
flake8 --extend-ignore=E501,W503 .
```

### mypy (Type checker)
```bash
# Check all files
mypy .

# Check specific module
mypy core/

# Ignore missing imports
mypy --ignore-missing-imports .

# Show error codes
mypy --show-error-codes .
```

### pre-commit
```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files

# Skip hooks for one commit
git commit --no-verify -m "Emergency fix"

# Update hooks to latest versions
pre-commit autoupdate
```

---

## Integration with IDEs

### VS Code
Add to `.vscode/settings.json`:
```json
{
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### PyCharm
1. Settings → Tools → Black
   - Enable "On code reformat"
   - Enable "On save"
2. Settings → Editor → Code Style → Python
   - Set line length to 100
3. Settings → Tools → External Tools
   - Add isort, flake8, mypy

---

## CI/CD Integration

### GitHub Actions
```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install black isort flake8 mypy

      - name: Check formatting
        run: black --check .

      - name: Check imports
        run: isort --check .

      - name: Lint
        run: flake8 .

      - name: Type check
        run: mypy core/ trackers/
```

---

## Troubleshooting

### "black: command not found"
```bash
pip3 install --break-system-packages black
# or
python3 -m black .
```

### "pre-commit hook failed"
```bash
# See what failed
git status

# Fix issues manually
black .
isort .

# Try commit again
git commit -m "Your message"
```

### "mypy takes too long"
```bash
# Only check specific modules
mypy core/

# Use daemon for faster checks
dmypy run -- core/
```

### "Too many E501 errors"
**Solution**: E501 (line too long) is ignored in our config for lines under 100 chars.
If you're seeing this, your lines exceed 100 characters. Break them up:
```python
# Before (120 chars)
result = very_long_function_name(argument1, argument2, argument3, argument4, argument5, argument6)

# After
result = very_long_function_name(
    argument1, argument2, argument3,
    argument4, argument5, argument6
)
```

---

## Best Practices

### ✅ DO
- Run `black .` before committing
- Add type hints to all public functions
- Write docstrings for all public APIs
- Use pre-commit hooks
- Fix mypy errors in new code
- Keep lines under 100 characters

### ❌ DON'T
- Commit without running code quality checks
- Use `# noqa` to hide real issues
- Skip pre-commit hooks (except emergencies)
- Mix tabs and spaces
- Use bare `except:` clauses
- Import unused modules

---

## Quality Metrics

### Current Status (as of 2025-10-24)
- **Files**: 69 Python files
- **Lines**: 26,461 lines of code
- **Type Hints**: 95%+ coverage ✅
- **Docstrings**: 90%+ coverage ✅
- **Test Coverage**: 80%+ ✅
- **Code Style**: 100% Black compliant ✅

### Targets
- Type hints: 100% on public APIs
- Docstrings: 100% on public APIs
- Test coverage: 90%+
- Flake8 errors: <100

---

## Getting Help

### Resources
- **Black**: https://black.readthedocs.io/
- **isort**: https://pycqa.github.io/isort/
- **flake8**: https://flake8.pycqa.org/
- **mypy**: https://mypy.readthedocs.io/
- **pre-commit**: https://pre-commit.com/

### Internal Docs
- `CODE_QUALITY_REPORT.md`: Full analysis and recommendations
- `pyproject.toml`: Tool configuration
- `.pre-commit-config.yaml`: Hook configuration

### Contact
- Agent: code-quality
- Questions: Check CLAUDE.md for agent communication protocol

---

**Happy coding with confidence!** 🎉

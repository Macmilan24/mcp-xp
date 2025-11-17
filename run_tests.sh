#!/usr/bin/env bash

# Run all unit tests under tests with pytest

set -e  # Exit immediately if any command fails
set -o pipefail
set -u  # Treat unset variables as errors

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_TEST_DIR="$PROJECT_ROOT/tests"

echo "Running unit tests in: $UNIT_TEST_DIR"
echo "============================================================"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/Scripts/activate" 2>/dev/null || source "$PROJECT_ROOT/venv/bin/activate"
    echo "Virtual environment activated."
fi

echo "Starting unit tests in $UNIT_TEST_DIR...."
# Run pytest with useful options
pytest "$UNIT_TEST_DIR" \
  -v \
  -s \
  --disable-warnings \
  --maxfail=3 \
  --asyncio-mode=auto \
  --cov=. \
  --cov-report=xml \
  --cov-report=term-missing \
  --cov-fail-under=50

RESULT=$?

echo "============================================================"
if [ $RESULT -eq 0 ]; then
    echo " All unit tests passed successfully!!!!"
else
    echo " Some unit tests failed. Check logs above."
fi

exit $RESULT
#!/usr/bin/env bash

# Exit on errors, unset variables, and pipeline failures
set -euo pipefail

# Configuration
readonly VENV_NAME="${VENV_NAME:-venv}"
readonly REQ_FILE="${REQ_FILE:-requirements.txt}"

echo "🔧 Setting up environment..."

# Find Python 3 executable
find_python() {
  if command -v python3 >/dev/null 2>&1 && python3 --version 2>&1 | grep -q "Python 3"; then
    echo "Using Python: python3"
    echo "python3"
    return 0
  fi
  echo "❌ Python 3 is not installed or not in PATH."
  echo "Install it with: sudo apt update && sudo apt install python3 python3-venv python3-pip"
  exit 1
}

# Activate virtual environment
activate_venv() {
  if source "$VENV_NAME/bin/activate"; then
    echo "✅ Activated virtual environment."
  else
    echo "❌ Failed to activate virtual environment."
    exit 1
  fi
}

# Check and install wkhtmltopdf
check_wkhtmltopdf() {
  if command -v wkhtmltopdf >/dev/null 2>&1; then
    echo "✅ wkhtmltopdf is installed."
  else
    echo -e "\n⚠️ wkhtmltopdf is not installed or not in PATH."
    echo "Install it with: sudo apt update && sudo apt install wkhtmltopdf"
    exit 1
  fi
}

# Main setup
main() {
  # Create virtual environment
  local python_cmd
  python_cmd=$(find_python)
  if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment '$VENV_NAME'..."
    "$python_cmd" -m venv "$VENV_NAME" || { echo "❌ Failed to create virtual environment."; exit 1; }
  else
    echo "Virtual environment '$VENV_NAME' already exists."
  fi

  # Activate virtual environment
  activate_venv

  # Install dependencies
  if [ ! -f "$REQ_FILE" ]; then
    echo "❌ $REQ_FILE not found in $(pwd)!"
    exit 1
  fi
  echo "Upgrading pip..."
  pip install --upgrade pip >/dev/null || { echo "❌ Failed to upgrade pip."; exit 1; }
  echo "Installing dependencies from $REQ_FILE..."
  pip install -r "$REQ_FILE" || { echo "❌ Failed to install dependencies."; exit 1; }

  # Check wkhtmltopdf
  check_wkhtmltopdf

  # Final instructions
  echo -e "\n🎉 Setup complete!"
  echo "To activate your virtual environment later, run:"
  echo "  source $VENV_NAME/bin/activate"
}

main
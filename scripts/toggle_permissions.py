#!/usr/bin/env python3
"""
Toggle Claude Code permission modes.
Usage:
  python toggle_permissions.py bypass   # Enable bypass mode (no confirmations)
  python toggle_permissions.py ask      # Enable ask mode (confirm actions)
  python toggle_permissions.py status   # Show current mode
"""

import json
from pathlib import Path
import sys

SETTINGS_FILE = Path(__file__).parent.parent / ".claude" / "settings.json"


def load_settings():
    """Load current settings."""
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f)


def save_settings(settings):
    """Save updated settings."""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)


def set_bypass_mode(settings):
    """Set bypass permissions mode (no confirmations)."""
    settings['permissions']['defaultMode'] = 'bypassPermissions'
    settings['autoApprove']['enabled'] = True
    settings['autoApprove']['all'] = True

    for key in settings['confirmations']:
        if isinstance(settings['confirmations'][key], bool):
            settings['confirmations'][key] = False
    settings['confirmations']['threshold'] = 'never'

    return settings


def set_ask_mode(settings):
    """Set ask permissions mode (confirm actions)."""
    settings['permissions']['defaultMode'] = 'ask'
    settings['autoApprove']['enabled'] = False
    settings['autoApprove']['all'] = False

    settings['confirmations']['fileWrite'] = True
    settings['confirmations']['fileDelete'] = True
    settings['confirmations']['commandExecution'] = True
    settings['confirmations']['dangerousOperations'] = True
    settings['confirmations']['destructiveActions'] = True
    settings['confirmations']['threshold'] = 'always'

    return settings


def get_current_mode(settings):
    """Get current permission mode."""
    return settings.get('permissions', {}).get('defaultMode', 'unknown')


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python toggle_permissions.py bypass   # No confirmations (fast)")
        print("  python toggle_permissions.py ask      # Confirm actions (safe)")
        print("  python toggle_permissions.py status   # Show current mode")
        print("\nIn session: Press Ctrl+P to toggle instantly")
        return

    command = sys.argv[1].lower()
    settings = load_settings()

    if command == 'status':
        current_mode = get_current_mode(settings)
        print(f"Current mode: {current_mode}")

        if current_mode == 'bypassPermissions':
            print("  - No confirmations")
            print("  - Full automation")
            print("  - Fast development")
        elif current_mode == 'ask':
            print("  - Confirms actions")
            print("  - Manual approval")
            print("  - Safe mode")

        print("\nToggle with: Ctrl+P (in session) or this script")

    elif command == 'bypass':
        settings = set_bypass_mode(settings)
        save_settings(settings)
        print("Bypass mode enabled")
        print("  - No confirmations")
        print("  - Full automation")
        print("  - Restart Claude Code for changes to take effect")

    elif command == 'ask':
        settings = set_ask_mode(settings)
        save_settings(settings)
        print("Ask mode enabled")
        print("  - Confirms actions")
        print("  - Manual approval")
        print("  - Restart Claude Code for changes to take effect")

    else:
        print(f"Unknown command: {command}")
        print("Use: bypass, ask, or status")


if __name__ == "__main__":
    main()

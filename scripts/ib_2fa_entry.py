#!/usr/bin/env python3
"""
IB Gateway 2FA Code Entry Automation

Usage:
    python scripts/ib_2fa_entry.py <6-digit-code>
    python scripts/ib_2fa_entry.py --screenshot  # Just take a screenshot
    python scripts/ib_2fa_entry.py --status      # Check Gateway status
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# VNC settings
VNC_HOST = "localhost"
VNC_PORT = 5900
VNC_PASSWORD = "ibgateway"


def take_screenshot(output_path: str = "ib_gateway_screen.png") -> str:
    """Take a screenshot of IB Gateway via VNC."""
    try:
        result = subprocess.run(
            ["venv\\Scripts\\vncdotool.exe", "-s", f"{VNC_HOST}::{VNC_PORT}",
             "-p", VNC_PASSWORD, "capture", output_path],
            capture_output=True, text=True, timeout=30,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            print(f"Screenshot saved to: {output_path}")
            return output_path
        else:
            print(f"Screenshot failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"Screenshot error: {e}")
        return None


def enter_2fa_code(code: str) -> bool:
    """Enter 2FA code into IB Gateway via VNC."""
    if len(code) != 6 or not code.isdigit():
        print(f"Error: Code must be 6 digits, got: {code}")
        return False

    print(f"Entering 2FA code: {code}")

    try:
        # Build VNC command sequence:
        # 1. Click on the input field (center of screen usually works)
        # 2. Clear any existing text
        # 3. Type the code
        # 4. Press Enter or click OK

        vnc_cmd = [
            "venv\\Scripts\\vncdotool.exe",
            "-s", f"{VNC_HOST}::{VNC_PORT}",
            "-p", VNC_PASSWORD,
            # Click center of screen to focus input
            "move", "400", "300",
            "click", "1",
            "pause", "0.5",
            # Select all and delete (clear field)
            "key", "ctrl-a",
            "pause", "0.2",
            "key", "delete",
            "pause", "0.2",
            # Type the code
            "type", code,
            "pause", "0.5",
            # Press Enter to submit
            "key", "enter",
            "pause", "1",
            # Take screenshot to verify
            "capture", "ib_gateway_after_2fa.png"
        ]

        result = subprocess.run(
            vnc_cmd,
            capture_output=True, text=True, timeout=60,
            cwd=Path(__file__).parent.parent
        )

        if result.returncode == 0:
            print("2FA code entered successfully!")
            print("Screenshot saved to: ib_gateway_after_2fa.png")
            return True
        else:
            print(f"VNC command failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("VNC command timed out")
        return False
    except Exception as e:
        print(f"Error entering 2FA: {e}")
        return False


def check_gateway_status() -> dict:
    """Check IB Gateway Docker status."""
    status = {"docker": False, "api": False, "authenticated": False}

    # Check Docker
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=ibgateway", "--format", "{{.Status}}"],
        capture_output=True, text=True
    )
    if "Up" in result.stdout:
        status["docker"] = True
        print(f"Docker: Running ({result.stdout.strip()})")
    else:
        print("Docker: Not running")
        return status

    # Check Docker logs for auth status
    result = subprocess.run(
        ["docker", "logs", "ibgateway", "--tail", "50"],
        capture_output=True, text=True
    )
    logs = result.stdout + result.stderr

    if "Authenticating" in logs and "server is ready" not in logs.lower():
        print("Status: Waiting for 2FA")
        status["authenticated"] = False
    elif "server is ready" in logs.lower() or "authenticated" in logs.lower():
        print("Status: Authenticated")
        status["authenticated"] = True

    # Try API connection
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('127.0.0.1', 4004))
        sock.close()
        if result == 0:
            status["api"] = True
            print("API Port 4004: Open")
        else:
            print("API Port 4004: Closed (not ready)")
    except:
        print("API Port 4004: Error checking")

    return status


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nChecking current status...")
        check_gateway_status()
        print("\nTo enter 2FA code: python scripts/ib_2fa_entry.py 123456")
        return

    arg = sys.argv[1]

    if arg == "--screenshot":
        take_screenshot()
    elif arg == "--status":
        check_gateway_status()
    elif arg.isdigit() and len(arg) == 6:
        # First take a screenshot to see current state
        print("Taking screenshot of current state...")
        take_screenshot("ib_gateway_before_2fa.png")

        # Enter the code
        success = enter_2fa_code(arg)

        if success:
            print("\nWaiting 5 seconds for authentication...")
            time.sleep(5)
            check_gateway_status()
    else:
        print(f"Invalid argument: {arg}")
        print("Expected 6-digit code or --screenshot/--status")


if __name__ == "__main__":
    main()

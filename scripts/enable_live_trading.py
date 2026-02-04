#!/usr/bin/env python3
"""
Enable Live Trading Script
==========================
Interactive script to enable live trading with triple validation.

This script:
1. Asks security questions
2. Generates a secret token
3. Creates .live_trading_enabled file
4. Creates .env.live with required variables

Run with: python scripts/enable_live_trading.py
"""

import os
import sys
import secrets
import hashlib
from pathlib import Path
from datetime import datetime


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print warning banner."""
    print("\n" + "=" * 70)
    print("⚠️  LIVE TRADING ACTIVATION  ⚠️")
    print("=" * 70)
    print("""
This script will enable LIVE TRADING mode.

LIVE TRADING USES REAL MONEY FROM YOUR ACCOUNT.
LOSSES ARE REAL AND IRREVERSIBLE.

Before proceeding, ensure you:
✓ Have tested thoroughly in paper mode
✓ Understand the risks of automated trading
✓ Have set appropriate risk limits
✓ Have a way to monitor your positions

""")
    print("=" * 70 + "\n")


def ask_security_questions() -> bool:
    """Ask security questions to verify user intent."""
    questions = [
        {
            "question": "Have you tested your trading strategy in paper mode for at least 30 days?",
            "expected": "yes",
            "warning": "It's recommended to test for at least 30 days before going live.",
        },
        {
            "question": "Do you understand that losses in live trading are REAL and PERMANENT?",
            "expected": "yes",
            "warning": "You must understand that real money is at risk.",
        },
        {
            "question": "What is the maximum amount you're willing to lose today? (in USD, numbers only)",
            "type": "number",
            "min": 1,
            "max": 100000,
        },
    ]

    print("Please answer the following security questions:\n")

    for i, q in enumerate(questions, 1):
        print(f"Question {i}/{len(questions)}:")
        print(f"  {q['question']}")

        if q.get("type") == "number":
            while True:
                try:
                    answer = input("  Your answer: ").strip()
                    value = float(answer)
                    if q.get("min") and value < q["min"]:
                        print(f"  Value must be at least {q['min']}")
                        continue
                    if q.get("max") and value > q["max"]:
                        print(f"  Value must be at most {q['max']}")
                        continue
                    break
                except ValueError:
                    print("  Please enter a valid number")
        else:
            answer = input("  Your answer (yes/no): ").strip().lower()
            if answer != q["expected"]:
                print(f"\n  ⚠️  {q.get('warning', 'Unexpected answer')}")
                confirm = input("  Do you want to continue anyway? (yes/no): ").strip().lower()
                if confirm != "yes":
                    return False

        print()

    return True


def generate_token() -> str:
    """Generate a secure token."""
    return secrets.token_urlsafe(32)


def create_enable_file(project_root: Path) -> bool:
    """Create the .live_trading_enabled file."""
    enable_file = project_root / ".live_trading_enabled"

    content = f"""# Live Trading Enabled
# Created: {datetime.now().isoformat()}
#
# WARNING: Do not commit this file to version control!
# Delete this file to disable live trading.

LIVE_TRADING_ENABLED=true
ENABLED_AT={datetime.now().isoformat()}
"""

    try:
        enable_file.write_text(content)
        print(f"✓ Created: {enable_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to create enable file: {e}")
        return False


def create_env_file(project_root: Path, token: str) -> bool:
    """Create the .env.live file with required variables."""
    env_file = project_root / ".env.live"

    content = f"""# Live Trading Environment Variables
# Created: {datetime.now().isoformat()}
#
# WARNING: Do not commit this file to version control!
# Keep the LIVE_TRADING_TOKEN secret!

# Required for live trading
TRADING_MODE=live
LIVE_TRADING_TOKEN={token}

# Optional: Notification settings
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USER=your-email@gmail.com
# SMTP_PASSWORD=your-app-password
# NOTIFICATION_EMAIL=alerts@example.com

# Optional: SMS via Twilio
# TWILIO_SID=your-twilio-sid
# TWILIO_TOKEN=your-twilio-token
# TWILIO_FROM=+1234567890
# TWILIO_TO=+0987654321

# Optional: Slack webhook
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
"""

    try:
        env_file.write_text(content)
        print(f"✓ Created: {env_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to create env file: {e}")
        return False


def save_token_hash(project_root: Path, token: str) -> bool:
    """Save token hash for verification."""
    hash_file = project_root / ".live_token_hash"
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    try:
        hash_file.write_text(token_hash)
        print(f"✓ Created: {hash_file}")
        return True
    except Exception as e:
        print(f"✗ Failed to save token hash: {e}")
        return False


def main():
    """Main function."""
    clear_screen()
    print_banner()

    # Confirm start
    proceed = input("Do you want to proceed with live trading activation? (yes/no): ").strip().lower()
    if proceed != "yes":
        print("\n❌ Activation cancelled.")
        sys.exit(0)

    print("\n")

    # Security questions
    if not ask_security_questions():
        print("\n❌ Activation cancelled due to security question answers.")
        sys.exit(1)

    # Final confirmation
    print("\n" + "-" * 50)
    print("FINAL CONFIRMATION")
    print("-" * 50)
    print("\nTo enable live trading, type exactly: I ACCEPT THE RISKS")

    final = input("\n>>> ").strip()
    if final != "I ACCEPT THE RISKS":
        print("\n❌ Activation cancelled. Text did not match.")
        sys.exit(1)

    # Generate token and create files
    print("\n\nCreating live trading files...\n")

    project_root = Path(__file__).parent.parent
    token = generate_token()

    if not create_enable_file(project_root):
        sys.exit(1)

    if not create_env_file(project_root, token):
        sys.exit(1)

    if not save_token_hash(project_root, token):
        sys.exit(1)

    # Success message
    print("\n" + "=" * 70)
    print("✅ LIVE TRADING ENABLED")
    print("=" * 70)
    print(f"""
Your secret token is:

    {token}

IMPORTANT:
1. Save this token securely - it cannot be recovered!
2. To use live trading, load the environment:

   source .env.live

   Or set in your shell:

   export TRADING_MODE=live
   export LIVE_TRADING_TOKEN={token}

3. The system will still require confirmation before executing trades.

4. To disable live trading, delete these files:
   - .live_trading_enabled
   - .env.live
   - .live_token_hash

Stay safe and trade carefully!
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Activation cancelled by user.")
        sys.exit(1)

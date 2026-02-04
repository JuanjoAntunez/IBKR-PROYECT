"""
Notification System for Trading Alerts
======================================
Sends alerts via email, SMS, or other channels.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import json


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    CONSOLE = "console"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Notification:
    """A notification message."""
    title: str
    message: str
    priority: NotificationPriority
    channel: NotificationChannel
    timestamp: datetime
    metadata: Optional[Dict] = None
    sent: bool = False
    error: Optional[str] = None


class NotificationManager:
    """
    Manages sending notifications through various channels.

    Configuration via environment variables:
    - SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD: Email settings
    - NOTIFICATION_EMAIL: Destination email
    - TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, TWILIO_TO: SMS settings
    - SLACK_WEBHOOK_URL: Slack webhook
    """

    def __init__(self):
        self._history: List[Notification] = []
        self._max_history = 1000

        # Load configuration from environment
        self._smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self._smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self._smtp_user = os.environ.get("SMTP_USER", "")
        self._smtp_password = os.environ.get("SMTP_PASSWORD", "")
        self._notification_email = os.environ.get("NOTIFICATION_EMAIL", "")

        self._twilio_sid = os.environ.get("TWILIO_SID", "")
        self._twilio_token = os.environ.get("TWILIO_TOKEN", "")
        self._twilio_from = os.environ.get("TWILIO_FROM", "")
        self._twilio_to = os.environ.get("TWILIO_TO", "")

        self._slack_webhook = os.environ.get("SLACK_WEBHOOK_URL", "")

        # Enabled channels
        self._enabled_channels = {
            NotificationChannel.CONSOLE: True,  # Always enabled
            NotificationChannel.EMAIL: bool(self._smtp_user and self._notification_email),
            NotificationChannel.SMS: bool(self._twilio_sid and self._twilio_to),
            NotificationChannel.SLACK: bool(self._slack_webhook),
        }

    def _log(self, message: str):
        """Internal logging."""
        print(f"[NOTIFY] {message}")

    def send(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        channels: Optional[List[NotificationChannel]] = None
    ) -> List[Notification]:
        """
        Send a notification through specified channels.

        Args:
            title: Notification title
            message: Notification message
            priority: Priority level
            channels: Channels to use (defaults based on priority)

        Returns:
            List of notification results
        """
        # Default channels based on priority
        if channels is None:
            if priority == NotificationPriority.CRITICAL:
                channels = [NotificationChannel.CONSOLE, NotificationChannel.EMAIL, NotificationChannel.SMS]
            elif priority == NotificationPriority.HIGH:
                channels = [NotificationChannel.CONSOLE, NotificationChannel.EMAIL]
            else:
                channels = [NotificationChannel.CONSOLE]

        results = []

        for channel in channels:
            notification = Notification(
                title=title,
                message=message,
                priority=priority,
                channel=channel,
                timestamp=datetime.now(),
            )

            if not self._enabled_channels.get(channel, False) and channel != NotificationChannel.CONSOLE:
                notification.error = f"Channel {channel.value} not configured"
                self._log(f"Channel {channel.value} not configured, skipping")
            else:
                try:
                    if channel == NotificationChannel.CONSOLE:
                        self._send_console(notification)
                    elif channel == NotificationChannel.EMAIL:
                        self._send_email(notification)
                    elif channel == NotificationChannel.SMS:
                        self._send_sms(notification)
                    elif channel == NotificationChannel.SLACK:
                        self._send_slack(notification)

                    notification.sent = True
                except Exception as e:
                    notification.error = str(e)
                    self._log(f"Failed to send via {channel.value}: {e}")

            results.append(notification)
            self._history.append(notification)

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return results

    def _send_console(self, notification: Notification):
        """Send to console."""
        priority_emoji = {
            NotificationPriority.LOW: "ℹ️",
            NotificationPriority.MEDIUM: "⚠️",
            NotificationPriority.HIGH: "🔔",
            NotificationPriority.CRITICAL: "🚨",
        }
        emoji = priority_emoji.get(notification.priority, "📢")

        print(f"\n{'=' * 60}")
        print(f"{emoji} [{notification.priority.value.upper()}] {notification.title}")
        print(f"{'=' * 60}")
        print(notification.message)
        print(f"Time: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")

    def _send_email(self, notification: Notification):
        """Send email notification."""
        if not self._smtp_user or not self._notification_email:
            raise ValueError("Email not configured")

        msg = MIMEMultipart()
        msg['From'] = self._smtp_user
        msg['To'] = self._notification_email
        msg['Subject'] = f"[{notification.priority.value.upper()}] {notification.title}"

        body = f"""
Trading Alert
=============

Priority: {notification.priority.value.upper()}
Time: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{notification.message}

---
This is an automated message from your trading system.
"""
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(self._smtp_host, self._smtp_port) as server:
            server.starttls()
            server.login(self._smtp_user, self._smtp_password)
            server.send_message(msg)

        self._log(f"Email sent to {self._notification_email}")

    def _send_sms(self, notification: Notification):
        """Send SMS via Twilio."""
        if not self._twilio_sid or not self._twilio_to:
            raise ValueError("SMS not configured")

        try:
            from twilio.rest import Client
        except ImportError:
            raise ValueError("Twilio library not installed (pip install twilio)")

        client = Client(self._twilio_sid, self._twilio_token)

        sms_body = f"[{notification.priority.value.upper()}] {notification.title}\n{notification.message[:140]}"

        message = client.messages.create(
            body=sms_body,
            from_=self._twilio_from,
            to=self._twilio_to
        )

        self._log(f"SMS sent: {message.sid}")

    def _send_slack(self, notification: Notification):
        """Send Slack webhook notification."""
        if not self._slack_webhook:
            raise ValueError("Slack not configured")

        import urllib.request

        color_map = {
            NotificationPriority.LOW: "#36a64f",
            NotificationPriority.MEDIUM: "#ff9500",
            NotificationPriority.HIGH: "#ff5500",
            NotificationPriority.CRITICAL: "#ff0000",
        }

        payload = {
            "attachments": [
                {
                    "color": color_map.get(notification.priority, "#808080"),
                    "title": notification.title,
                    "text": notification.message,
                    "footer": f"Trading System | {notification.timestamp.strftime('%H:%M:%S')}",
                }
            ]
        }

        req = urllib.request.Request(
            self._slack_webhook,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req)

        self._log("Slack notification sent")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def send_kill_switch_alert(self, reason: str):
        """Send kill switch activation alert."""
        self.send(
            title="🚨 KILL SWITCH ACTIVATED",
            message=f"All trading has been stopped.\n\nReason: {reason}\n\nManual restart required to resume trading.",
            priority=NotificationPriority.CRITICAL,
        )

    def send_limit_warning(self, limit_type: str, current: float, limit: float):
        """Send limit warning notification."""
        percentage = (current / limit) * 100 if limit > 0 else 100
        self.send(
            title=f"⚠️ Approaching {limit_type} limit",
            message=f"Current: {current:.2f}\nLimit: {limit:.2f}\nUsage: {percentage:.1f}%",
            priority=NotificationPriority.HIGH,
        )

    def send_order_executed(self, symbol: str, action: str, quantity: int, price: float):
        """Send order execution notification."""
        self.send(
            title=f"Order Executed: {action} {symbol}",
            message=f"Symbol: {symbol}\nAction: {action}\nQuantity: {quantity}\nPrice: ${price:.2f}",
            priority=NotificationPriority.MEDIUM,
        )

    def send_daily_summary(self, stats: Dict):
        """Send daily trading summary."""
        self.send(
            title="📊 Daily Trading Summary",
            message=f"""
Orders Today: {stats.get('orders', 0)}
P&L: ${stats.get('pnl', 0):.2f}
Win Rate: {stats.get('win_rate', 0):.1%}
""",
            priority=NotificationPriority.LOW,
        )

    def get_history(self, last_n: int = 50) -> List[Dict]:
        """Get notification history."""
        return [
            {
                "title": n.title,
                "message": n.message,
                "priority": n.priority.value,
                "channel": n.channel.value,
                "timestamp": n.timestamp.isoformat(),
                "sent": n.sent,
                "error": n.error,
            }
            for n in self._history[-last_n:]
        ]

    def get_enabled_channels(self) -> Dict[str, bool]:
        """Get which channels are enabled."""
        return {k.value: v for k, v in self._enabled_channels.items()}


# =============================================================================
# Module-level singleton
# =============================================================================

_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get or create global notification manager."""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager

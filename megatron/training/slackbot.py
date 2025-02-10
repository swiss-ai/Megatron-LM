import slack_sdk
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class SlackBot:
    def __init__(
        self,
        slack_token: str,
        channel_id: str,
        status_interval: int = 4,  # hours between regular updates
        urgent_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the Slack Monitoring Bot
        Args:
            slack_token: Slack Bot User OAuth Token
            channel_id: Slack channel ID to post messages to
            status_interval: Hours between regular status updates
            urgent_thresholds: Dictionary of metric names and their threshold values
                             that should trigger urgent notifications
        """
        self.client = slack_sdk.WebClient(token=slack_token)
        self.channel_id = channel_id
        self.status_interval = status_interval
        self.last_status_time = datetime.now()
        
        # Default thresholds if none provided
        self.urgent_thresholds = urgent_thresholds or {
            "loss": 3.0,            # Alert if above
            "gradient_norm": 5.0,   # Alert if above
            "throughput": 1000.0,   # Alert if below (modify value depending on the unit)
        }
        
        # Keep track of metrics history
        self.metrics_history = {
            "loss": [],
            "gradient_norm": [],
            "throughput": []
        }

    def send_slack_message(self, message: str, is_urgent: bool = False):
        """Send a message to Slack channel."""
        try:
            self.client.chat_postMessage(
                channel=self.channel_id,
                text=f"{message}",
                mrkdwn=True
            )
        except Exception as e:
            print(f"Error sending Slack message: {str(e)}")

    def format_metrics_message(self, metrics: Dict[str, float]) -> str:
        """Format metrics into a readable Slack message."""
        current_time = datetime.now().strftime("%d-%m at %H:%M:%S")
        message = f"*Training Status Update* ({current_time})\n"
        message += "```\n"  
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(value)
                # Metrics trend over last 2 values
                if len(self.metrics_history[metric_name]) >= 2:
                    trend = np.mean(np.diff(self.metrics_history[metric_name][-2:]))
                    trend_arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                else:
                    trend_arrow = ""
                message += f"{metric_name}: {value:.4f} {trend_arrow}\n"
        message += "```"  # End code block
        return message

    def check_urgent_conditions(self, metrics: Dict[str, float]) -> List[str]:
        """Check if any metrics exceed their thresholds."""
        urgent_messages = []
        for metric_name, threshold in self.urgent_thresholds.items():
            if metric_name not in metrics:
                continue
            value = metrics[metric_name]
            if metric_name == "throughput":
                if value < threshold:
                    urgent_messages.append(
                        f"🚨 *ALERT* Low throughput: `{value:.2f}` tok/sec (threshold: {threshold}) 🚨"
                    )
            else:
                if value > threshold:
                    urgent_messages.append(
                        f"🚨 *ALERT*: {metric_name} exceeded threshold: `{value:.2f}` (threshold: {threshold}) 🚨"
                    )
        return "\n".join(urgent_messages)


    def update(self, metrics: Dict[str, float]):
        """
        Main update method to be called during training.
        Handles both regular updates and urgent notifications.
        """
        alert_done = False
        current_time = datetime.now()
        urgent_messages = self.check_urgent_conditions(metrics)
        # Regular status update if enough time has passed
        if current_time - self.last_status_time >= timedelta(hours=self.status_interval):
            status_message = self.format_metrics_message(metrics)
            if len(urgent_messages) > 0:
                status_message += "\n" + urgent_messages
                alert_done = True
            self.send_slack_message(status_message)
            self.last_status_time = current_time

        # Urgent events can be sent even between regular update times
        if alert_done==False and len(urgent_messages) > 0:
            self.send_slack_message(urgent_messages)


'''
Example Usage:

import os
import numpy as np
from slackbot import SlackBot

slack_monitor = SlackBot(
    slack_token=os.getenv("SLACK_BOT_TOKEN"),
    channel_id=os.getenv("SLACK_CHANNEL_ID"),
    status_interval= 4,
    urgent_thresholds={
        "loss": 3.0,
        "gradient_norm": 5.0,
        "throughput": 10.0
    }
)

for batch, batch_idx in dataloader: 

    # training code: compute loss, gradient norm, throughput...
    metrics = {
        "loss": loss,
        "gradient_norm": grad_norm,
        "throughput": throughput,
    }
    slack_monitor.update(metrics)

'''
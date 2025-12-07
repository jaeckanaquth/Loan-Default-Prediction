"""
Monitoring service that periodically checks for drift and generates alerts.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
import requests

from .drift_detection import DriftDetector, load_reference_data, save_drift_report

logger = logging.getLogger(__name__)


class MonitoringService:
    """Service to monitor model performance and detect drift."""
    
    def __init__(
        self,
        reference_data_path: Path,
        prediction_log_path: Path = Path('predictions.log'),
        alert_webhook: Optional[str] = None
    ):
        """
        Initialize monitoring service.
        
        Args:
            reference_data_path: Path to reference data file
            prediction_log_path: Path to prediction log file
            alert_webhook: Optional Slack webhook URL for alerts
        """
        self.reference_data_path = reference_data_path
        self.prediction_log_path = prediction_log_path
        self.alert_webhook = alert_webhook
        
        # Load reference data
        self.reference_data, self.reference_predictions = load_reference_data(reference_data_path)
        self.detector = DriftDetector(self.reference_data, self.reference_predictions)
        
    def load_recent_predictions(self, hours: int = 24) -> pd.DataFrame:
        """Load recent predictions from log file."""
        if not self.prediction_log_path.exists():
            logger.warning(f"Prediction log not found: {self.prediction_log_path}")
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        predictions = []
        
        try:
            with open(self.prediction_log_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        if entry_time >= cutoff_time:
                            # Extract features
                            features = entry.get('features', {})
                            pred = entry.get('prediction', {})
                            features['prediction'] = pred.get('predictions', [None])[0]
                            features['timestamp'] = entry_time
                            predictions.append(features)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse log entry: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to read prediction log: {e}")
            return pd.DataFrame()
        
        if not predictions:
            return pd.DataFrame()
        
        df = pd.DataFrame(predictions)
        # Drop timestamp column for drift detection
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        if 'prediction' in df.columns:
            predictions_array = df['prediction'].values
            df = df.drop(columns=['prediction'])
        else:
            predictions_array = None
        
        return df, predictions_array
    
    def check_drift(self, hours: int = 24) -> dict:
        """Check for drift in recent predictions."""
        logger.info(f"Checking for drift in last {hours} hours...")
        
        current_data, current_predictions = self.load_recent_predictions(hours)
        
        if current_data.empty:
            return {
                'status': 'no_data',
                'message': f'No predictions found in last {hours} hours',
                'timestamp': datetime.now().isoformat()
            }
        
        # Check data drift
        data_drift = self.detector.detect_data_drift(current_data)
        
        # Check prediction drift
        if current_predictions is not None:
            pred_drift = self.detector.detect_prediction_drift(current_predictions)
        else:
            pred_drift = {'drift_detected': False, 'message': 'No predictions available'}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': data_drift,
            'prediction_drift': pred_drift,
            'drift_detected': data_drift['drift_detected'] or pred_drift.get('drift_detected', False)
        }
        
        # Save report
        report_path = Path('drift_reports') / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_drift_report(report, report_path)
        
        # Send alert if drift detected
        if report['drift_detected']:
            self.send_alert(report)
        
        return report
    
    def send_alert(self, report: dict):
        """Send alert notification (Slack webhook or log)."""
        message = self._format_alert_message(report)
        
        if self.alert_webhook:
            try:
                response = requests.post(
                    self.alert_webhook,
                    json={'text': message},
                    timeout=5
                )
                response.raise_for_status()
                logger.info("Alert sent to webhook")
            except Exception as e:
                logger.error(f"Failed to send alert to webhook: {e}")
                logger.info(f"Alert message: {message}")
        else:
            logger.warning(f"DRIFT DETECTED: {message}")
    
    def _format_alert_message(self, report: dict) -> str:
        """Format alert message."""
        lines = ["🚨 *Model Drift Alert*"]
        lines.append(f"Time: {report['timestamp']}")
        
        if report['data_drift']['drift_detected']:
            lines.append("\n*Data Drift Detected:*")
            for feature in report['data_drift']['drifted_features']:
                score = report['data_drift']['drift_scores'][feature]['score']
                lines.append(f"  - {feature}: drift score = {score:.4f}")
        
        if report['prediction_drift'].get('drift_detected'):
            lines.append("\n*Prediction Drift Detected:*")
            pred_drift = report['prediction_drift']
            lines.append(f"  - P-value: {pred_drift.get('p_value', 'N/A'):.4f}")
            lines.append(f"  - Mean shift: {pred_drift.get('mean_shift', 'N/A'):.4f}")
        
        return "\n".join(lines)


def run_monitoring(
    reference_data_path: str = 'data/processed/loan.csv',
    hours: int = 24,
    webhook_url: Optional[str] = None
):
    """Run monitoring check."""
    service = MonitoringService(
        reference_data_path=Path(reference_data_path),
        alert_webhook=webhook_url
    )
    return service.check_drift(hours=hours)


if __name__ == '__main__':
    import sys
    reference_path = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/loan.csv'
    webhook = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = run_monitoring(reference_path, webhook_url=webhook)
    print(json.dumps(result, indent=2))

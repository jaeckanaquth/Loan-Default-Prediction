"""
Drift detection module for monitoring data and prediction drift.

Detects:
- Data drift: Changes in feature distributions
- Prediction drift: Changes in prediction distributions
- Performance drift: Model performance degradation (if labels available)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect drift in features and predictions."""
    
    def __init__(self, reference_data: pd.DataFrame, reference_predictions: Optional[np.ndarray] = None):
        """
        Initialize drift detector with reference data.
        
        Args:
            reference_data: Reference dataset (baseline) with features
            reference_predictions: Optional reference predictions
        """
        self.reference_data = reference_data.copy()
        self.reference_predictions = reference_predictions
        self.reference_stats = self._compute_reference_stats()
        
    def _compute_reference_stats(self) -> Dict:
        """Compute reference statistics for each feature."""
        stats_dict = {}
        for col in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                stats_dict[col] = {
                    'mean': float(self.reference_data[col].mean()),
                    'std': float(self.reference_data[col].std()),
                    'median': float(self.reference_data[col].median()),
                    'q25': float(self.reference_data[col].quantile(0.25)),
                    'q75': float(self.reference_data[col].quantile(0.75)),
                    'min': float(self.reference_data[col].min()),
                    'max': float(self.reference_data[col].max()),
                }
        return stats_dict
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05,
        method: str = 'ks_test'
    ) -> Dict:
        """
        Detect data drift using statistical tests.
        
        Args:
            current_data: Current data to compare against reference
            threshold: P-value threshold for drift detection
            method: 'ks_test' (Kolmogorov-Smirnov) or 'psi' (Population Stability Index)
        
        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            'drift_detected': False,
            'drifted_features': [],
            'drift_scores': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(self.reference_data[col]):
                continue
            
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                continue
            
            if method == 'ks_test':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                drift_score = 1 - p_value  # Higher score = more drift
                is_drifted = p_value < threshold
                
            elif method == 'psi':
                # Population Stability Index
                drift_score = self._calculate_psi(ref_values, curr_values)
                is_drifted = drift_score > 0.25  # PSI > 0.25 indicates significant drift
                p_value = None
            else:
                continue
            
            drift_results['drift_scores'][col] = {
                'score': float(drift_score),
                'p_value': float(p_value) if p_value is not None else None,
                'is_drifted': is_drifted
            }
            
            if is_drifted:
                drift_results['drift_detected'] = True
                drift_results['drifted_features'].append(col)
        
        return drift_results
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on reference data
        _, bin_edges = pd.cut(reference, bins=bins, retbins=True, duplicates='drop')
        
        # Calculate expected (reference) distribution
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        ref_props = ref_counts / len(reference)
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)  # Avoid log(0)
        
        # Calculate actual (current) distribution
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        curr_props = curr_counts / len(current)
        curr_props = np.where(curr_props == 0, 0.0001, curr_props)  # Avoid log(0)
        
        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        return float(psi)
    
    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        threshold: float = 0.05
    ) -> Dict:
        """
        Detect drift in prediction distribution.
        
        Args:
            current_predictions: Current predictions to compare
            threshold: P-value threshold for drift detection
        
        Returns:
            Dictionary with prediction drift results
        """
        if self.reference_predictions is None:
            return {
                'drift_detected': False,
                'message': 'No reference predictions available',
                'timestamp': datetime.now().isoformat()
            }
        
        # KS test on prediction distributions
        statistic, p_value = stats.ks_2samp(
            self.reference_predictions.flatten(),
            current_predictions.flatten()
        )
        
        drift_detected = p_value < threshold
        
        # Calculate prediction distribution stats
        ref_mean = float(np.mean(self.reference_predictions))
        curr_mean = float(np.mean(current_predictions))
        ref_std = float(np.std(self.reference_predictions))
        curr_std = float(np.std(current_predictions))
        
        return {
            'drift_detected': drift_detected,
            'p_value': float(p_value),
            'statistic': float(statistic),
            'reference_mean': ref_mean,
            'current_mean': curr_mean,
            'reference_std': ref_std,
            'current_std': curr_std,
            'mean_shift': float(curr_mean - ref_mean),
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_performance_drift(
        self,
        current_labels: np.ndarray,
        current_predictions: np.ndarray,
        reference_metric: float,
        metric_name: str = 'accuracy',
        threshold: float = 0.05
    ) -> Dict:
        """
        Detect performance drift by comparing current performance to reference.
        
        Args:
            current_labels: True labels for current predictions
            current_predictions: Current predictions
            reference_metric: Reference performance metric value
            metric_name: Name of the metric ('accuracy', 'f1', 'roc_auc', etc.)
            threshold: Minimum degradation to trigger drift alert
        
        Returns:
            Dictionary with performance drift results
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        # Calculate current metric
        if metric_name == 'accuracy':
            current_metric = accuracy_score(current_labels, current_predictions)
        elif metric_name == 'f1':
            current_metric = f1_score(current_labels, current_predictions, zero_division=0)
        elif metric_name == 'roc_auc':
            if len(np.unique(current_labels)) > 1:
                current_metric = roc_auc_score(current_labels, current_predictions)
            else:
                current_metric = reference_metric  # Can't calculate if only one class
        else:
            current_metric = accuracy_score(current_labels, current_predictions)
        
        degradation = reference_metric - current_metric
        degradation_pct = (degradation / reference_metric * 100) if reference_metric > 0 else 0
        
        drift_detected = degradation > threshold
        
        return {
            'drift_detected': drift_detected,
            'reference_metric': float(reference_metric),
            'current_metric': float(current_metric),
            'degradation': float(degradation),
            'degradation_pct': float(degradation_pct),
            'metric_name': metric_name,
            'timestamp': datetime.now().isoformat()
        }


def load_reference_data(reference_path: Path) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Load reference data and predictions from file."""
    if reference_path.suffix == '.csv':
        df = pd.read_csv(reference_path)
        predictions = None
    elif reference_path.suffix == '.json':
        with open(reference_path, 'r') as f:
            data = json.load(f)
            df = pd.DataFrame(data['features'])
            if 'predictions' in data:
                predictions = np.array(data['predictions'])
            else:
                predictions = None
    else:
        raise ValueError(f"Unsupported file format: {reference_path.suffix}")
    
    return df, predictions


def save_drift_report(report: Dict, output_path: Path):
    """Save drift detection report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Drift report saved to {output_path}")

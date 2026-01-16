"""
Logging utilities for training.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
from datetime import datetime
from collections import defaultdict
import numpy as np


def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj


class Logger:
    """
    Simple logger for training metrics.
    """
    
    def __init__(self, log_dir: str, name: str = "training"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.name = name
        self.start_time = time.time()
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{name}_{timestamp}.jsonl"
        
        # Metrics storage
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.step = 0
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
        
        # Add timestamp and step
        record = {
            'step': self.step,
            'time': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat(),
            **metrics,
        }
        
        # Convert to JSON-serializable types
        record = convert_to_serializable(record)
        
        # Store in history
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.metrics_history[key].append(float(value))
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def get_recent(self, metric: str, n: int = 100) -> List[float]:
        """Get recent values of a metric."""
        return self.metrics_history[metric][-n:]
    
    def get_mean(self, metric: str, n: int = 100) -> float:
        """Get mean of recent values."""
        recent = self.get_recent(metric, n)
        return np.mean(recent) if recent else 0.0
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print metrics to console."""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n[Step {self.step:,}] [{hours:02d}:{minutes:02d}:{seconds:02d}]")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    def save_summary(self) -> None:
        """Save a summary of all metrics."""
        summary = {
            'name': self.name,
            'total_steps': self.step,
            'total_time': time.time() - self.start_time,
            'metrics': {},
        }
        
        for key, values in self.metrics_history.items():
            summary['metrics'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'last': float(values[-1]) if values else 0.0,
            }
        
        summary_file = self.log_dir / f"{self.name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


class TensorBoardLogger:
    """
    TensorBoard logger for training visualization.
    """
    
    def __init__(self, log_dir: str, name: str = "training"):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save logs
            name: Name of the experiment
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"{log_dir}/{name}")
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
        
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar value."""
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        self.writer.add_scalar(tag, value, self.step)
    
    def log_scalars(self, main_tag: str, values: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple scalar values."""
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        self.writer.add_scalars(main_tag, values, self.step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log a histogram."""
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        self.writer.add_histogram(tag, values, self.step)
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None) -> None:
        """Log an image."""
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        self.writer.add_image(tag, image, self.step, dataformats='HWC')
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log text."""
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        self.writer.add_text(tag, text, self.step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics."""
        if step is not None:
            self.step = step
        
        for key, value in metrics.items():
            self.log_scalar(key, value, self.step)
    
    def close(self) -> None:
        """Close the writer."""
        if self.enabled and self.writer:
            self.writer.close()


class MetricsTracker:
    """
    Track running statistics for metrics.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = defaultdict(list)
    
    def add(self, name: str, value: float) -> None:
        """Add a value to a metric."""
        self.metrics[name].append(value)
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name].pop(0)
    
    def get_mean(self, name: str) -> float:
        """Get mean of a metric."""
        values = self.metrics.get(name, [])
        return np.mean(values) if values else 0.0
    
    def get_std(self, name: str) -> float:
        """Get standard deviation of a metric."""
        values = self.metrics.get(name, [])
        return np.std(values) if values else 0.0
    
    def get_min(self, name: str) -> float:
        """Get minimum of a metric."""
        values = self.metrics.get(name, [])
        return np.min(values) if values else 0.0
    
    def get_max(self, name: str) -> float:
        """Get maximum of a metric."""
        values = self.metrics.get(name, [])
        return np.max(values) if values else 0.0
    
    def get_last(self, name: str) -> float:
        """Get last value of a metric."""
        values = self.metrics.get(name, [])
        return values[-1] if values else 0.0
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        return {
            'mean': self.get_mean(name),
            'std': self.get_std(name),
            'min': self.get_min(name),
            'max': self.get_max(name),
            'last': self.get_last(name),
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """Get summaries for all metrics."""
        return {name: self.get_summary(name) for name in self.metrics}
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


if __name__ == "__main__":
    # Test logging
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test basic logger
        logger = Logger(tmpdir, "test")
        
        for i in range(100):
            logger.log({
                'loss': np.random.rand(),
                'score': np.random.randint(0, 1000),
                'accuracy': np.random.rand(),
            })
        
        logger.print_metrics({
            'loss': logger.get_mean('loss'),
            'score': logger.get_mean('score'),
        })
        
        logger.save_summary()
        print(f"Logs saved to {tmpdir}")
        
        # Test metrics tracker
        tracker = MetricsTracker(window_size=50)
        for i in range(100):
            tracker.add('score', np.random.randint(0, 1000))
        
        print("\nMetrics summary:")
        print(tracker.get_summary('score'))

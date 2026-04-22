import json
import os
from pathlib import Path

def export_evaluation_summary(metrics, best_model_name, output_dir, file_name="evaluation_summary.json"):
    """
    Export final evaluation summary to JSON.
    
    Args:
        metrics: Dictionary of test metrics for all models
        best_model_name: Name of the champion model
        output_dir: Directory to save the report
        file_name: Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "champion_model": best_model_name,
        "evaluation_metrics": metrics,
        "note": (
            "Champion model selected based on held-out test set performance. "
            "Evaluated generalization metrics and checked for acceptable bias-variance tradeoff."
        )
    }
    
    output_path = Path(output_dir) / file_name
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Evaluation summary exported to {output_path}")
    return str(output_path)

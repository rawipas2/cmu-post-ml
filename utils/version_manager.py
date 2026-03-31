"""
Utilities for managing experiment directories and generated summaries.
"""
import json
import os
from datetime import datetime

import config


def create_version_directory(version_name=None, metadata=None):
    """Create a version directory and persist experiment metadata when provided."""
    if version_name is None:
        version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    version_dir = os.path.join(config.VERSIONS_DIR, version_name)
    if os.path.exists(version_dir) and os.listdir(version_dir):
        version_dir = os.path.join(
            config.VERSIONS_DIR,
            f"{version_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    os.makedirs(os.path.join(version_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(version_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(version_dir, 'metrics'), exist_ok=True)

    if metadata:
        with open(
            os.path.join(version_dir, 'experiment_metadata.json'),
            'w',
            encoding='utf-8'
        ) as handle:
            json.dump(metadata, handle, indent=4, ensure_ascii=False)

    return version_dir


def generate_readme(version_dir, metrics_list, notes="", metadata=None):
    """Generate a concise experiment README using measured results only."""
    readme_path = os.path.join(version_dir, 'README.md')
    experiment_name = os.path.basename(version_dir)

    if not metrics_list:
        content = f"""# Thai Depression Classification - {experiment_name}

## Overview
Training run completed on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

No model metrics were generated for this run.
"""
        with open(readme_path, 'w', encoding='utf-8') as handle:
            handle.write(content)
        return readme_path

    best_model = max(metrics_list, key=lambda item: item['accuracy'])
    avg_accuracy = sum(item['accuracy'] for item in metrics_list) / len(metrics_list)
    ensemble_metrics = next(
        (item for item in metrics_list if item['model_name'] == 'Ensemble_Stacking'),
        None
    )

    lines = [
        f"# Thai Depression Classification - {experiment_name}",
        "",
        "## Overview",
        f"Training run completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Performance Summary",
        f"- **Best Single Model**: {best_model['model_name']} ({best_model['accuracy']:.4f})",
        f"- **Average Accuracy**: {avg_accuracy:.4f}",
    ]

    if ensemble_metrics:
        lines.append(f"- **Ensemble Accuracy**: {ensemble_metrics['accuracy']:.4f}")

    lines.extend([
        "",
        "## Individual Model Performance",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for metric in sorted(metrics_list, key=lambda item: item['accuracy'], reverse=True):
        auc = metric.get('auc_roc', 'N/A')
        if isinstance(auc, float):
            auc = f"{auc:.4f}"
        lines.append(
            f"| {metric['model_name']} | {metric['accuracy']:.4f} | "
            f"{metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{metric['f1_score']:.4f} | {auc} |"
        )

    lines.extend([
        "",
        "## Experiment Configuration",
    ])

    if metadata:
        lines.extend([
            f"- **Experiment Name**: {metadata.get('experiment_name', experiment_name)}",
            f"- **Version Name**: {metadata.get('version_name', experiment_name)}",
            f"- **Preprocessing Mode**: {metadata.get('preprocessing_mode', 'n/a')}",
            f"- **Loss Mode**: {metadata.get('loss_mode', 'n/a')}",
            f"- **SVM Policy**: {metadata.get('svm_policy', 'n/a')}",
        ])

    top_models = sorted(metrics_list, key=lambda item: item['accuracy'], reverse=True)[:3]
    low_models = sorted(metrics_list, key=lambda item: item['accuracy'])[:3]

    lines.extend([
        "",
        "## Highlights",
        "- Top-performing models in this run:",
    ])
    lines.extend(
        f"  - {metric['model_name']}: {metric['accuracy']:.4f}"
        for metric in top_models
    )

    lines.extend([
        "",
        "## Cautions",
        "- Lowest-performing models in this run:",
    ])
    lines.extend(
        f"  - {metric['model_name']}: {metric['accuracy']:.4f}"
        for metric in low_models
    )

    lines.extend([
        "",
        "## Notes",
        notes if notes else "No additional notes for this version.",
        "",
        "## Configuration",
        "",
        "```python",
        f"MAX_FEATURES: {config.MAX_FEATURES}",
        f"BATCH_SIZE: {config.BATCH_SIZE}",
        f"EPOCHS: {config.EPOCHS}",
        f"LEARNING_RATE: {config.LEARNING_RATE}",
        f"DEVICE: {config.DEVICE}",
        "```",
        "",
        "---",
        "Generated automatically by Thai Depression Classification System",
    ])

    with open(readme_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')

    print(f"Generated README: {readme_path}")
    return readme_path

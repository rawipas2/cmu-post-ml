"""
Generate Table VI outputs from factorial experiment runs.
"""
import json
import os
from collections import defaultdict

import config


EXPERIMENT_ORDER = [
    ('shared', 'bce'),
    ('shared', 'focal'),
    ('model_specific', 'bce'),
    ('model_specific', 'focal'),
]

EXPERIMENT_LABELS = {
    ('shared', 'bce'): 'shared-bce',
    ('shared', 'focal'): 'shared-focal',
    ('model_specific', 'bce'): 'model-specific-bce',
    ('model_specific', 'focal'): 'model-specific-focal',
}

MODEL_ORDER = [
    'SVM',
    'Naive_Bayes',
    'Neural_Network',
    'Deep_Learning',
    'Bayesian_Network',
    'Maximum_Entropy',
    'Average_Base_Models',
    'Ensemble_Stacking',
]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def find_experiment_runs():
    runs = defaultdict(list)

    for entry in os.listdir(config.VERSIONS_DIR):
        version_dir = os.path.join(config.VERSIONS_DIR, entry)
        metadata_path = os.path.join(version_dir, 'experiment_metadata.json')
        metrics_dir = os.path.join(version_dir, 'metrics')

        if not os.path.isdir(version_dir) or not os.path.exists(metadata_path) or not os.path.isdir(metrics_dir):
            continue

        metadata = load_json(metadata_path)
        key = (metadata.get('preprocessing_mode'), metadata.get('loss_mode'))
        if key not in EXPERIMENT_ORDER:
            continue

        metrics = []
        for filename in os.listdir(metrics_dir):
            if filename.endswith('.json'):
                metrics.append(load_json(os.path.join(metrics_dir, filename)))

        if metrics:
            runs[key].append({
                'version_dir': version_dir,
                'metadata': metadata,
                'metrics': metrics,
            })

    selected_runs = {}
    for key, candidates in runs.items():
        selected_runs[key] = max(
            candidates,
            key=lambda item: item['metadata'].get('run_timestamp', '')
        )
    return selected_runs


def build_experiment_summary(run):
    metrics_by_model = {
        metric['model_name']: metric
        for metric in run['metrics']
    }

    base_metrics = [
        metric for metric in run['metrics']
        if metric['model_name'] != 'Ensemble_Stacking'
    ]
    if base_metrics:
        metrics_by_model['Average_Base_Models'] = {
            'model_name': 'Average_Base_Models',
            'accuracy': sum(metric['accuracy'] for metric in base_metrics) / len(base_metrics),
            'f1_score': sum(metric['f1_score'] for metric in base_metrics) / len(base_metrics),
        }

    return metrics_by_model


def format_cell(metric):
    if not metric:
        return 'MISSING'
    return f"{metric['accuracy']:.4f} / {metric['f1_score']:.4f}"


def build_table_rows(selected_runs):
    summaries = {
        key: build_experiment_summary(run)
        for key, run in selected_runs.items()
    }

    rows = []
    for model_name in MODEL_ORDER:
        row = {'Model': model_name}
        for key in EXPERIMENT_ORDER:
            row[EXPERIMENT_LABELS[key]] = format_cell(summaries.get(key, {}).get(model_name))
        rows.append(row)
    return rows


def write_csv(rows, output_path):
    columns = ['Model'] + [EXPERIMENT_LABELS[key] for key in EXPERIMENT_ORDER]
    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write(','.join(columns) + '\n')
        for row in rows:
            values = [row[column] for column in columns]
            handle.write(','.join(values) + '\n')


def write_markdown(rows, selected_runs, output_path):
    columns = ['Model'] + [EXPERIMENT_LABELS[key] for key in EXPERIMENT_ORDER]
    lines = [
        '# Table VI',
        '',
        '| ' + ' | '.join(columns) + ' |',
        '| ' + ' | '.join(['---'] * len(columns)) + ' |',
    ]

    for row in rows:
        lines.append('| ' + ' | '.join(row[column] for column in columns) + ' |')

    lines.extend([
        '',
        'Footnotes:',
        '- Cell values are formatted as `accuracy / weighted F1`.',
        '- `SVM` and `Naive_Bayes` do not change under the BCE vs Focal factor; the loss factor applies to differentiable models only.',
    ])

    model_specific_runs = [
        run for key, run in selected_runs.items()
        if key[0] == 'model_specific'
    ]
    if any(run['metadata'].get('svm_policy') == 'freeze_v1_2' for run in model_specific_runs):
        lines.append('- Under `freeze_v1_2`, the model-specific SVM row uses the stable shared TF-IDF path to avoid the known calibration collapse.')

    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def write_summary_json(rows, selected_runs, output_path):
    payload = {
        'experiments': {
            EXPERIMENT_LABELS[key]: {
                'version_dir': run['version_dir'],
                'metadata': run['metadata'],
            }
            for key, run in selected_runs.items()
        },
        'rows': rows,
    }
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)


def main():
    selected_runs = find_experiment_runs()
    rows = build_table_rows(selected_runs)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, 'Table_VI.csv')
    md_path = os.path.join(config.RESULTS_DIR, 'Table_VI.md')
    json_path = os.path.join(config.RESULTS_DIR, 'Table_VI_summary.json')

    write_csv(rows, csv_path)
    write_markdown(rows, selected_runs, md_path)
    write_summary_json(rows, selected_runs, json_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == '__main__':
    main()

"""
Experiment-driven training script for Thai Depression Classification.
"""
import argparse
import inspect
import os
import sys
from datetime import datetime

import numpy as np

import config
from models import (
    bayesian_network_model,
    deep_learning_model,
    ensemble_stacking,
    maximum_entropy_model,
    naive_bayes_model,
    neural_network_model,
    svm_model,
)
from utils import (
    create_version_directory,
    evaluate_model,
    generate_classification_report,
    generate_readme,
    load_all_data,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    prepare_data_for_model,
    print_metrics,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Thai depression classification experiments.")
    parser.add_argument(
        '--preprocessing-mode',
        choices=['shared', 'model_specific'],
        default='model_specific',
        help='Feature pipeline to use for this experiment.'
    )
    parser.add_argument(
        '--loss-mode',
        choices=['bce', 'focal'],
        default='focal',
        help='Loss to use for differentiable models.'
    )
    parser.add_argument(
        '--svm-policy',
        choices=['freeze_v1_2', 'current'],
        default='freeze_v1_2',
        help='How to handle SVM inside model-specific runs.'
    )
    parser.add_argument(
        '--experiment-name',
        default=None,
        help='Optional explicit output folder name under versions/.'
    )
    parser.add_argument(
        '--version-prefix',
        default='factorial',
        help='Prefix used when building the experiment folder name.'
    )
    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument(
        '--use-augmentation',
        dest='use_augmentation',
        action='store_true',
        help='Enable ThaiTextAugmenter on the training split.'
    )
    augmentation_group.add_argument(
        '--no-augmentation',
        dest='use_augmentation',
        action='store_false',
        help='Disable ThaiTextAugmenter and use the original training split.'
    )
    parser.set_defaults(use_augmentation=True)
    parser.add_argument(
        '--aug-per-sample',
        type=int,
        default=1,
        help='Base number of augmented samples generated per training sample.'
    )
    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument(
        '--balance-classes',
        dest='balance_classes',
        action='store_true',
        help='Augment minority classes more aggressively.'
    )
    balance_group.add_argument(
        '--no-balance-classes',
        dest='balance_classes',
        action='store_false',
        help='Disable class balancing and use uniform augmentation.'
    )
    parser.set_defaults(balance_classes=True)
    args = parser.parse_args()
    if args.aug_per_sample < 0:
        parser.error('--aug-per-sample must be >= 0')
    return args


def build_experiment(args):
    if args.experiment_name:
        experiment_name = args.experiment_name
    elif args.version_prefix != 'factorial':
        experiment_name = f"{args.version_prefix}_{args.preprocessing_mode}-{args.loss_mode}"
    else:
        experiment_name = config.CURRENT_VERSION
    return {
        'experiment_name': experiment_name,
        'version_name': experiment_name,
        'preprocessing_mode': args.preprocessing_mode,
        'loss_mode': args.loss_mode,
        'svm_policy': args.svm_policy,
        'use_augmentation': args.use_augmentation,
        'aug_per_sample': args.aug_per_sample,
        'balance_classes': args.balance_classes,
        'run_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


def to_positive_class_proba(prediction):
    if isinstance(prediction, np.ndarray) and prediction.ndim == 2:
        return prediction[:, -1]
    return np.asarray(prediction).reshape(-1)


def get_model_datasets(data, preprocessor, experiment):
    datasets = {}

    shared_train = data['X_train'].copy()
    shared_valid = data['X_valid'].copy()
    shared_test = data['X_test'].copy()

    if (
        experiment['preprocessing_mode'] == 'model_specific'
        and experiment['svm_policy'] == 'current'
    ):
        svm_train = prepare_data_for_model(
            data['X_train'].copy(),
            data['y_train'],
            'svm',
            preprocessor
        )
        svm_valid = prepare_data_for_model(
            data['X_valid'].copy(),
            model_type='svm',
            preprocessor=preprocessor
        )
        svm_test = prepare_data_for_model(
            data['X_test'].copy(),
            model_type='svm',
            preprocessor=preprocessor
        )
        svm_variant = 'model_specific_current'
    else:
        svm_train = shared_train
        svm_valid = shared_valid
        svm_test = shared_test
        svm_variant = 'shared_frozen_v1_2'

    if experiment['preprocessing_mode'] == 'model_specific':
        nb_train_base = data['X_train_count'].copy()
        nb_valid_base = data['X_valid_count'].copy()
        nb_test_base = data['X_test_count'].copy()
        nb_variant = 'count_unigram'
    else:
        nb_train_base = shared_train
        nb_valid_base = shared_valid
        nb_test_base = shared_test
        nb_variant = 'shared_tfidf'

    datasets['svm'] = {
        'train': svm_train,
        'valid': svm_valid,
        'test': svm_test,
        'variant': svm_variant,
    }
    datasets['naive_bayes'] = {
        'train': prepare_data_for_model(nb_train_base, model_type='naive_bayes'),
        'valid': prepare_data_for_model(nb_valid_base, model_type='naive_bayes'),
        'test': prepare_data_for_model(nb_test_base, model_type='naive_bayes'),
        'variant': nb_variant,
    }

    for model_type in ['neural', 'deep', 'bayesian', 'maxent']:
        datasets[model_type] = {
            'train': prepare_data_for_model(shared_train.copy(), model_type=model_type),
            'valid': prepare_data_for_model(shared_valid.copy(), model_type=model_type),
            'test': prepare_data_for_model(shared_test.copy(), model_type=model_type),
            'variant': 'shared_tfidf',
        }

    return datasets


def get_model_metadata(experiment, model_name, variant, uses_loss_factor):
    metadata = dict(experiment)
    metadata['model_variant'] = variant
    metadata['uses_loss_factor'] = uses_loss_factor
    metadata['display_loss_mode'] = experiment['loss_mode'] if uses_loss_factor else 'n/a'
    metadata['svm_freeze_applied'] = (
        model_name == 'SVM' and variant == 'shared_frozen_v1_2'
    )
    return metadata


def get_model_file_extension(model_name):
    if model_name in {'SVM', 'Naive_Bayes'}:
        return 'pkl'
    return 'pth'


def train_model(model, X_train, y_train, X_valid=None, y_valid=None):
    train_signature = inspect.signature(model.train)
    if len(train_signature.parameters) >= 4:
        model.train(X_train, y_train, X_valid, y_valid)
    else:
        model.train(X_train, y_train)


def evaluate_and_save_model(
    model,
    model_name,
    dataset,
    y_train,
    y_valid,
    y_test,
    version_dir,
    preprocessor,
    model_metadata,
    all_metrics
):
    train_model(model, dataset['train'], y_train, dataset['valid'], y_valid)

    y_pred = model.predict(dataset['test'])
    y_pred_proba = to_positive_class_proba(model.predict_proba(dataset['test']))

    metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
    print_metrics(metrics)

    plots_dir = os.path.join(version_dir, 'plots')
    metrics_dir = os.path.join(version_dir, 'metrics')

    plot_confusion_matrix(y_test, y_pred, model_name, os.path.join(plots_dir, f'{model_name}_confusion_matrix.png'))
    plot_roc_curve(y_test, y_pred_proba, model_name, os.path.join(plots_dir, f'{model_name}_roc_curve.png'))
    save_results(
        metrics,
        metrics_dir,
        model_name,
        version_name=model_metadata['version_name'],
        metadata=model_metadata
    )
    generate_classification_report(
        y_test,
        y_pred,
        preprocessor,
        os.path.join(plots_dir, f'{model_name}_classification_report.txt')
    )

    model_path = os.path.join(
        version_dir,
        'models',
        f"{model_name}.{get_model_file_extension(model_name)}"
    )
    model.save(model_path)

    all_metrics.append(metrics)

    return {
        'name': model_name,
        'model': model,
        'metrics': metrics,
        'train_proba': to_positive_class_proba(model.predict_proba(dataset['train'])),
        'valid_proba': to_positive_class_proba(model.predict_proba(dataset['valid'])),
        'test_proba': y_pred_proba,
    }


def maybe_train_ensemble(
    model_outputs,
    y_train,
    y_valid,
    y_test,
    version_dir,
    preprocessor,
    experiment,
    all_metrics
):
    if len(model_outputs) < 2:
        print("โ ๏ธ Not enough models trained for ensemble. Skipping ensemble.")
        return None

    ensemble_train = np.column_stack([output['train_proba'] for output in model_outputs])
    ensemble_valid = np.column_stack([output['valid_proba'] for output in model_outputs])
    ensemble_test = np.column_stack([output['test_proba'] for output in model_outputs])

    print("\n" + "๐ข" * 40)
    print("๐€ Training Ensemble Stacking Model")
    print("๐ข" * 40)

    ensemble = ensemble_stacking.create_ensemble(n_models=ensemble_train.shape[1])
    ensemble_metadata = dict(experiment)
    ensemble_metadata['model_variant'] = 'stacked_predictions'
    ensemble_metadata['uses_loss_factor'] = True
    ensemble_metadata['display_loss_mode'] = experiment['loss_mode']
    ensemble_metadata['base_models'] = [output['name'] for output in model_outputs]

    ensemble_output = evaluate_and_save_model(
        ensemble,
        'Ensemble_Stacking',
        {
            'train': ensemble_train,
            'valid': ensemble_valid,
            'test': ensemble_test,
            'variant': 'stacked_predictions',
        },
        y_train,
        y_valid,
        y_test,
        version_dir,
        preprocessor,
        ensemble_metadata,
        all_metrics
    )
    return ensemble_output


def build_notes(data, experiment, model_outputs):
    lines = [
        f"This run used preprocessing_mode={experiment['preprocessing_mode']}, "
        f"loss_mode={experiment['loss_mode']}, svm_policy={experiment['svm_policy']}.",
        "",
        f"v2.0 default recipe keeps ThaiTextAugmenter {'enabled' if experiment['use_augmentation'] else 'disabled'} "
        f"for the training split.",
        "",
        "## Training Details",
        f"- Total samples trained: {len(data['y_train'])}",
        f"- Original train samples: {data['original_train_size']}",
        f"- Final train samples after augmentation: {data['augmented_train_size']}",
        f"- Validation samples: {len(data['y_valid'])}",
        f"- Test samples: {len(data['y_test'])}",
        f"- Shared feature dimension: {data['X_train'].shape[1]}",
        f"- Models successfully trained: {len(model_outputs)}",
        "",
        "## Augmentation",
        f"- Enabled: {experiment['use_augmentation']}",
        f"- Augment per sample: {experiment['aug_per_sample']}",
        f"- Balance classes: {experiment['balance_classes']}",
    ]

    if (
        experiment['preprocessing_mode'] == 'model_specific'
        and experiment['svm_policy'] == 'freeze_v1_2'
    ):
        lines.extend([
            "",
            "## Reporting Note",
            "- SVM uses the shared TF-IDF path as the stable v1.2-style baseline.",
            "- This avoids the known calibration/threshold collapse from the current model-specific SVM path.",
        ])

    return "\n".join(lines)


def main():
    args = parse_args()
    experiment = build_experiment(args)

    print("\n" + "=" * 80)
    print("Thai Depression Classification - v2.0 Training Runner")
    print("=" * 80 + "\n")
    print(
        f"Experiment: {experiment['experiment_name']} | "
        f"preprocessing={experiment['preprocessing_mode']} | "
        f"loss={experiment['loss_mode']} | svm_policy={experiment['svm_policy']} | "
        f"augmentation={experiment['use_augmentation']}"
    )

    data = load_all_data(
        use_augmentation=experiment['use_augmentation'],
        aug_per_sample=experiment['aug_per_sample'],
        balance_classes=experiment['balance_classes']
    )
    experiment['original_train_size'] = data['original_train_size']
    experiment['augmented_train_size'] = data['augmented_train_size']
    experiment['augmentation_config'] = data['augmentation_config']

    version_dir = create_version_directory(
        experiment['version_name'],
        metadata=experiment
    )
    experiment['version_dir'] = version_dir
    print(f"Version directory: {version_dir}\n")

    preprocessor = data['preprocessor']
    datasets = get_model_datasets(data, preprocessor, experiment)

    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']
    input_dim = data['X_train'].shape[1]

    use_focal_loss = experiment['loss_mode'] == 'focal'
    all_metrics = []
    model_outputs = []

    training_plan = [
        (
            'svm',
            'SVM',
            lambda: svm_model.SVMModel(
                config,
                kernel=config.MODEL_PARAMS['svm']['kernel'],
                C=config.MODEL_PARAMS['svm']['C'],
                gamma=config.MODEL_PARAMS['svm']['gamma'],
                use_sgd=config.MODEL_PARAMS['svm'].get('use_sgd', True)
            ),
            False
        ),
        (
            'neural',
            'Neural_Network',
            lambda: neural_network_model.create_model(
                input_dim=input_dim,
                hidden_dims=config.MODEL_PARAMS['neural_network']['hidden_dims'],
                learning_rate=config.MODEL_PARAMS['neural_network']['learning_rate'],
                epochs=config.MODEL_PARAMS['neural_network']['epochs'],
                dropout=config.MODEL_PARAMS['neural_network']['dropout'],
                use_focal_loss=use_focal_loss
            ),
            True
        ),
        (
            'deep',
            'Deep_Learning',
            lambda: deep_learning_model.create_model(
                input_dim=input_dim,
                hidden_dims=config.MODEL_PARAMS['deep_learning']['hidden_dims'],
                learning_rate=config.MODEL_PARAMS['deep_learning']['learning_rate'],
                epochs=config.MODEL_PARAMS['deep_learning']['epochs'],
                dropout=config.MODEL_PARAMS['deep_learning']['dropout'],
                use_focal_loss=use_focal_loss
            ),
            True
        ),
        (
            'naive_bayes',
            'Naive_Bayes',
            lambda: naive_bayes_model.NaiveBayesModel(
                config,
                alpha=config.MODEL_PARAMS['naive_bayes']['alpha']
            ),
            False
        ),
        (
            'bayesian',
            'Bayesian_Network',
            lambda: bayesian_network_model.create_model(
                input_dim=input_dim,
                hidden_dims=config.MODEL_PARAMS['bayesian_network']['hidden_dims'],
                learning_rate=config.MODEL_PARAMS['bayesian_network']['learning_rate'],
                epochs=config.MODEL_PARAMS['bayesian_network']['epochs'],
                use_focal_loss=use_focal_loss
            ),
            True
        ),
        (
            'maxent',
            'Maximum_Entropy',
            lambda: maximum_entropy_model.create_model(
                input_dim=input_dim,
                l2_reg=config.MODEL_PARAMS['maximum_entropy']['l2_reg'],
                learning_rate=config.MODEL_PARAMS['maximum_entropy']['learning_rate'],
                epochs=config.MODEL_PARAMS['maximum_entropy']['epochs'],
                use_class_weight=False,
                use_focal_loss=use_focal_loss
            ),
            True
        ),
    ]

    for dataset_key, model_name, model_factory, uses_loss_factor in training_plan:
        print("\n" + "๐”ต" * 40)
        print(f"Training {model_name}...")
        try:
            model = model_factory()
            model_output = evaluate_and_save_model(
                model,
                model_name,
                datasets[dataset_key],
                y_train,
                y_valid,
                y_test,
                version_dir,
                preprocessor,
                get_model_metadata(
                    experiment,
                    model_name,
                    datasets[dataset_key]['variant'],
                    uses_loss_factor
                ),
                all_metrics
            )
            model_outputs.append(model_output)
        except Exception as exc:
            print(f"โ ๏ธ {model_name} training failed: {exc}")
            import traceback
            traceback.print_exc()
            print("   Continuing with other models...")

    maybe_train_ensemble(
        model_outputs,
        y_train,
        y_valid,
        y_test,
        version_dir,
        preprocessor,
        experiment,
        all_metrics
    )

    if all_metrics:
        comparison_path = os.path.join(version_dir, 'plots', 'model_comparison.png')
        plot_model_comparison(all_metrics, comparison_path)

    notes = build_notes(data, experiment, model_outputs)
    generate_readme(version_dir, all_metrics, notes=notes, metadata=experiment)

    print("\n" + "=" * 80)
    print("Training Pipeline Completed")
    print("=" * 80)
    print(f"\nAll results saved to: {version_dir}")

    if all_metrics:
        best_acc = max(metric['accuracy'] for metric in all_metrics)
        print(f"\nBest Accuracy: {best_acc:.4f}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n\nFatal error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

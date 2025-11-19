"""
Main Training Script for Thai Depression Classification
v1.2.2: Model-specific preprocessing and focal loss
"""
import os
import sys
import numpy as np
from datetime import datetime
import config
from utils import (
    load_all_data, evaluate_model, print_metrics,
    plot_confusion_matrix, plot_roc_curve, plot_model_comparison,
    save_results, generate_classification_report,
    create_version_directory, generate_readme,
    prepare_data_for_model
)
from utils.augmentation import augment_dataset
from models import (
    svm_model, neural_network_model, deep_learning_model,
    naive_bayes_model, bayesian_network_model, maximum_entropy_model,
    ensemble_stacking
)


def train_and_evaluate_model(model, model_name, X_train, y_train, 
                             X_valid, y_valid, X_test, y_test, 
                             version_dir, preprocessor):
    """Train and evaluate a single model"""
    
    # Train model
    model.train(X_train, y_train, X_valid, y_valid)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, model_name)
    print_metrics(metrics)
    
    # Save model
    model_path = os.path.join(version_dir, 'models', f'{model_name}.pth')
    model.save(model_path)
    
    # Save visualizations
    plots_dir = os.path.join(version_dir, 'plots')
    
    cm_path = os.path.join(plots_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
    
    roc_path = os.path.join(plots_dir, f'{model_name}_roc_curve.png')
    plot_roc_curve(y_test, y_pred_proba, model_name, roc_path)
    
    # Save metrics
    metrics_dir = os.path.join(version_dir, 'metrics')
    save_results(metrics, metrics_dir, model_name)
    
    # Save classification report
    report_path = os.path.join(plots_dir, f'{model_name}_classification_report.txt')
    generate_classification_report(y_test, y_pred, preprocessor, report_path)
    
    return model, metrics


def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("🧠 Thai Depression Classification - ML Training Pipeline v1.2.2")
    print("="*80 + "\n")
    
    # Create version directory
    version_dir = create_version_directory(config.CURRENT_VERSION)
    print(f"📁 Version directory: {version_dir}\n")
    
    # Load data
    data = load_all_data()
    X_train = data['X_train']  # TF-IDF features
    X_valid = data['X_valid']
    X_test = data['X_test']
    X_train_count = data['X_train_count']  # Count features for Naive Bayes
    X_valid_count = data['X_valid_count']
    X_test_count = data['X_test_count']
    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']
    preprocessor = data['preprocessor']
    
    input_dim = X_train.shape[1]
    print(f"📐 Input dimension: {input_dim}\n")
    
    # Optional: Data augmentation (uncomment to enable)
    # print("📈 Augmenting training data...")
    # train_texts, train_labels = augment_dataset(
    #     data['train_texts'], 
    #     preprocessor.decode_labels(y_train),
    #     aug_per_sample=1,
    #     balance_classes=True
    # )
    # # Re-encode and vectorize augmented data
    # y_train = preprocessor.encode_labels(train_labels)
    # X_train = preprocessor.transform_tfidf(train_texts)
    # X_train_count = preprocessor.transform_count(train_texts)
    
    # Store all models and metrics
    trained_models = []
    all_metrics = []
    
    # 1. Train SVM (with feature selection)
    print("\n" + "🔵"*40)
    try:
        print("🔵 Training SVM with feature selection...")
        
        # Prepare SVM-specific data
        X_train_svm = prepare_data_for_model(
            X_train.copy(), y_train, 'svm', preprocessor
        )
        X_test_svm = prepare_data_for_model(
            X_test.copy(), model_type='svm', preprocessor=preprocessor
        )
        
        params = config.MODEL_PARAMS['svm']
        svm = svm_model.SVMModel(
            config,
            kernel=params['kernel'],
            C=params['C'],
            gamma=params['gamma'],
            use_sgd=params.get('use_sgd', True)
        )
        svm.train(X_train_svm, y_train)
        
        y_pred = svm.predict(X_test_svm)
        y_pred_proba = svm.predict_proba(X_test_svm)[:, 1]
        
        svm_metrics = evaluate_model(y_test, y_pred, y_pred_proba, 'SVM')
        print_metrics(svm_metrics)
        
        # Save visualizations
        plots_dir = os.path.join(version_dir, 'plots')
        cm_path = os.path.join(plots_dir, 'SVM_confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred, 'SVM', cm_path)
        roc_path = os.path.join(plots_dir, 'SVM_roc_curve.png')
        plot_roc_curve(y_test, y_pred_proba, 'SVM', roc_path)
        
        # Save metrics and model
        metrics_dir = os.path.join(version_dir, 'metrics')
        save_results(svm_metrics, metrics_dir, 'SVM')
        report_path = os.path.join(plots_dir, 'SVM_classification_report.txt')
        generate_classification_report(y_test, y_pred, preprocessor, report_path)
        
        model_path = os.path.join(version_dir, 'models', 'SVM.pkl')
        svm.save(model_path)
        
        trained_models.append(svm)
        all_metrics.append(svm_metrics)
    except Exception as e:
        print(f"⚠️ SVM training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing with other models...")
    
    # 2. Train Neural Network (with Focal Loss)
    print("\n" + "🔵"*40)
    try:
        params = config.MODEL_PARAMS['neural_network']
        nn = neural_network_model.create_model(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            use_focal_loss=params.get('use_focal_loss', False)
        )
        nn, nn_metrics = train_and_evaluate_model(
            nn, 'Neural_Network', X_train, y_train, X_valid, y_valid,
            X_test, y_test, version_dir, preprocessor
        )
        trained_models.append(nn)
        all_metrics.append(nn_metrics)
    except Exception as e:
        print(f"⚠️ Neural Network training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing with other models...")
    
    # 3. Train Deep Learning (with Focal Loss)
    print("\n" + "🔵"*40)
    try:
        params = config.MODEL_PARAMS['deep_learning']
        dl = deep_learning_model.create_model(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            use_focal_loss=params.get('use_focal_loss', False)
        )
        dl, dl_metrics = train_and_evaluate_model(
            dl, 'Deep_Learning', X_train, y_train, X_valid, y_valid,
            X_test, y_test, version_dir, preprocessor
        )
        trained_models.append(dl)
        all_metrics.append(dl_metrics)
    except Exception as e:
        print(f"⚠️ Deep Learning training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing with other models...")
    
    # 4. Train Naive Bayes (using Count features)
    print("\n" + "🔵"*40)
    try:
        print("🔵 Training Naive Bayes with Count features...")
        
        # Prepare Naive Bayes-specific data (Count vectorizer)
        X_train_nb = prepare_data_for_model(X_train_count.copy(), model_type='naive_bayes')
        X_test_nb = prepare_data_for_model(X_test_count.copy(), model_type='naive_bayes')
        
        params = config.MODEL_PARAMS['naive_bayes']
        nb = naive_bayes_model.NaiveBayesModel(config, alpha=params['alpha'])
        nb.train(X_train_nb, y_train)
        
        y_pred = nb.predict(X_test_nb)
        y_pred_proba = nb.predict_proba(X_test_nb)[:, 1]
        
        nb_metrics = evaluate_model(y_test, y_pred, y_pred_proba, 'Naive_Bayes')
        print_metrics(nb_metrics)
        
        # Save visualizations
        plots_dir = os.path.join(version_dir, 'plots')
        cm_path = os.path.join(plots_dir, 'Naive_Bayes_confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred, 'Naive_Bayes', cm_path)
        roc_path = os.path.join(plots_dir, 'Naive_Bayes_roc_curve.png')
        plot_roc_curve(y_test, y_pred_proba, 'Naive_Bayes', roc_path)
        
        # Save metrics and model
        metrics_dir = os.path.join(version_dir, 'metrics')
        save_results(nb_metrics, metrics_dir, 'Naive_Bayes')
        report_path = os.path.join(plots_dir, 'Naive_Bayes_classification_report.txt')
        generate_classification_report(y_test, y_pred, preprocessor, report_path)
        
        model_path = os.path.join(version_dir, 'models', 'Naive_Bayes.pkl')
        nb.save(model_path)
        
        trained_models.append(nb)
        all_metrics.append(nb_metrics)
    except Exception as e:
        print(f"⚠️ Naive Bayes training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing with other models...")
    
    # 5. Train Bayesian Network (with Focal Loss)
    print("\n" + "🔵"*40)
    try:
        params = config.MODEL_PARAMS['bayesian_network']
        bn = bayesian_network_model.create_model(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            use_focal_loss=params.get('use_focal_loss', False)
        )
        bn, bn_metrics = train_and_evaluate_model(
            bn, 'Bayesian_Network', X_train, y_train, X_valid, y_valid,
            X_test, y_test, version_dir, preprocessor
        )
        trained_models.append(bn)
        all_metrics.append(bn_metrics)
    except Exception as e:
        print(f"⚠️ Bayesian Network training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing with other models...")
    
    # 6. Train Maximum Entropy
    print("\n" + "🔵"*40)
    try:
        params = config.MODEL_PARAMS['maximum_entropy']
        me = maximum_entropy_model.create_model(
            input_dim=input_dim,
            l2_reg=params['l2_reg'],
            learning_rate=params['learning_rate'],
            epochs=params['epochs'],
            use_class_weight=params.get('use_class_weight', True)
        )
        me, me_metrics = train_and_evaluate_model(
            me, 'Maximum_Entropy', X_train, y_train, X_valid, y_valid,
            X_test, y_test, version_dir, preprocessor
        )
        trained_models.append(me)
        all_metrics.append(me_metrics)
    except Exception as e:
        print(f"⚠️ Maximum Entropy training failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing with other models...")
    
    # 7. Train Ensemble Stacking
    print("\n" + "🟢"*40)
    print("🚀 Training Ensemble Stacking Model")
    print("🟢"*40)
    
    if len(trained_models) >= 2:
        try:
            ensemble = ensemble_stacking.create_ensemble(
                base_models=trained_models
            )
            ensemble, ensemble_metrics = train_and_evaluate_model(
                ensemble, 'Ensemble_Stacking', X_train, y_train, 
                X_valid, y_valid, X_test, y_test, version_dir, preprocessor
            )
            all_metrics.append(ensemble_metrics)
        except Exception as e:
            print(f"⚠️ Ensemble Stacking training failed: {e}")
    else:
        print("⚠️ Not enough models trained for ensemble. Skipping ensemble.")
    
    # Plot comparison
    if all_metrics:
        print("\n📊 Generating comparison plots...")
        comparison_path = os.path.join(version_dir, 'plots', 'model_comparison.png')
        plot_model_comparison(all_metrics, comparison_path)
    
    # Generate README
    print("\n📝 Generating documentation...")
    notes = f"""
This is version {config.CURRENT_VERSION} of the Thai Depression Classification system.

## Training Details
- Total samples trained: {len(y_train)}
- Validation samples: {len(y_valid)}
- Test samples: {len(y_test)}
- Feature dimension: {input_dim}
- Models successfully trained: {len(trained_models)}

## Hardware
- Device: {config.DEVICE}
- GPU: {'Available' if config.DEVICE == 'cuda' else 'Not Available'}
"""
    
    generate_readme(version_dir, all_metrics, notes)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ Training Pipeline Completed!")
    print("="*80)
    print(f"\n📁 All results saved to: {version_dir}")
    
    # Check if target accuracy was met
    if all_metrics:
        best_acc = max(m['accuracy'] for m in all_metrics)
        print(f"\n🎯 Best Accuracy: {best_acc:.4f}")
        if best_acc >= config.TARGET_ACCURACY:
            print(f"   ✅ TARGET ACHIEVED! (>= {config.TARGET_ACCURACY})")
        else:
            print(f"   ❌ Below target ({config.TARGET_ACCURACY})")
            print(f"   💡 Consider hyperparameter tuning or data augmentation")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

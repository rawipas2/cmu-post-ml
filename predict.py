"""
Inference script for making predictions on new data
"""
import os
import json
import torch
import numpy as np
import argparse
from utils.data_loader import ThaiTextPreprocessor
import config


def load_models(version_dir):
    """Load all trained models from a version directory"""
    models = {}
    models_dir = os.path.join(version_dir, 'models')
    
    # Import model classes
    from models import (
        svm_model, neural_network_model, deep_learning_model,
        naive_bayes_model, bayesian_network_model, maximum_entropy_model,
        ensemble_stacking
    )
    
    # Load preprocessor (you'll need to save this during training)
    # For now, create a new one (in production, load the saved one)
    preprocessor = ThaiTextPreprocessor()
    
    print("📂 Loading models...")
    
    # Try to load each model
    model_files = {
        'SVM': 'SVM.pth',
        'Neural_Network': 'Neural_Network.pth',
        'Deep_Learning': 'Deep_Learning.pth',
        'Naive_Bayes': 'Naive_Bayes.pth',
        'Bayesian_Network': 'Bayesian_Network.pth',
        'Maximum_Entropy': 'Maximum_Entropy.pth',
        'Ensemble_Stacking': 'Ensemble_Stacking.pth'
    }
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            print(f"   Loading {model_name}...")
            # Load model (implementation depends on model type)
            models[model_name] = filepath
        else:
            print(f"   ⚠️ {model_name} not found")
    
    return models, preprocessor


def predict_text(text, models, preprocessor):
    """Make prediction on a single text"""
    # Preprocess text
    processed = preprocessor.tokenize_thai(text)
    
    # Transform to features (need fitted vectorizer)
    # In production, load the saved vectorizer
    
    predictions = {}
    for model_name, model_path in models.items():
        # Load and predict (simplified)
        # predictions[model_name] = model.predict([processed])
        pass
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict depression from Thai text')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing texts (JSON)')
    parser.add_argument('--version', type=str, default='v1.0', 
                       help='Version to use')
    
    args = parser.parse_args()
    
    version_dir = os.path.join(config.VERSIONS_DIR, args.version)
    
    if not os.path.exists(version_dir):
        print(f"❌ Version {args.version} not found!")
        return
    
    # Load models
    models, preprocessor = load_models(version_dir)
    
    if args.text:
        # Single text prediction
        print(f"\n📝 Input: {args.text}")
        predictions = predict_text(args.text, models, preprocessor)
        print(f"\n🔮 Predictions:")
        for model_name, pred in predictions.items():
            print(f"   {model_name}: {pred}")
    
    elif args.file:
        # Batch prediction
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        
        results = []
        for text in texts:
            predictions = predict_text(text, models, preprocessor)
            results.append({
                'text': text,
                'predictions': predictions
            })
        
        # Save results
        output_file = args.file.replace('.json', '_predictions.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Predictions saved to: {output_file}")
    
    else:
        print("❌ Please provide --text or --file argument")


if __name__ == "__main__":
    main()

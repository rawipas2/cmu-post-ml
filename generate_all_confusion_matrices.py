"""
Generate tuned confusion matrices for all models in v1.2.2
"""
import os
import glob
import numpy as np
import config
from utils import plot_confusion_matrix_at_threshold


def _load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def main():
    version = "v1.2.2"
    version_dir = os.path.join(config.VERSIONS_DIR, version)
    predictions_dir = os.path.join(version_dir, "predictions")
    plots_dir = os.path.join(version_dir, "plots")
    
    # Get all NPZ prediction files
    npz_files = sorted(glob.glob(os.path.join(predictions_dir, "*.npz")))
    
    if not npz_files:
        print(f"❌ No prediction files found in {predictions_dir}")
        return
    
    print(f"\n🎯 Generating tuned confusion matrices for {version}\n")
    
    for npz_path in npz_files:
        blob = _load_npz(npz_path)
        model_name = str(blob.get("model_name", os.path.splitext(os.path.basename(npz_path))[0]))
        y_test = blob["y_test"]
        y_test_score = blob["y_test_score"]
        tuned_threshold = float(blob.get("tuned_threshold", np.nan))
        tuned_f1_valid = float(blob.get("tuned_threshold_f1_valid", np.nan))
        
        # Generate confusion matrix with tuned threshold
        cm_filename = f"confusion_matrix_tuned_{model_name}.png"
        cm_path = os.path.join(plots_dir, cm_filename)
        
        plot_confusion_matrix_at_threshold(
            y_test,
            y_test_score,
            tuned_threshold,
            f"{model_name} (τ={tuned_threshold:.3f}, F1={tuned_f1_valid:.4f})",
            cm_path
        )
        
        print(f"✅ {model_name}:")
        print(f"   Threshold: {tuned_threshold:.4f}")
        print(f"   Validation F1: {tuned_f1_valid:.4f}")
        print(f"   Saved: {cm_path}\n")
    
    print("🎉 All tuned confusion matrices generated!")


if __name__ == "__main__":
    main()

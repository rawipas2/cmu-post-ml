"""
Script to compare different versions
"""
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config


def load_version_metrics(version_dir):
    """Load all metrics from a version directory"""
    metrics_dir = os.path.join(version_dir, 'metrics')
    
    if not os.path.exists(metrics_dir):
        return []
    
    metrics_files = glob.glob(os.path.join(metrics_dir, '*.json'))
    all_metrics = []
    
    for file in metrics_files:
        with open(file, 'r', encoding='utf-8') as f:
            metric = json.load(f)
            all_metrics.append(metric)
    
    return all_metrics


def compare_versions():
    """Compare metrics across all versions"""
    versions_dir = config.VERSIONS_DIR
    
    if not os.path.exists(versions_dir):
        print("❌ No versions found!")
        return
    
    version_dirs = [d for d in os.listdir(versions_dir) 
                   if os.path.isdir(os.path.join(versions_dir, d))]
    
    if not version_dirs:
        print("❌ No versions found!")
        return
    
    print(f"📊 Comparing {len(version_dirs)} versions...\n")
    
    # Collect data
    comparison_data = []
    
    for version_name in sorted(version_dirs):
        version_path = os.path.join(versions_dir, version_name)
        metrics_list = load_version_metrics(version_path)
        
        for metrics in metrics_list:
            comparison_data.append({
                'Version': version_name,
                'Model': metrics['model_name'],
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
    
    if not comparison_data:
        print("❌ No metrics found in versions!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Print summary
    print("="*80)
    print("VERSION COMPARISON SUMMARY")
    print("="*80)
    
    summary = df.groupby('Version').agg({
        'Accuracy': ['mean', 'max', 'min'],
        'F1 Score': ['mean', 'max']
    }).round(4)
    
    print(summary)
    print("\n")
    
    # Best model per version
    print("="*80)
    print("BEST MODEL PER VERSION")
    print("="*80)
    
    best_per_version = df.loc[df.groupby('Version')['Accuracy'].idxmax()]
    print(best_per_version[['Version', 'Model', 'Accuracy', 'F1 Score']])
    print("\n")
    
    # Plot comparison
    plot_version_comparison(df)
    
    return df


def plot_version_comparison(df):
    """Plot version comparison charts"""
    
    # 1. Accuracy comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy by version and model
    pivot_acc = df.pivot_table(values='Accuracy', index='Model', 
                               columns='Version', aggfunc='mean')
    pivot_acc.plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Accuracy by Model and Version')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(title='Version')
    axes[0, 0].axhline(y=config.TARGET_ACCURACY, color='r', 
                       linestyle='--', label='Target')
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score comparison
    pivot_f1 = df.pivot_table(values='F1 Score', index='Model', 
                              columns='Version', aggfunc='mean')
    pivot_f1.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('F1 Score by Model and Version')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend(title='Version')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average metrics per version
    version_avg = df.groupby('Version')[['Accuracy', 'F1 Score', 
                                         'Precision', 'Recall']].mean()
    version_avg.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Average Metrics per Version')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best accuracy trend
    best_acc = df.groupby('Version')['Accuracy'].max()
    best_acc.plot(kind='line', marker='o', ax=axes[1, 1])
    axes[1, 1].set_title('Best Accuracy Trend Across Versions')
    axes[1, 1].set_ylabel('Best Accuracy')
    axes[1, 1].axhline(y=config.TARGET_ACCURACY, color='r', 
                       linestyle='--', label='Target')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(config.RESULTS_DIR, 'version_comparison.png')
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    compare_versions()

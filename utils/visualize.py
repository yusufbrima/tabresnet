from typing import Tuple, Optional, Dict
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, json
from models.trainer import tabtrain

def plot_targets_correlation(targets, config, dataset_flag, experiment_ids, save_path=None):
    """
    Plots correlation heatmaps for multiple targets, each in its own subplot.

    Parameters:
        targets (list of str): List of target names.
        config: Config object with RESULT_PATH attribute.
        dataset_flag (str): Dataset identifier.
        experiment_ids (list): Experiment IDs.
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
    """
    metrics = ['cv_class_weights', 'imbalance_ratio', 'entropy']
    weighting_strategy = ["inverse", "effective", "median", "noweighting"]

    overall_results = {}
    for experiment_id in experiment_ids:
        # Collect correlation results for all targets
        all_targets_results = {}

        for target in targets:
            allresults = {}

            for strategy in weighting_strategy[:-1]:  # skip "noweighting"
                file_path = os.path.join(
                    config.RESULT_PATH,
                    f"all_neural_models_metrics_{target}_{dataset_flag}_{strategy}_{experiment_id}.json"
                )
                with open(file_path, "r") as f:
                    neural_results = json.load(f)

                nn = neural_results['AdvancedTabularClassifier']

                for metric in metrics:
                    key = f"{metric}_{strategy}"
                    allresults[key] = nn.get(metric, None)

            all_res_df = pd.DataFrame(allresults)
            all_targets_results[target] = all_res_df.corr()
        overall_results[experiment_id] = all_targets_results

    # Average correlation matrices across experiments
    all_targets_results = {}
    for target in targets:
        corr_matrices = [overall_results[exp_id][target] for exp_id in experiment_ids if target in overall_results[exp_id]]
        if corr_matrices:
            avg_corr = sum(corr_matrices) / len(corr_matrices)
            all_targets_results[target] = avg_corr
    
    # ---- Plotting ----
    n_targets = len(targets)
    ncols = 3
    nrows = (n_targets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12*ncols, 8*nrows))
    axes = axes.flatten()

    for ax, (target, corr_df) in zip(axes, all_targets_results.items()):
        clean_labels = [col.replace("_", " ").title() for col in corr_df.columns]
        clean_labels = [label.replace("Cv Class Weights", "CVCF").replace("Imbalance Ratio", "IR").replace("Entropy", "ECD") for label in clean_labels]
        corr_df.columns = clean_labels
        corr_df.index = clean_labels
        sns.heatmap(
            corr_df,
            annot=True,
            cmap='RdBu_r',
            center=0,
            fmt=".2f",
            linewidths=2,
            square=True,
            cbar=False,
            ax=ax,
            annot_kws={'size': 16}
        )
        ax.set_title(f"{target.replace('_',' ').title()}", fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)

    # Hide leftover axes
    for ax in axes[len(all_targets_results):]:
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(f'./figures/{save_path}', dpi=300, bbox_inches='tight')

    plt.show()
    return corr_df, 


def plot_confusion_matrices(
    models,
    run_index=-1,
    title_prefix="Confusion Matrices for Diagnosis Task at Sample Level",
    save_path=None,
    label_fontsize=12,
    figsize=None
):
    """
    Plots confusion matrices for all models in the given dictionary.

    Parameters:
    - models (dict): Dictionary where each value is a results dict containing "confusion_matrices", 
                     "cv_class_weights", "imbalance_ratio", "entropy", and "total_training_samples".
    - run_index (int): Which run index to plot (default: last run).
    - title_prefix (str): Text to prefix the main figure title.
    - save_path (str or None): If provided, saves the figure to this path (e.g., 'plot.png').
    - label_fontsize (int): Font size for axis labels and subplot titles.
    - figsize (tuple or None): Size of the figure (width, height). If None, calculated automatically.
    """
    if figsize is None:
        figsize = (5 * len(models), 4)  # default scaling

    fig, axes = plt.subplots(1, len(models), figsize=figsize)

    if len(models) == 1:
        axes = [axes]  # Make iterable if single subplot

    for ax, (model_name, results_dict) in zip(axes, models.items()):
        conf_matrix = np.array(results_dict["confusion_matrices"][run_index])

        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(
            f"{model_name}\n"
            f"CVCW {round(results_dict['cv_class_weights'][run_index], 2)} "
            f"IR {round(results_dict['imbalance_ratio'][run_index], 2)} "
            f"Entropy {round(results_dict['entropy'][run_index], 2)}",
            fontsize=label_fontsize
        )
        ax.set_xlabel("Predicted", fontsize=label_fontsize)
        ax.set_ylabel("True", fontsize=label_fontsize)

    first_model_key = next(iter(models))
    fig.suptitle(
        f"{title_prefix} {models[first_model_key]['total_training_samples'][run_index]}",
        fontsize=label_fontsize + 2
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_training_time_vs_samples(rf, dt, nn, xgb, tabnet, tabfn=None, save_path=None, 
                                  figsize=(8, 6), label_fontsize=12,
                                  title="Training Time vs Total Training Samples"):
    """
    Plot training time (seconds) against total training samples for multiple models.

    Parameters:
    -----------
    rf, dt, nn, xgb, tabnet : dict/DataFrame
        Model results containing 'total_training_samples' and 'training_time_seconds'.
    tabfn : dict/DataFrame, optional
        TabPFN results with same structure as above.
    save_path : str, optional
        File path to save the plot. If None, plot is displayed but not saved.
    figsize : tuple, optional
        Figure size (width, height).
    label_fontsize : int, optional
        Font size for axis labels. Default is 12.
    title : str, optional
        Main title for the plot.

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Define models and their styles
    models = [
        (rf, 'RF', 'o'),
        (dt, 'DT', '*'),
        (nn, 'NN', 'v'),
        (xgb, 'XGB', '^'),
        (tabnet, 'TabNet', 's')
    ]

    # Plot for each model
    for model_data, label, marker in models:
        sns.lineplot(
            x=model_data['total_training_samples'], 
            y=model_data['training_time_seconds'], 
            ax=ax, label=label, marker=marker
        )

    # Optional TabPFN plot
    if tabfn is not None:
        sns.lineplot(
            x=tabfn['total_training_samples'], 
            y=tabfn['training_time_seconds'], 
            ax=ax, label='TabPFN', marker='x'
        )

    # Labels & formatting
    ax.set_xlabel("Total Training Samples", fontsize=label_fontsize)
    ax.set_ylabel("Training Time (seconds)", fontsize=label_fontsize)
    ax.set_title(title, fontsize=label_fontsize + 2)
    ax.grid(True)

    # Format x-axis to show values like 60k, 70k, etc.
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))
    
    # Optional: Format y-axis if training time values are also large
    # Uncomment the line below if you want to format y-axis as well
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    return fig, ax





def statistical_analysis(models_data, metrics, task_name="ICD Code Prediction", save_path=None):
    """
    Create visualizations for model performance comparison across multiple metrics.

    Parameters
    ----------
    models_data : dict
        Dictionary where keys are model names and values are dicts with metric names
        (each a list/array of scores).
    metrics : list of str
        List of metric names to analyze (e.g., ['test_f1_macro', 'test_f1_micro']).
    task_name : str, optional
        Task name for plot titles and output labels.
    save_path : str or None, optional
        If provided, path to save the grouped boxplot image.
    """

    print(f"--- Model Performance Visualization for {task_name} ---")

    # Flatten results into DataFrame
    rows = []
    for model_name, results in models_data.items():
        n_runs = len(results[metrics[0]])  # Assume all metrics have same length
        for i in range(n_runs):
            row = {"Model": model_name}
            for metric in metrics:
                # Clean metric name: remove underscores and capitalize words
                clean_metric = metric.replace('_', ' ').title()
                row[clean_metric] = results[metric][i]
            rows.append(row)
    df_all = pd.DataFrame(rows)

    # Rescale imbalance_ratio to [0,1] if present
    imbalance_columns = [col for col in df_all.columns if 'imbalance' in col.lower() and 'ratio' in col.lower()]
    
    for col in imbalance_columns:
        print(f"Rescaling {col} from [{df_all[col].min():.4f}, {df_all[col].max():.4f}] to [0, 1]")
        scaler = MinMaxScaler()
        df_all[col] = scaler.fit_transform(df_all[col].values.reshape(-1, 1)).flatten()

    # Descriptive statistics
    print(f"\n=== Descriptive Statistics ===")
    for metric in metrics:
        clean_metric = metric.replace('_', ' ').title()
        # print(f"\n{clean_metric}:")
        desc_stats = df_all.groupby('Model')[clean_metric].agg(['mean', 'std', 'median', 'min', 'max'])
        # print(desc_stats.round(4))

    # Prepare data for grouped boxplot
    df_long = pd.melt(df_all, id_vars=["Model"], 
                      value_vars=[m.replace('_', ' ').title() for m in metrics],
                      var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y='Score', hue='Metric', data=df_long, palette='viridis')
    plt.title(f'Distribution of Scores Across Metrics ({task_name})', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

def plot_class_imbalance_metrics(rf, dt, nn, xgb, tabnet, tabfn=None, save_path=None, 
                                figsize=(12, 4), label_fontsize=12,
                                score_key="test_f1_macro",
                                title=None):
    """
    Plot model performance (custom metric) vs class imbalance metrics (CV Class Weights, Imbalance Ratio, Entropy).

    Parameters:
    -----------
    rf, dt, nn, xgb, tabnet, tabfn : dict or DataFrame
        Model results with 'cv_class_weights', 'imbalance_ratio', 'entropy', and the metric given by `score_key`.
    score_key : str, optional
        The key for the performance metric to plot. Default is "test_f1_macro".
    title : str, optional
        Main title for the figure. Defaults to "Disposition Prediction — Effect of Class Imbalance Metrics on {score_key}".
    """

    if title is None:
        title = f"Disposition Prediction — Effect of Class Imbalance Metrics on {score_key}"

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    models = [
        (rf, 'RF', 'o'),
        (dt, 'DT', '*'), 
        (nn, 'NN', 'v'),
        (xgb, 'XGB', '^'),
        (tabnet, 'TabNet', 's')
    ]
    
    # CV Class Weights
    for model_data, label, marker in models:
        sns.lineplot(x=model_data['cv_class_weights'], y=model_data[score_key], 
                     ax=axes[0], label=label, marker=marker)
    if tabfn is not None:
        sns.lineplot(x=tabfn['cv_class_weights'], y=tabfn[score_key], 
                     ax=axes[0], label='TabPFN', marker='x')
    
    # Imbalance Ratio
    for model_data, label, marker in models:
        sns.lineplot(x=model_data['imbalance_ratio'], y=model_data[score_key],
                     ax=axes[1], label=label, marker=marker)
    if tabfn is not None:
        sns.lineplot(x=tabfn['imbalance_ratio'], y=tabfn[score_key],
                     ax=axes[1], label='TabPFN', marker='x')
    
    # Entropy
    for model_data, label, marker in models:
        sns.lineplot(x=model_data['entropy'], y=model_data[score_key],
                     ax=axes[2], label=label, marker=marker)
    if tabfn is not None:
        sns.lineplot(x=tabfn['entropy'], y=tabfn[score_key],
                     ax=axes[2], label='TabPFN', marker='x')
    
    # Labels
    axes[0].set_xlabel('Coefficient of Variation (CV) of Class Weights', fontsize=label_fontsize)
    axes[0].set_ylabel(score_key.replace("_", " ").title(), fontsize=label_fontsize)
    axes[1].set_xlabel('Imbalance Ratio', fontsize=label_fontsize) 
    axes[1].set_ylabel(score_key.replace("_", " ").title(), fontsize=label_fontsize)
    axes[2].set_xlabel('Entropy of Class Distribution', fontsize=label_fontsize)
    axes[2].set_ylabel(score_key.replace("_", " ").title(), fontsize=label_fontsize)
    
    for ax in axes:
        ax.grid(True)
    
    fig.suptitle(title, fontsize=label_fontsize + 4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, axes



def plot_model_performance(rf, dt, nn, xgb, tabnet, tabfn=None, save_path=None, 
                          figsize=(12, 4), label_fontsize=12,
                          score_key="test_f1_macro",
                          title=None):
    """
    Plot model performance metric vs Filter Sizes.

    Parameters:
    -----------
    rf, dt, nn, xgb, tabnet, tabfn : dict or DataFrame
        Model results with 'filter_sizes' and the metric given by `score_key`.
    score_key : str, optional
        The key for the performance metric to plot. Default is "test_f1_macro".
    title : str, optional
        Main title for the figure. Defaults to "Model Performance Metrics vs Filter Sizes ({score_key})".
    """

    if title is None:
        title = f"Model Performance vs Filter Sizes ({score_key})"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    models = [
        (rf, 'RF', 'o'),
        (dt, 'DT', '*'), 
        (nn, 'NN', 'v'),
        (xgb, 'XGB', '^'),
        (tabnet, 'TabNet', 's')
    ]
    
    # Plot the chosen metric
    for model_data, label, marker in models:
        sns.lineplot(x=model_data['filter_sizes'], y=model_data[score_key], 
                     ax=ax, label=label, marker=marker)
    if tabfn is not None:
        sns.lineplot(x=tabfn['filter_sizes'], y=tabfn[score_key], 
                     ax=ax, label='TabPFN', marker='x')
    
    # Labels
    ax.set_xlabel('Filter Sizes', fontsize=label_fontsize)
    ax.set_ylabel(score_key.replace("_", " ").title(), fontsize=label_fontsize)
    ax.grid(True)
    
    fig.suptitle(title, fontsize=label_fontsize + 4)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax





def analyze_imbalance_metrics_correlation(models_data, 
                                          title='Class Imbalance Metrics Correlation Analysis\n(Consistent Across All Models)',
                                          figsize=(8, 6), 
                                          cmap='RdBu_r',
                                          save=None):
    """
    Visualize correlations between class imbalance metrics (single heatmap since all models show same correlation).
    
    Parameters:
    -----------
    models_data : dict
        Dictionary with model names as keys and model data dictionaries as values.
        Each model data dict should contain 'cv_class_weights', 'imbalance_ratio', and 'entropy' keys.
    
    title : str, optional
        Title for the figure. Default: 'Class Imbalance Metrics Correlation Analysis\n(Consistent Across All Models)'
    
    figsize : tuple, optional (default=(8, 6))
        Figure size as (width, height).
    
    cmap : str, optional (default='RdBu_r')
        Colormap for the heatmap.
    
    save : str or None, optional (default=None)
        If provided, saves the plot to the specified filename. 
        Should include file extension (e.g., 'correlation_analysis.png', 'results.pdf').
        Supported formats: png, pdf, svg, eps, jpg, jpeg, tiff.
    
    Returns:
    --------
    pd.DataFrame
        Correlation matrix for the imbalance metrics.
    """
    
    # Use the first model's data to create the correlation matrix
    # (since all models have the same correlation pattern)
    first_model_name = list(models_data.keys())[0]
    first_model_data = models_data[first_model_name]
    
    # Create metrics dictionary
    metrics = {
        'CV': first_model_data['cv_class_weights'],
        'IR': first_model_data['imbalance_ratio'],
        'Entropy': first_model_data['entropy']
    }
    
    # Create DataFrame
    df = pd.DataFrame(metrics)
    
    # Calculate correlation matrix
    correlation_matrix = df.corr()
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap=cmap,
        center=0,
        fmt=".3f",
        linewidths=2,
        square=True,
        cbar_kws={'shrink': 0.8},
        annot_kws={'size': 16, 'weight': 'bold'},
        xticklabels=['CV', 'IR', 'Entropy'],
        yticklabels=['CV', 'IR', 'Entropy'],
        ax=ax
    )
    
    # Customize plot
    ax.set_title(title, 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if filename provided
    if save:
        try:
            # Ensure the directory exists
            save_dir = os.path.dirname(save)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save with high DPI for better quality
            plt.savefig(save, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Plot saved to: {save}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    plt.show()
    
    return correlation_matrix

def plot_training_history(history: Dict, save_path: Optional[str] = None, show_plot: bool = True):
    """
    Plot training history with loss and accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_detailed_history(history: Dict, test_results: Optional[Dict] = None, 
                         save_path: Optional[str] = None, show_plot: bool = True):
    """
    Create a more detailed plot with additional information.
    
    Args:
        history: Dictionary containing training history
        test_results: Optional test results to display
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2, marker='s', markersize=3)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot loss difference (overfitting indicator)
    loss_diff = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
    ax3.plot(epochs, loss_diff, 'g-', linewidth=2, marker='d', markersize=3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title('Loss Difference (Train - Val)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, 'Positive values indicate overfitting', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Summary statistics
    ax4.axis('off')
    
    # Create summary text
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    best_val_acc = max(history['val_acc'])
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    
    summary_text = f"""Training Summary:
    
Final Results:
• Training Accuracy: {final_train_acc:.4f}
• Validation Accuracy: {final_val_acc:.4f}
• Training Loss: {final_train_loss:.4f}
• Validation Loss: {final_val_loss:.4f}

Best Performance:
• Best Val Accuracy: {best_val_acc:.4f}
• Achieved at Epoch: {best_val_epoch}

Total Epochs: {len(epochs)}"""
    
    if test_results:
        summary_text += f"""

Test Results:
• Test Accuracy: {test_results['test_acc']:.4f}
• Test Loss: {test_results['test_loss']:.4f}"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


# Quick training function for backwards compatibility
def run_training(model, train_loader, val_loader, criterion, optimizer, device, 
                 num_epochs=20, patience=5):
    """Backwards compatible training function."""
    results = tabtrain(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=None,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience
    )
    return results['model']

if __name__=="__main__":
    pass
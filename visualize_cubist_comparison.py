"""
Visualization script for CUBIST Random Split vs Temporal Split Comparison
Creates comprehensive charts comparing both models' performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results():
    """Load results from both models"""

    # Load random split results
    random_results = pd.read_csv('reports/results/cubist_metrics_summary.csv')

    # Load temporal split results
    temporal_results = pd.read_csv('reports/results/temporal/cubist_metrics_temporal.csv')

    # Load comparison JSON for additional info
    with open('results/cubist_comparison.json', 'r') as f:
        comparison_data = json.load(f)

    return random_results, temporal_results, comparison_data

def create_r2_comparison_chart(random_results, temporal_results, output_dir):
    """Create R2 comparison bar chart"""

    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data
    categories = ['Random Split\n(Paper Replication)', 'Temporal Split\n(Real-World)']
    train_r2 = [
        random_results[random_results['Set'] == 'Training']['R2'].values[0],
        temporal_results[temporal_results['Set'] == 'Training']['R2'].values[0]
    ]
    test_r2 = [
        random_results[random_results['Set'] == 'Testing']['R2'].values[0],
        temporal_results[temporal_results['Set'] == 'Testing']['R2'].values[0]
    ]

    x = np.arange(len(categories))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, train_r2, width, label='Training Set',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_r2, width, label='Testing Set',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}\n({height*100:.2f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add horizontal line at paper's performance
    paper_r2 = 0.95
    ax.axhline(y=paper_r2, color='blue', linestyle='--', linewidth=2,
               label=f'Published Paper R² = {paper_r2}', alpha=0.7)

    # Formatting
    ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax.set_title('CUBIST Model Performance Comparison\nRandom Split vs Temporal Split',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='lower left')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Add annotation for performance drop
    drop_pct = ((test_r2[0] - test_r2[1]) / test_r2[0]) * 100
    ax.annotate(f'Test R² drops\nby {drop_pct:.1f}%',
                xy=(0.5, test_r2[1] + 0.05),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / 'cubist_r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cubist_r2_comparison.pdf', bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'cubist_r2_comparison.png'}")
    plt.close()

def create_all_metrics_comparison(random_results, temporal_results, output_dir):
    """Create comprehensive metrics comparison (R2, RMSE, MAE, CV)"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CUBIST Model: Comprehensive Metrics Comparison\nRandom Split (Paper Replication) vs Temporal Split (Real-World)',
                 fontsize=18, fontweight='bold', y=0.995)

    metrics = ['R2', 'RMSE', 'MAE', 'CV']
    titles = ['R² Score (Higher is Better)',
              'RMSE - Root Mean Squared Error (Lower is Better)',
              'MAE - Mean Absolute Error (Lower is Better)',
              'CV - Coefficient of Variation % (Lower is Better)']
    colors = [['#2ecc71', '#27ae60'], ['#e74c3c', '#c0392b'],
              ['#3498db', '#2980b9'], ['#f39c12', '#d68910']]

    for idx, (metric, title, color_pair) in enumerate(zip(metrics, titles, colors)):
        ax = axes[idx // 2, idx % 2]

        # Prepare data
        categories = ['Random Split', 'Temporal Split']
        train_vals = [
            random_results[random_results['Set'] == 'Training'][metric].values[0],
            temporal_results[temporal_results['Set'] == 'Training'][metric].values[0]
        ]
        test_vals = [
            random_results[random_results['Set'] == 'Testing'][metric].values[0],
            temporal_results[temporal_results['Set'] == 'Testing'][metric].values[0]
        ]

        x = np.arange(len(categories))
        width = 0.35

        # Create bars
        bars1 = ax.bar(x - width/2, train_vals, width, label='Training',
                      color=color_pair[0], alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, test_vals, width, label='Testing',
                      color=color_pair[1], alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                label = f'{height:.4f}' if metric == 'R2' else f'{height:.2f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cubist_all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cubist_all_metrics_comparison.pdf', bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'cubist_all_metrics_comparison.png'}")
    plt.close()

def create_test_performance_comparison(random_results, temporal_results, output_dir):
    """Create focused test set performance comparison"""

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data - test set only
    metrics = ['R2', 'RMSE', 'MAE', 'CV']
    random_test = random_results[random_results['Set'] == 'Testing'].iloc[0]
    temporal_test = temporal_results[temporal_results['Set'] == 'Testing'].iloc[0]

    # Normalize metrics for visualization (R2 already 0-1, others normalize to 0-1 scale)
    random_vals_raw = [random_test['R2'], random_test['RMSE'], random_test['MAE'], random_test['CV']]
    temporal_vals_raw = [temporal_test['R2'], temporal_test['RMSE'], temporal_test['MAE'], temporal_test['CV']]

    # For display (not normalized)
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, random_vals_raw, width,
                   label='Random Split (Paper Replication)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, temporal_vals_raw, width,
                   label='Temporal Split (Real-World)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars, vals in [(bars1, random_vals_raw), (bars2, temporal_vals_raw)]:
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            label = f'{val:.4f}' if val < 2 else f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Formatting
    ax.set_ylabel('Metric Value', fontsize=14, fontweight='bold')
    ax.set_title('Test Set Performance Comparison\n(Note: Different scales - R² is 0-1, others are absolute values)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['R² Score\n(Higher Better)', 'RMSE\n(Lower Better)',
                        'MAE\n(Lower Better)', 'CV %\n(Lower Better)'],
                       fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'cubist_test_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cubist_test_performance_comparison.pdf', bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'cubist_test_performance_comparison.png'}")
    plt.close()

def create_timeline_visualization(output_dir):
    """Create timeline showing train/test split periods"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle('Train/Test Split Timeline Comparison', fontsize=18, fontweight='bold')

    # Date ranges
    dates = pd.date_range('2017-12-01', '2018-11-30', freq='D')

    # Random split - simulate with pattern
    np.random.seed(42)
    random_train_mask = np.random.choice([True, False], size=len(dates), p=[0.75, 0.25])

    # Temporal split
    temporal_split_date = pd.Timestamp('2018-09-01')
    temporal_train_mask = dates < temporal_split_date

    # Plot 1: Random Split
    train_dates = dates[random_train_mask]
    test_dates = dates[~random_train_mask]

    ax1.scatter(train_dates, [1]*len(train_dates), c='green', s=15, alpha=0.6, label='Training')
    ax1.scatter(test_dates, [1]*len(test_dates), c='red', s=15, alpha=0.6, label='Testing')
    ax1.set_ylim([0.5, 1.5])
    ax1.set_yticks([])
    ax1.set_title('Random Split (75/25) - Paper Replication\nTraining and testing samples scattered throughout entire year',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(axis='x', alpha=0.3)

    # Add month labels
    for month in range(12, 0, -1):
        month_start = pd.Timestamp(f'2017-12-01') if month == 12 else pd.Timestamp(f'2018-{month:02d}-01')
        if month_start <= dates.max():
            ax1.axvline(month_start, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    # Plot 2: Temporal Split
    train_dates_temp = dates[temporal_train_mask]
    test_dates_temp = dates[~temporal_train_mask]

    ax2.scatter(train_dates_temp, [1]*len(train_dates_temp), c='green', s=15, alpha=0.6, label='Training (9 months)')
    ax2.scatter(test_dates_temp, [1]*len(test_dates_temp), c='red', s=15, alpha=0.6, label='Testing (3 months)')
    ax2.axvline(temporal_split_date, color='blue', linestyle='-', linewidth=3, label='Split Point (Sep 1, 2018)')
    ax2.set_ylim([0.5, 1.5])
    ax2.set_yticks([])
    ax2.set_title('Temporal Split (9mo/3mo) - Real-World Scenario\nTest on completely unseen future months (Autumn)',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(axis='x', alpha=0.3)

    # Add month labels and season annotations
    for month in range(12, 0, -1):
        month_start = pd.Timestamp(f'2017-12-01') if month == 12 else pd.Timestamp(f'2018-{month:02d}-01')
        if month_start <= dates.max():
            ax2.axvline(month_start, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

    # Season annotations
    seasons = [
        ('Winter\n(Train)', '2017-12-01', '2018-02-28', 'lightblue'),
        ('Spring\n(Train)', '2018-03-01', '2018-05-31', 'lightgreen'),
        ('Summer\n(Train)', '2018-06-01', '2018-08-31', 'yellow'),
        ('Autumn\n(TEST)', '2018-09-01', '2018-11-30', 'orange')
    ]

    for season, start, end, color in seasons:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        ax2.axvspan(start_date, end_date, alpha=0.15, color=color)
        mid_date = start_date + (end_date - start_date) / 2
        ax2.text(mid_date, 0.7, season, ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'cubist_timeline_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cubist_timeline_comparison.pdf', bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'cubist_timeline_comparison.png'}")
    plt.close()

def create_performance_drop_visualization(random_results, temporal_results, output_dir):
    """Create visualization showing the performance drop"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate percentage changes
    random_test = random_results[random_results['Set'] == 'Testing'].iloc[0]
    temporal_test = temporal_results[temporal_results['Set'] == 'Testing'].iloc[0]

    metrics = ['R2', 'RMSE', 'MAE', 'CV']

    # Calculate relative changes (positive = worse for RMSE/MAE/CV, negative = worse for R2)
    changes = []
    for metric in metrics:
        random_val = random_test[metric]
        temporal_val = temporal_test[metric]

        if metric == 'R2':
            # For R2, calculate drop
            change = ((temporal_val - random_val) / random_val) * 100
        else:
            # For others (RMSE, MAE, CV), calculate increase
            change = ((temporal_val - random_val) / random_val) * 100

        changes.append(change)

    # Create bar chart
    colors = ['red' if c < 0 else 'orange' for c in changes]
    bars = ax.barh(metrics, changes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels
    for i, (bar, change) in enumerate(zip(bars, changes)):
        width = bar.get_width()
        label = f'{change:+.1f}%'
        ax.text(width, bar.get_y() + bar.get_height()/2., label,
               ha='left' if width > 0 else 'right', va='center',
               fontsize=12, fontweight='bold')

    # Add vertical line at 0
    ax.axvline(0, color='black', linestyle='-', linewidth=2)

    # Formatting
    ax.set_xlabel('Percentage Change in Temporal Split vs Random Split',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=14, fontweight='bold')
    ax.set_title('Performance Change: Temporal Split vs Random Split\n(Negative is worse for R², Positive is worse for RMSE/MAE/CV)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Add annotation
    ax.text(0.02, 0.98,
            'Red bars = Performance degradation\nOrange bars = Error metrics increased',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'cubist_performance_drop.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cubist_performance_drop.pdf', bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'cubist_performance_drop.png'}")
    plt.close()

def create_summary_dashboard(random_results, temporal_results, output_dir):
    """Create a comprehensive summary dashboard"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('CUBIST Model Comparison Dashboard\nRandom Split (Paper Replication) vs Temporal Split (Real-World)',
                 fontsize=20, fontweight='bold', y=0.98)

    # 1. R2 Comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    categories = ['Random Split', 'Temporal Split']
    train_r2 = [
        random_results[random_results['Set'] == 'Training']['R2'].values[0],
        temporal_results[temporal_results['Set'] == 'Training']['R2'].values[0]
    ]
    test_r2 = [
        random_results[random_results['Set'] == 'Testing']['R2'].values[0],
        temporal_results[temporal_results['Set'] == 'Testing']['R2'].values[0]
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax1.bar(x - width/2, train_r2, width, label='Training', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width/2, test_r2, width, label='Testing', color='#e74c3c', alpha=0.8)
    ax1.axhline(y=0.95, color='blue', linestyle='--', linewidth=2, label='Paper R²=0.95', alpha=0.7)

    for i, (tr, te) in enumerate(zip(train_r2, test_r2)):
        ax1.text(i - width/2, tr, f'{tr:.3f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, te, f'{te:.3f}', ha='center', va='bottom', fontweight='bold')

    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('R² Score Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # 2. Summary Stats Table (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')

    random_test = random_results[random_results['Set'] == 'Testing'].iloc[0]
    temporal_test = temporal_results[temporal_results['Set'] == 'Testing'].iloc[0]

    table_data = [
        ['Metric', 'Random', 'Temporal', 'Change'],
        ['R²', f"{random_test['R2']:.4f}", f"{temporal_test['R2']:.4f}",
         f"{((temporal_test['R2'] - random_test['R2']) / random_test['R2'] * 100):+.1f}%"],
        ['RMSE', f"{random_test['RMSE']:.2f}", f"{temporal_test['RMSE']:.2f}",
         f"{((temporal_test['RMSE'] - random_test['RMSE']) / random_test['RMSE'] * 100):+.1f}%"],
        ['MAE', f"{random_test['MAE']:.2f}", f"{temporal_test['MAE']:.2f}",
         f"{((temporal_test['MAE'] - random_test['MAE']) / random_test['MAE'] * 100):+.1f}%"],
    ]

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax2.set_title('Test Set Performance', fontweight='bold', fontsize=14, pad=20)

    # 3. RMSE Comparison (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    train_rmse = [
        random_results[random_results['Set'] == 'Training']['RMSE'].values[0],
        temporal_results[temporal_results['Set'] == 'Training']['RMSE'].values[0]
    ]
    test_rmse = [
        random_results[random_results['Set'] == 'Testing']['RMSE'].values[0],
        temporal_results[temporal_results['Set'] == 'Testing']['RMSE'].values[0]
    ]

    ax3.bar(x - width/2, train_rmse, width, label='Training', color='#3498db', alpha=0.8)
    ax3.bar(x + width/2, test_rmse, width, label='Testing', color='#e67e22', alpha=0.8)
    ax3.set_ylabel('RMSE (bikes)', fontweight='bold')
    ax3.set_title('RMSE Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. MAE Comparison (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    train_mae = [
        random_results[random_results['Set'] == 'Training']['MAE'].values[0],
        temporal_results[temporal_results['Set'] == 'Training']['MAE'].values[0]
    ]
    test_mae = [
        random_results[random_results['Set'] == 'Testing']['MAE'].values[0],
        temporal_results[temporal_results['Set'] == 'Testing']['MAE'].values[0]
    ]

    ax4.bar(x - width/2, train_mae, width, label='Training', color='#9b59b6', alpha=0.8)
    ax4.bar(x + width/2, test_mae, width, label='Testing', color='#8e44ad', alpha=0.8)
    ax4.set_ylabel('MAE (bikes)', fontweight='bold')
    ax4.set_title('MAE Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # 5. CV Comparison (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    train_cv = [
        random_results[random_results['Set'] == 'Training']['CV'].values[0],
        temporal_results[temporal_results['Set'] == 'Training']['CV'].values[0]
    ]
    test_cv = [
        random_results[random_results['Set'] == 'Testing']['CV'].values[0],
        temporal_results[temporal_results['Set'] == 'Testing']['CV'].values[0]
    ]

    ax5.bar(x - width/2, train_cv, width, label='Training', color='#16a085', alpha=0.8)
    ax5.bar(x + width/2, test_cv, width, label='Testing', color='#1abc9c', alpha=0.8)
    ax5.set_ylabel('CV (%)', fontweight='bold')
    ax5.set_title('Coefficient of Variation', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, fontsize=9)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # 6. Key Insights Text (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    insights_text = f"""
KEY FINDINGS:

1. Random Split (Paper Replication):
   • Test R² = {random_test['R2']:.4f} - Closely matches published result (R² = 0.95)
   • Validates implementation correctness
   • Training and testing samples scattered throughout all 12 months

2. Temporal Split (Real-World Scenario):
   • Test R² = {temporal_test['R2']:.4f} - Significantly lower than random split
   • Tests on completely unseen Autumn season (Sep-Nov 2018)
   • Training covers Winter, Spring, Summer only (Dec 2017 - Aug 2018)

3. Performance Gap:
   • R² drops by {((temporal_test['R2'] - random_test['R2']) / random_test['R2'] * 100):.1f}% in temporal split
   • RMSE increases by {((temporal_test['RMSE'] - random_test['RMSE']) / random_test['RMSE'] * 100):.1f}%
   • This gap represents the difference between research benchmarks and real-world deployment

4. Implications:
   • Random split overestimates real-world performance
   • Temporal validation is critical for time series forecasting
   • Model struggles with unseen seasonal patterns
   • Recommendation: Train on full year minimum, retrain periodically in production
    """

    ax6.text(0.05, 0.95, insights_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(output_dir / 'cubist_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'cubist_summary_dashboard.pdf', bbox_inches='tight')
    print(f"[OK] Saved: {output_dir / 'cubist_summary_dashboard.png'}")
    plt.close()

def main():
    """Main execution"""
    print("="*80)
    print("CUBIST MODEL COMPARISON VISUALIZATION")
    print("="*80)
    print()

    # Create output directory
    output_dir = Path('reports/figures/comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Load results
    print("Loading results...")
    random_results, temporal_results, comparison_data = load_results()
    print("[OK] Results loaded")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    print()

    create_r2_comparison_chart(random_results, temporal_results, output_dir)
    create_all_metrics_comparison(random_results, temporal_results, output_dir)
    create_test_performance_comparison(random_results, temporal_results, output_dir)
    create_timeline_visualization(output_dir)
    create_performance_drop_visualization(random_results, temporal_results, output_dir)
    create_summary_dashboard(random_results, temporal_results, output_dir)

    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print()
    print(f"All visualizations saved to: {output_dir}")
    print()
    print("Generated files:")
    print("  1. cubist_r2_comparison.png/pdf - R² comparison chart")
    print("  2. cubist_all_metrics_comparison.png/pdf - All metrics (R², RMSE, MAE, CV)")
    print("  3. cubist_test_performance_comparison.png/pdf - Test set focus")
    print("  4. cubist_timeline_comparison.png/pdf - Train/test split timeline")
    print("  5. cubist_performance_drop.png/pdf - Performance degradation analysis")
    print("  6. cubist_summary_dashboard.png/pdf - Comprehensive dashboard")
    print()

if __name__ == "__main__":
    main()

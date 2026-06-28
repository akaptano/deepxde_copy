#!/usr/bin/env python3
"""
Plot architecture comparison graphs similar to Jang et al. paper figures.

Generates:
1. Network Depth comparison (like Fig. 12)
2. Network Width comparison (like Fig. 13)

Usage:
    # Compare runs in a directory
    python plot_architecture_comparison.py --runs_dir /path/to/saved_models_new
    
    # Compare specific runs
    python plot_architecture_comparison.py --run_dirs dir1 dir2 dir3
    
    # Filter by experiment name
    python plot_architecture_comparison.py --runs_dir /path/to/models --filter "arch_scan"

"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Set up matplotlib style similar to the paper
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2


def load_run_data(run_dir: str) -> Optional[dict]:
    """Load configuration and loss history from a run directory."""
    config_path = os.path.join(run_dir, 'config.json')
    loss_path = os.path.join(run_dir, 'loss_history.json')
    stats_path = os.path.join(run_dir, 'final_stats.json')
    
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        history = []
        adam_end_step = 0
        if os.path.exists(loss_path):
            with open(loss_path, 'r') as f:
                loss_data = json.load(f)
                history = loss_data.get('history', [])
                adam_end_step = loss_data.get('adam_end_step', 0)
        
        final_stats = {}
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                final_stats = json.load(f)
        
        return {
            'dir': run_dir,
            'config': config,
            'history': history,
            'adam_end_step': adam_end_step,
            'final_stats': final_stats,
        }
    except Exception as e:
        print(f"Warning: Could not load {run_dir}: {e}")
        return None


def find_runs(runs_dir: str, filter_pattern: str = None) -> List[dict]:
    """Find all valid runs in a directory."""
    runs = []
    
    for item in os.listdir(runs_dir):
        item_path = os.path.join(runs_dir, item)
        if os.path.isdir(item_path):
            if filter_pattern and filter_pattern not in item:
                continue
            
            data = load_run_data(item_path)
            if data:
                runs.append(data)
                print(f"Loaded: {item} (depth={data['config'].get('depth')}, width={data['config'].get('width')})")
    
    return runs


def group_runs_by_param(runs: List[dict], group_by: str, fixed_params: dict = None) -> Dict[int, List[dict]]:
    """Group runs by a parameter value, optionally filtering by fixed params."""
    groups = defaultdict(list)
    
    for run in runs:
        config = run['config']
        
        # Check fixed params
        if fixed_params:
            skip = False
            for key, value in fixed_params.items():
                if config.get(key) != value:
                    skip = True
                    break
            if skip:
                continue
        
        param_value = config.get(group_by)
        if param_value is not None:
            groups[param_value].append(run)
    
    return dict(groups)


def plot_depth_comparison(runs: List[dict], output_path: str, 
                          fixed_width: int = None, num_params: int = None):
    """
    Create depth comparison plot similar to Fig. 12 in the paper.
    """
    # Filter runs
    fixed = {}
    if fixed_width:
        fixed['width'] = fixed_width
    if num_params:
        fixed['num_params'] = num_params
    
    groups = group_runs_by_param(runs, 'depth', fixed)
    
    if not groups:
        print("No runs found for depth comparison!")
        return
    
    # Sort depths
    depths = sorted(groups.keys())
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(depths))))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, depth in enumerate(depths):
        runs_for_depth = groups[depth]
        
        # Use the first run (or could average multiple runs)
        run = runs_for_depth[0]
        history = run['history']
        adam_end = run['adam_end_step']
        
        if not history:
            continue
        
        steps = [h['step'] for h in history]
        losses = [h['train_loss_total'] for h in history]
        
        label = f'{depth:02d}' if depth < 10 else str(depth)
        ax.plot(steps, losses, color=colors[i], label=label, alpha=0.9)
    
    # Add vertical line for Adam/L-BFGS transition
    if adam_end > 0:
        ax.axvline(x=adam_end, color='gray', linestyle='--', alpha=0.5)
        ax.text(adam_end + 50, ax.get_ylim()[1] * 0.9, 'L-BFGS-B', fontsize=10, alpha=0.7)
        ax.text(adam_end - 200, ax.get_ylim()[1] * 0.9, 'Adam', fontsize=10, alpha=0.7)
    
    ax.set_yscale('log')
    ax.set_xlabel('# Steps')
    ax.set_ylabel('Loss Term')
    
    title = 'Network Depth'
    if fixed_width:
        title += f' (width={fixed_width})'
    if num_params:
        title += f' (params={num_params})'
    ax.set_title(title)
    
    ax.legend(title='Depth', loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_width_comparison(runs: List[dict], output_path: str,
                          fixed_depth: int = None, num_params: int = None):
    """
    Create width comparison plot similar to Fig. 13 in the paper.
    """
    # Filter runs
    fixed = {}
    if fixed_depth:
        fixed['depth'] = fixed_depth
    if num_params:
        fixed['num_params'] = num_params
    
    groups = group_runs_by_param(runs, 'width', fixed)
    
    if not groups:
        print("No runs found for width comparison!")
        return
    
    # Sort widths
    widths = sorted(groups.keys())
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(widths))))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, width in enumerate(widths):
        runs_for_width = groups[width]
        
        run = runs_for_width[0]
        history = run['history']
        adam_end = run['adam_end_step']
        
        if not history:
            continue
        
        steps = [h['step'] for h in history]
        losses = [h['train_loss_total'] for h in history]
        
        ax.plot(steps, losses, color=colors[i], label=str(width), alpha=0.9)
    
    # Add vertical line for Adam/L-BFGS transition
    if adam_end > 0:
        ax.axvline(x=adam_end, color='gray', linestyle='--', alpha=0.5)
        ax.text(adam_end + 50, ax.get_ylim()[1] * 0.9, 'L-BFGS-B', fontsize=10, alpha=0.7)
        ax.text(adam_end - 200, ax.get_ylim()[1] * 0.9, 'Adam', fontsize=10, alpha=0.7)
    
    ax.set_yscale('log')
    ax.set_xlabel('# Steps')
    ax.set_ylabel('Loss Term')
    
    title = 'Network Width'
    if fixed_depth:
        title += f' (depth={fixed_depth})'
    if num_params:
        title += f' (params={num_params})'
    ax.set_title(title)
    
    ax.legend(title='Width', loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_final_loss_bars(runs: List[dict], output_path: str):
    """Create bar chart of final losses by architecture."""
    
    data = []
    for run in runs:
        config = run['config']
        stats = run.get('final_stats', {})
        
        if 'final_train_loss' in stats:
            data.append({
                'depth': config.get('depth', 0),
                'width': config.get('width', 0),
                'loss': stats['final_train_loss'],
                'psi_min': stats.get('psi_min', 0),
                'has_zero_crossing': stats.get('has_zero_crossing', False),
            })
    
    if not data:
        print("No final stats found for bar chart!")
        return
    
    # Sort by depth then width
    data.sort(key=lambda x: (x['depth'], x['width']))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Final loss
    ax1 = axes[0]
    labels = [f"d{d['depth']}_w{d['width']}" for d in data]
    losses = [d['loss'] for d in data]
    colors = ['green' if d['has_zero_crossing'] else 'red' for d in data]
    
    bars = ax1.bar(range(len(labels)), losses, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Final Training Loss')
    ax1.set_title('Final Loss by Architecture\n(green = proper ψ zero crossing)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ψ min values
    ax2 = axes[1]
    psi_mins = [d['psi_min'] for d in data]
    colors2 = ['green' if d['has_zero_crossing'] else 'red' for d in data]
    
    bars2 = ax2.bar(range(len(labels)), psi_mins, color=colors2, alpha=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('ψ minimum value')
    ax2.set_title('ψ Depth by Architecture\n(more negative = better)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def create_summary_table(runs: List[dict], output_path: str):
    """Create a summary table of all runs."""
    
    rows = []
    for run in runs:
        config = run['config']
        stats = run.get('final_stats', {})
        
        rows.append({
            'Depth': config.get('depth', '-'),
            'Width': config.get('width', '-'),
            'Params': config.get('num_params', '-'),
            'Final Loss': f"{stats.get('final_train_loss', 0):.2e}",
            'ψ min': f"{stats.get('psi_min', 0):.4f}",
            'ψ max': f"{stats.get('psi_max', 0):.4f}",
            'Zero Cross': '✓' if stats.get('has_zero_crossing') else '✗',
            'Steps': stats.get('total_steps', '-'),
        })
    
    # Sort by depth then width
    rows.sort(key=lambda x: (x['Depth'] if isinstance(x['Depth'], int) else 0,
                             x['Width'] if isinstance(x['Width'], int) else 0))
    
    # Write as markdown table
    with open(output_path, 'w') as f:
        f.write("# Architecture Comparison Summary\n\n")
        
        if rows:
            headers = list(rows[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            for row in rows:
                f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot architecture comparison')
    parser.add_argument('--runs_dir', type=str, 
                        default='/scratch/yx3044/Projects/deepxde_copy/gs_2d_surrogate/saved_models_new',
                        help='Directory containing run folders')
    parser.add_argument('--run_dirs', nargs='+', type=str, default=None,
                        help='Specific run directories to compare')
    parser.add_argument('--filter', type=str, default='arch_scan',
                        help='Filter pattern for run directories')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--fixed_width', type=int, default=None,
                        help='Fixed width for depth comparison')
    parser.add_argument('--fixed_depth', type=int, default=None,
                        help='Fixed depth for width comparison')
    parser.add_argument('--num_params', type=int, default=None,
                        help='Filter by num_params')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.runs_dir, 'architecture_comparison')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load runs
    print("Loading runs...")
    if args.run_dirs:
        runs = []
        for d in args.run_dirs:
            data = load_run_data(d)
            if data:
                runs.append(data)
                print(f"Loaded: {d}")
    else:
        runs = find_runs(args.runs_dir, args.filter)
    
    if not runs:
        print("No valid runs found!")
        return
    
    print(f"\nFound {len(runs)} runs")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    
    # Depth comparison
    plot_depth_comparison(
        runs, 
        os.path.join(output_dir, 'depth_comparison.png'),
        fixed_width=args.fixed_width,
        num_params=args.num_params
    )
    
    # Width comparison
    plot_width_comparison(
        runs,
        os.path.join(output_dir, 'width_comparison.png'),
        fixed_depth=args.fixed_depth,
        num_params=args.num_params
    )
    
    # Final loss bar chart
    plot_final_loss_bars(runs, os.path.join(output_dir, 'final_loss_comparison.png'))
    
    # Summary table
    create_summary_table(runs, os.path.join(output_dir, 'summary.md'))
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

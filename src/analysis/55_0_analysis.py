#!/usr/bin/env python3
"""
Analysis and plotting tool for network datasets.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # Import pearsonr for correlation and p-value calculation
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import warnings

def _maybe_log(df, cols, use_log=True):
    if not use_log:
        return df[cols].copy()
    return np.log1p(df[cols].replace([np.inf, -np.inf], np.nan)).fillna(0)


def _make_outpath(outpath, basename):
    p = Path(outpath)
    if basename:
        # Check if the path already contains the basename to avoid duplication
        path_str = str(p)
        if basename not in path_str:
            p = p.with_name(f"{basename}_{p.name}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def plot_violin_reg_and_histgrid(pop_stats_df,
                                 outpath="analysis_out/summary_violin_reg_hist.png",
                                 cols=None,
                                 use_log=True,
                                 basename=None,
                                 dpi=150):
    """
    Plot violin plots with regression lines and histograms in a grid.
    
    Parameters:
      - pop_stats_df: DataFrame with population statistics
      - outpath: output path for the plot
      - cols: columns to plot (if None, use defaults)
      - use_log: whether to use log scale
      - basename: base name for the plot
    """
    # Apply log transformation if requested
    df = pop_stats_df.copy()
    if use_log:
        log_cols = ["cont_out", "cont_in", "elec_out", "elec_in", "exc_inputs", "inh_inputs"]
        for col in log_cols:
            if col in df.columns:
                # Handle negative values which can't be log transformed
                df[col] = np.log1p(df[col].clip(lower=0))
    
    # Handle case where there's no region information
    if "region" in df.columns and not df["region"].isnull().all() and len(df["region"].unique()) > 1:
        has_regions = True
        # Create a single violin plot with region grouping
        unique_regions = df["region"].unique()
    else:
        df["region"] = "All populations"
        has_regions = False
        unique_regions = ["All populations"]
    
    # Create figure with subplots in a 1x3 layout
    n_cols = 3
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6))
    fig.suptitle(f"{'Log ' if use_log else ''}Distributions and Regressions by Region" + 
                 (f" - {basename}" if basename else ""), fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_cols * n_rows > 1 else [axes]
    
    # Define the three paired column combinations with updated color tone
    paired_cols = [
        {"left": "cont_out", "right": "cont_in", "title": "Chemical (cont_out | cont_in)", 
         "left_color": "skyblue", "right_color": "salmon", 
         "left_edge": "blue", "right_edge": "red"},
        {"left": "elec_out", "right": "elec_in", "title": "Electrical (elec_out | elec_in)",
         "left_color": "lightgreen", "right_color": "plum",
         "left_edge": "green", "right_edge": "purple"},
        {"left": "exc_inputs", "right": "inh_inputs", "title": "Inputs (exc_inputs | inh_inputs)",
         "left_color": "gold", "right_color": "lightcoral",
         "left_edge": "orange", "right_edge": "crimson"}
    ]
    
    # Create each paired violin plot
    for i, col_pair in enumerate(paired_cols):
        ax = axes_flat[i]
        left_col = col_pair["left"]
        right_col = col_pair["right"]
        
        # Check if columns exist
        if left_col not in df.columns or right_col not in df.columns:
            ax.text(0.5, 0.5, f"Data not available\n{left_col} vs {right_col}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col_pair["title"])
            continue
        
        # Prepare data for violin plot in long-form
        plot_data = []
        
        # Collect all data for both left and right columns
        all_values = []
        all_types = []
        all_regions = []
        
        left_data = df[left_col].dropna()
        right_data = df[right_col].dropna()
        
        # Add left data
        for idx, val in left_data.items():
            all_values.append(val)
            all_types.append(left_col)
            all_regions.append(df.loc[idx, "region"])
        
        # Add right data
        for idx, val in right_data.items():
            all_values.append(val)
            all_types.append(right_col)
            all_regions.append(df.loc[idx, "region"])
        
        # Check if we have data to plot
        if len(all_values) == 0:
            ax.text(0.5, 0.5, "No valid data", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col_pair["title"])
            continue
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'value': all_values,
            'type': all_types,
            'region': all_regions
        })
        
        # Create more readable labels for the types
        type_labels = {
            left_col: col_pair["title"].split(" (")[1].split(" | ")[0],  # Extract left label from title
            right_col: col_pair["title"].split(" | ")[1].split(")")[0]   # Extract right label from title
        }
        
        # Map the types to more readable labels
        plot_df['type_label'] = plot_df['type'].map(type_labels)
        
        # Create violin plot using seaborn
        palette = {left_col: col_pair["left_color"], right_col: col_pair["right_color"]}
        
        if has_regions and len(unique_regions) > 1:
            # With multiple regions - create grouped violin plot
            sns.violinplot(data=plot_df, x='region', 
                           y='value', hue='type', 
                          ax=ax, inner="quart", cut=0, 
                          palette=palette, 
                          gap=0.1, split=True)
            ax.set_xlabel("Region")
        else:
            # Without regions or single region - create violin plot with proper labels
            # Prepare data for violin plot
            plot_df_melted = plot_df.copy()
            plot_df_melted['type_label'] = plot_df_melted['type'].map(type_labels)
            
            # Create violin plot using seaborn
            violin_parts = sns.violinplot(data=plot_df_melted, hue='type_label', y='value', 
                                         ax=ax, inner="quart", cut=0, gap=0.1, split=True,legend=True,
                                        #  palette=palette, 
                                         )
            
            # Set x-axis labels
            ax.set_xlabel("Connection Type")
        
        ax.set_title(col_pair["title"], fontsize=12)
        ax.set_ylabel("Values" if not use_log else "Log Values")
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), ha="right")
        
        # Add legend only if needed
        if has_regions and len(unique_regions) > 1:
            ax.legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.get_legend().remove()
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.97])
        except Exception:
            pass
    final_path = _make_outpath(outpath, basename)
    fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return {"outpath": final_path}


def plot_connection_type_violins(pop_stats_df,
                                 outpath="analysis_out/summary_connection_violins.png",
                                 cols=None,
                                 use_log=True,
                                 basename=None,
                                 dpi=150):
    """
    Plot violin plots for different connection types with regression lines.
    
    Parameters:
      - pop_stats_df: DataFrame with population statistics
      - outpath: output path for the plot
      - cols: columns to plot (if None, use defaults)
      - use_log: whether to use log scale
      - basename: base name for the plot
    """
    # Apply log transformation if requested
    df = pop_stats_df.copy()
    if use_log:
        log_cols = ["cont_out", "cont_in", "elec_out", "elec_in", "exc_inputs", "inh_inputs"]
        for col in log_cols:
            if col in df.columns:
                # Handle negative values which can't be log transformed
                df[col] = np.log1p(df[col].clip(lower=0))
    
    # Handle case where there's no region information
    if "region" in df.columns and not df["region"].isnull().all() and len(df["region"].unique()) > 1:
        has_regions = True
        # Create a single violin plot with region grouping
        unique_regions = df["region"].unique()
    else:
        df["region"] = "All populations"
        has_regions = False
        unique_regions = ["All populations"]
    
    # Create figure with subplots in a 1x3 layout
    n_cols = 3
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6))
    fig.suptitle(f"{'Log ' if use_log else ''}Connection Type Distributions by Region" + 
                 (f" - {basename}" if basename else ""), fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_cols * n_rows > 1 else [axes]
    
    # Define the three paired column combinations with updated color tone
    paired_cols = [
        {"left": "cont_out", "right": "cont_in", "title": "Chemical (cont_out | cont_in)", 
         "left_color": "skyblue", "right_color": "salmon", 
         "left_edge": "blue", "right_edge": "red"},
        {"left": "elec_out", "right": "elec_in", "title": "Electrical (elec_out | elec_in)",
         "left_color": "lightgreen", "right_color": "plum",
         "left_edge": "green", "right_edge": "purple"},
        {"left": "exc_inputs", "right": "inh_inputs", "title": "Inputs (exc_inputs | inh_inputs)",
         "left_color": "gold", "right_color": "lightcoral",
         "left_edge": "orange", "right_edge": "crimson"}
    ]
    
    # Create each paired violin plot
    for i, col_pair in enumerate(paired_cols):
        ax = axes_flat[i]
        left_col = col_pair["left"]
        right_col = col_pair["right"]
        
        # Check if columns exist
        if left_col not in df.columns or right_col not in df.columns:
            ax.text(0.5, 0.5, f"Data not available\n{left_col} vs {right_col}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col_pair["title"])
            continue
        
        # Prepare data for violin plot in long-form
        plot_data = []
        
        # Collect all data for both left and right columns
        all_values = []
        all_types = []
        all_regions = []
        
        left_data = df[left_col].dropna()
        right_data = df[right_col].dropna()
        
        # Add left data
        for idx, val in left_data.items():
            all_values.append(val)
            all_types.append(left_col)
            all_regions.append(df.loc[idx, "region"])
        
        # Add right data
        for idx, val in right_data.items():
            all_values.append(val)
            all_types.append(right_col)
            all_regions.append(df.loc[idx, "region"])
        
        # Check if we have data to plot
        if len(all_values) == 0:
            ax.text(0.5, 0.5, "No valid data", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col_pair["title"])
            continue
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'value': all_values,
            'type': all_types,
            'region': all_regions
        })
        
        # Create more readable labels for the types
        type_labels = {
            left_col: col_pair["title"].split(" (")[1].split(" | ")[0],  # Extract left label from title
            right_col: col_pair["title"].split(" | ")[1].split(")")[0]   # Extract right label from title
        }
        
        # Map the types to more readable labels
        plot_df['type_label'] = plot_df['type'].map(type_labels)
        
        # Create violin plot using seaborn
        palette = {left_col: col_pair["left_color"], right_col: col_pair["right_color"]}
        
        if has_regions and len(unique_regions) > 1:
            # With multiple regions - create grouped violin plot
            sns.violinplot(data=plot_df, x='region', y='value', hue='type', 
                          ax=ax, inner="box", cut=0, palette=palette, gap=0.1, split=True)
            ax.set_xlabel("Region")
        else:
            # Without regions or single region - create violin plot with proper labels
            # Prepare data for violin plot
            plot_df_melted = plot_df.copy()
            plot_df_melted['type_label'] = plot_df_melted['type'].map(type_labels)
            
            # Create violin plot using seaborn
            violin_parts = sns.violinplot(data=plot_df_melted, x='type_label',hue='type_label', y='value', 
                                         ax=ax, inner="box", cut=0, palette=palette, legend=True)
            
            # Set x-axis labels
            ax.set_xlabel("Connection Type")
        
        ax.set_title(col_pair["title"], fontsize=12)
        ax.set_ylabel("Values" if not use_log else "Log Values")
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), ha="right")
        
        # Add legend only if needed
        if has_regions and len(unique_regions) > 1:
            ax.legend(title="Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.get_legend().remove()
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.97])
        except Exception:
            pass
    final_path = _make_outpath(outpath, basename)
    fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return {"outpath": final_path}


def plot_clustered_heatmap(pop_stats_df, cols=None, outpath="analysis_out/clustermap.png", 
                          method="spearman", basename=None, dpi=150):
    """
    Plot clustered heatmap of correlations.
    
    Parameters:
      - pop_stats_df: DataFrame with population statistics
      - cols: columns to plot (if None, use defaults)
      - outpath: output path for the plot
      - method: correlation method
      - basename: base name for the plot
    """
    if cols is None:
        cols = [
            "cont_exc_in",
            "cont_inh_in",
            "cont_exc_out",
            "cont_inh_out",
            "elec_exc_in",
            "elec_inh_in",
            "elec_exc_out",
            "elec_inh_out",
            "exc_inputs",
            "inh_inputs",
        ]
    
    # Check if all columns exist in the dataframe
    missing_cols = [col for col in cols if col not in pop_stats_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, using available columns")
        cols = [col for col in cols if col in pop_stats_df.columns]
    
    # Calculate correlation matrix
    df = pop_stats_df[cols]
    
    # Check if we have enough data and finite values for correlation calculation
    if df.empty or len(df.columns) < 2:
        print(f"Warning: Not enough data for correlation matrix in {basename}")
        return {"outpath": outpath}
    
    # Calculate correlation and handle non-finite values
    corr_df = df.corr(method=method)
    
    # Check if correlation matrix has non-finite values
    if not np.isfinite(corr_df).all().all():
        print(f"Warning: Non-finite values found in correlation matrix for {basename}, replacing with 0")
        corr_df = corr_df.fillna(0)
        # If all values are NaN or there are still non-finite values, skip clustering
        if not np.isfinite(corr_df).all().all():
            print(f"Warning: Still non-finite values after replacement, using regular heatmap for {basename}")
            # Create regular heatmap without clustering
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                       cbar_kws={"shrink": 0.8})
            plt.title(f"{method.capitalize()} Correlation Heatmap (No Clustering)" + 
                     (f" - {basename}" if basename else ""))
            
            final_path = _make_outpath(outpath, basename)
            plt.savefig(final_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            return {"outpath": final_path}
    
    # Create clustered heatmap with improved color scheme
    plt.figure(figsize=(10, 8))
    # Using a more contrasting blue color scheme with RdBu (red-blue) diverging colormap
    try:
        cg = sns.clustermap(corr_df, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                           cbar_kws={"shrink": 0.8})
        plt.title(f"{method.capitalize()} Correlation Clustermap" + 
                 (f" - {basename}" if basename else ""))
    except ValueError as e:
        if "condensed distance matrix must contain only finite values" in str(e):
            print(f"Warning: Clustering failed due to non-finite values, using regular heatmap for {basename}")
            # Create regular heatmap without clustering
            sns.heatmap(corr_df, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                       cbar_kws={"shrink": 0.8})
            plt.title(f"{method.capitalize()} Correlation Heatmap (No Clustering)" + 
                     (f" - {basename}" if basename else ""))
        else:
            raise e
    
    final_path = _make_outpath(outpath, basename)
    plt.savefig(final_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return {"outpath": final_path}


def plot_combined_heatmaps(pop_stats_df, cols=None, outpath="analysis_out/combined_heatmaps.png", 
                          methods=["spearman", "pearson"], basename=None, dpi=150):
    """
    Plot combined heatmaps side-by-side for different correlation methods.
    
    Parameters:
      - pop_stats_df: DataFrame with population statistics
      - cols: columns to plot (if None, use defaults)
      - outpath: output path for the plot
      - methods: list of correlation methods to use
      - basename: base name for the plot
    """
    if cols is None:
        cols = [
            "cont_exc_in",
            "cont_inh_in",
            "cont_exc_out",
            "cont_inh_out",
            "elec_exc_in",
            "elec_inh_in",
            "elec_exc_out",
            "elec_inh_out",
            "exc_inputs",
            "inh_inputs",
        ]
    
    # Check if all columns exist in the dataframe
    missing_cols = [col for col in cols if col not in pop_stats_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, using available columns")
        cols = [col for col in cols if col in pop_stats_df.columns]
    
    # Check if we have enough data
    if len(cols) < 2:
        print(f"Warning: Not enough columns for correlation matrix in {basename}")
        return {"outpath": outpath}
    
    # Create side-by-side heatmaps
    fig, axes = plt.subplots(1, len(methods), figsize=(8*len(methods), 6))
    if len(methods) == 1:
        axes = [axes]  # Make it iterable
    
    df = pop_stats_df[cols]
    
    for i, method in enumerate(methods):
        # Calculate correlation and handle non-finite values
        try:
            corr_df = df.corr(method=method)
            
            # Check for non-finite values
            if not np.isfinite(corr_df).all().all():
                print(f"Warning: Non-finite values found in {method} correlation matrix for {basename}, replacing with 0")
                corr_df = corr_df.fillna(0)
            
            sns.heatmap(corr_df, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                        ax=axes[i], cbar_kws={"shrink": 0.8})
            axes[i].set_title(f"{method.capitalize()} Correlation")
        except Exception as e:
            print(f"Warning: Failed to compute {method} correlation for {basename}: {e}")
            axes[i].text(0.5, 0.5, f"Error computing\n{method} correlation", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"{method.capitalize()} Correlation")
    
    plt.suptitle(f"Combined Correlation Heatmaps" + (f" - {basename}" if basename else ""), 
                 fontsize=16)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
        except Exception:
            pass
    
    final_path = _make_outpath(outpath, basename)
    fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return {"outpath": final_path}


def plot_ei_scatter_with_stacked(pop_stats_df,
                                 outpath="analysis_out/ei_with_stacked.png",
                                 cols=None,
                                 counts_log1p=True,
                                 top_n=30,
                                 basename=None,
                                 dpi=150):
    """
    Plot E/I scatter with better visibility and legend.
    
    Parameters:
      - pop_stats_df: DataFrame with population statistics
      - outpath: output path for the plot
      - cols: columns to plot (if None, use defaults)
      - counts_log1p: whether to use log scale for counts
      - top_n: number of top populations to show
      - basename: base name for the plot
    """
    # Prepare data
    df = pop_stats_df.copy()
    
    # Apply log transformation if requested for all relevant columns
    columns_to_log = ["cont_in", "cont_out", "elec_in", "elec_out"]
    if counts_log1p:
        for col in columns_to_log:
            if col in df.columns:
                df[col] = np.log1p(df[col])
    
    # Calculate combined metrics
    if "cont_in" in df.columns and "elec_in" in df.columns:
        df["cont_elec_in"] = df["cont_in"] + df["elec_in"]
    if "cont_out" in df.columns and "elec_out" in df.columns:
        df["cont_elec_out"] = df["cont_out"] + df["elec_out"]
    
    # Define the plot configurations
    plot_configs = [
        {
            "title": "Synaptic (cont_in vs cont_out)" if cols is None else f"Custom ({cols[0]} vs {cols[1]})",
            "x_col": cols[0] if cols and len(cols) > 0 else "cont_in",
            "y_col": cols[1] if cols and len(cols) > 1 else "cont_out"
        },
        {
            "title": "Overall (cont+elec_in vs cont+elec_out)",
            "x_col": "cont_elec_in",
            "y_col": "cont_elec_out"
        },
        {
            "title": "E/I Input Ratio (Continuous)",
            "x_col": "cont_inh_in",
            "y_col": "cont_exc_in"
        },
        {
            "title": "E/I Output Ratio (Continuous)",
            "x_col": "cont_inh_out",
            "y_col": "cont_exc_out"
        },
        {
            "title": "E/I Ratio (Overall)",
            "x_col": "cont_elec_in",
            "y_col": "cont_elec_out"
        },
        {
            "title": "I/E Ratio (Overall)",
            "x_col": "cont_elec_out",
            "y_col": "cont_elec_in"
        },
    ]
    
    sort_column = None
    if "cont_in" in df.columns:
        sort_column = "cont_in"
    elif "elec_in" in df.columns:
        sort_column = "elec_in"
    elif "cont_elec_in" in df.columns:
        sort_column = "cont_elec_in"
    
    if sort_column:
        df["total_in"] = df[sort_column]
        df = df.nlargest(top_n, "total_in")

    if "cont_out" in df.columns:
        sort_column = "cont_out"
    elif "elec_out" in df.columns:
        sort_column = "elec_out"
    elif "cont_elec_out" in df.columns:
        sort_column = "cont_elec_out"
    
    if sort_column:
        df["total_out"] = df[sort_column]
        df = df.nlargest(top_n, "total_out")

    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[8, 8, 1, 1], hspace=0.4, wspace=0.3)
    
    # Main plots (first row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Additional plots (second row)
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Legend area below the plots
    legend_ax = fig.add_subplot(gs[2:, :])
    legend_ax.axis('off')
    
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    fig.suptitle(f"E/I Comparisons" + (f" - {basename}" if basename else ""), fontsize=16)
    
    # Color handling
    if "region" in df.columns:
        unique_regions = df["region"].unique()
        colors = plt.cm.tab20c(np.linspace(0, 1, len(unique_regions)))
        region_colors = dict(zip(unique_regions, colors))
    else:
        region_colors = None
    
    regressions = {}
    
    # Collect legend information
    legend_info = []
    
    # Create each subplot
    for i, config in enumerate(plot_configs):
        ax = axes[i]
        x_col = config["x_col"]
        y_col = config["y_col"]
        
        # Special handling for E/I ratio plots
        if "E/I" in config["title"] and "Ratio" in config["title"]:
            # Calculate E/I ratio on the fly for this plot
            if "exc_inputs" in df.columns and "inh_inputs" in df.columns:
                # Avoid division by zero
                ei_ratio = np.where(df["inh_inputs"] > 0, df["exc_inputs"] / df["inh_inputs"], np.nan)
                ax.scatter(df[x_col], ei_ratio, alpha=0.9, s=15, 
                          color='steelblue', edgecolors='white', linewidth=0.5)
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel("E/I Ratio", fontsize=12)
                ax.set_title(config["title"], fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # Add a horizontal line at ratio = 1 for reference
                ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        else:
            # Check if columns exist
            if x_col not in df.columns or y_col not in df.columns:
                ax.text(0.5, 0.5, f"Data not available\n{x_col} vs {y_col}", 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(config["title"])
                continue
                
            # Scatter plot with improved visibility
            if region_colors:
                for region in unique_regions:
                    region_data = df[df["region"] == region]
                    ax.scatter(region_data[x_col], region_data[y_col], 
                              alpha=0.9, s=15, label=region, 
                              color=region_colors[region], edgecolors='white', linewidth=0.5)
            else:
                ax.scatter(df[x_col], df[y_col], alpha=0.9, s=15, 
                          color='steelblue', edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(config["title"], fontsize=14)
            ax.grid(True, alpha=0.3)
        
        # Add regression line
        x_vals = df[x_col].values
        y_vals = df[y_col].values
        slope = None
        intercept = None
        corr = None
        p_value = None
        
        if len(x_vals) > 1:
            # Remove NaN values for regression
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            x_vals = x_vals[mask]
            y_vals = y_vals[mask]
            
            if len(x_vals) > 1:  # Check again after removing NaNs
                try:
                    # Use a more robust approach to handle poorly conditioned cases
                    with np.errstate(divide='ignore', invalid='ignore'):
                        # Check if we have enough unique values for a meaningful regression
                        if len(np.unique(x_vals)) > 1 and len(np.unique(y_vals)) > 1:
                            z = np.polyfit(x_vals, y_vals, 1, rcond=None)
                            p = np.poly1d(z)
                            ax.plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2)
                            slope, intercept = z
                            
                            # Calculate correlation and p-value using scipy
                            # Check for constant arrays to avoid warnings
                            if np.std(x_vals) == 0 or np.std(y_vals) == 0:
                                # One or both arrays are constant
                                corr = np.nan if np.std(x_vals) == 0 or np.std(y_vals) == 0 else 0.0
                                p_value = np.nan
                            else:
                                corr, p_value = pearsonr(x_vals, y_vals)
                            
                            ax.text(0.05, 0.95, f"R = {corr:.3f}\np = {p_value:.3e}", transform=ax.transAxes, 
                                    verticalalignment='top', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                        elif len(np.unique(x_vals)) == 1 and len(np.unique(y_vals)) == 1:
                            # Both arrays are constant
                            ax.text(0.05, 0.95, "Both arrays constant\nR = undefined", transform=ax.transAxes, 
                                    verticalalignment='top', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                        elif len(np.unique(x_vals)) == 1:
                            # Only x array is constant
                            ax.text(0.05, 0.95, "X values constant\nR = undefined", transform=ax.transAxes, 
                                    verticalalignment='top', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                        else:
                            # Only y array is constant
                            ax.text(0.05, 0.95, "Y values constant\nR = undefined", transform=ax.transAxes, 
                                    verticalalignment='top', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                except Exception as e:
                    # Handle any other regression errors
                    ax.text(0.05, 0.95, f"Regression error: {str(e)[:20]}", transform=ax.transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Add labels directly to plot if 10 or fewer populations
        if len(df) <= 10:
            for idx, row in df.iterrows():
                ax.annotate(row.name, (row[x_col], row[y_col]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Store regression results
        if corr is not None:
            regressions[f"{basename if basename else 'Dataset'} - {config['title']}"] = {
                "slope": slope,
                "intercept": intercept,
                "r": corr,
                "p": p_value,
                "n": len(x_vals)
            }
    
    # Improve legend for regions
    if region_colors:
        # Check if there are labeled artists before creating legend
        handles, labels = ax3.get_legend_handles_labels()
        if handles and labels:
            legend = ax3.legend(loc='lower right', fontsize=10, framealpha=0.9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
    
    
    final_path = _make_outpath(outpath, basename)
    fig.savefig(final_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return {"outpath": final_path, "regressions": regressions}


def aggregate_EI_stats(pop_stats_df):
    """
    Aggregate E/I statistics.
    
    Parameters:
      - pop_stats_df: DataFrame with population statistics
      
    Returns:
      dict with aggregated statistics
    """
    # Define the columns to aggregate
    col_groups = {
        "cont_out": ["cont_out"],
        "cont_in": ["cont_in"],
        "elec_out": ["elec_out"],
        "elec_in": ["elec_in"],
        "exc_inputs": ["exc_inputs"],
        "inh_inputs": ["inh_inputs"],
    }
    
    # Calculate totals
    col_groups["total_out"] = ["cont_out", "elec_out"]
    col_groups["total_in"] = ["cont_in", "elec_in"]
    
    results = {}
    for group_name, cols in col_groups.items():
        # Sum multiple columns
        values = pop_stats_df[cols].sum(axis=1).values
        
        # Calculate exc and inh statistics based on the group type
        exc_mean = 0
        exc_std = 0
        inh_mean = 0
        inh_std = 0
        
        # For connection columns, calculate E/I stats from the detailed breakdown
        if group_name == "cont_out":
            exc_values = pop_stats_df["cont_exc_out"].values
            inh_values = pop_stats_df["cont_inh_out"].values
            exc_mean = np.mean(exc_values)
            exc_std = np.std(exc_values)
            inh_mean = np.mean(inh_values)
            inh_std = np.std(inh_values)
        elif group_name == "cont_in":
            exc_values = pop_stats_df["cont_exc_in"].values
            inh_values = pop_stats_df["cont_inh_in"].values
            exc_mean = np.mean(exc_values)
            exc_std = np.std(exc_values)
            inh_mean = np.mean(inh_values)
            inh_std = np.std(inh_values)
        elif group_name == "elec_out":
            exc_values = pop_stats_df["elec_exc_out"].values
            inh_values = pop_stats_df["elec_inh_out"].values
            exc_mean = np.mean(exc_values)
            exc_std = np.std(exc_values)
            inh_mean = np.mean(inh_values)
            inh_std = np.std(inh_values)
        elif group_name == "elec_in":
            exc_values = pop_stats_df["elec_exc_in"].values
            inh_values = pop_stats_df["elec_inh_in"].values
            exc_mean = np.mean(exc_values)
            exc_std = np.std(exc_values)
            inh_mean = np.mean(inh_values)
            inh_std = np.std(inh_values)
        elif group_name == "total_out":
            # Combine chemical and electrical excitatory outputs
            exc_values = pop_stats_df[["cont_exc_out", "elec_exc_out"]].sum(axis=1).values
            inh_values = pop_stats_df[["cont_inh_out", "elec_inh_out"]].sum(axis=1).values
            exc_mean = np.mean(exc_values)
            exc_std = np.std(exc_values)
            inh_mean = np.mean(inh_values)
            inh_std = np.std(inh_values)
        elif group_name == "total_in":
            # Combine chemical and electrical excitatory inputs
            exc_values = pop_stats_df[["cont_exc_in", "elec_exc_in"]].sum(axis=1).values
            inh_values = pop_stats_df[["cont_inh_in", "elec_inh_in"]].sum(axis=1).values
            exc_mean = np.mean(exc_values)
            exc_std = np.std(exc_values)
            inh_mean = np.mean(inh_values)
            inh_std = np.std(inh_values)
        elif group_name == "exc_inputs":
            exc_mean = np.mean(values)
            exc_std = np.std(values)
        elif group_name == "inh_inputs":
            inh_mean = np.mean(values)
            inh_std = np.std(values)
        
        results[group_name] = {
            "exc_mean": exc_mean,
            "exc_std": exc_std,
            "inh_mean": inh_mean,
            "inh_std": inh_std,
            "total": np.sum(values)
        }
    
    return results


def load_dataset(basename, indir="analysis_out"):
    indir = Path(indir)
    
    # Load CSV files
    pops_df = pd.read_csv(indir / f"{basename}_populations.csv", index_col=0)
    pop_stats_df = pd.read_csv(indir / f"{basename}_population_stats.csv", index_col=0)
    
    # Strip whitespace from column names
    pops_df.columns = pops_df.columns.str.strip()
    pop_stats_df.columns = pop_stats_df.columns.str.strip()
    
    # Load JSON summary
    with open(indir / f"{basename}_summary.json", "r") as fh:
        summary = json.load(fh)
        
    dataset = {
        "pops_df": pops_df,
        "pop_stats_df": pop_stats_df,
        "summary": summary
    }
    return dataset


def find_available_datasets(indir="analysis_out"):
    indir_path = Path(indir)
    csv_files = indir_path.glob("*_populations.csv")
    basenames = [f.name.replace("_populations.csv", "") for f in csv_files]
    return basenames


def load_dataset_safe(basename, indir="analysis_out"):
    """
    Load dataset with error handling.
    
    Parameters:
      - basename: base name used when saving the files (required)
      - indir: directory where files are stored
      
    Returns:
      dict with keys: pops_df, pop_stats_df, summary or None if files not found
    """
    try:
        return load_dataset(basename, indir)
    except FileNotFoundError as e:
        print(f"Dataset with basename '{basename}' not found in {indir}")
        print(f"Available datasets: {find_available_datasets(indir)}")
        return None
    except Exception as e:
        print(f"Error loading dataset '{basename}': {e}")
        return None


def process_multiple_datasets(indir="analysis_out", outdir="plots"):
    """
    Process all available datasets in a directory.
    
    Parameters:
      - indir: directory where files are stored
      - outdir: directory where plots should be saved
      
    Returns:
      dict with results for each dataset
    """
    # Find all available datasets
    basenames = find_available_datasets(indir)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(basenames)} datasets to process:")
    for bn in basenames:
        print(f"  - {bn}")
    
    results = {}
    
    for basename in basenames:
        try:
            print(f"\nProcessing dataset: {basename}")
            dataset = load_dataset(basename, indir=indir)
            
            # Generate all plots
            plot_connection_type_violins(
                dataset["pop_stats_df"], 
                outpath=str(outdir_path / f"{basename}_summary_connection_violins.png"), 
                use_log=True, 
                basename=basename
            )
            
            plot_violin_reg_and_histgrid(
                dataset["pop_stats_df"], 
                outpath=str(outdir_path / f"{basename}_summary_violin_reg_hist.png"), 
                use_log=True, 
                basename=basename
            )
            
            # Generate both types of heatmaps
            plot_clustered_heatmap(
                dataset["pop_stats_df"], 
                outpath=str(outdir_path / f"{basename}_clustermap.png"), 
                basename=basename
            )
            
            plot_combined_heatmaps(
                dataset["pop_stats_df"], 
                outpath=str(outdir_path / f"{basename}_combined_heatmaps.png"), 
                basename=basename
            )
            
            plot_ei_scatter_with_stacked(
                dataset["pop_stats_df"], 
                outpath=str(outdir_path / f"{basename}_ei_with_stacked.png"), 
                basename=basename
            )
            
            results[basename] = {
                "status": "success",
                "plots_dir": str(outdir_path)
            }
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            results[basename] = {
                "status": "error",
                "error": str(e)
            }
    
    return results


def main():
    """
    Main function to run analysis independently by loading data from analysis_out directory.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis on previously generated network data")
    parser.add_argument("--basename", "-b", help="Base name for the files (if not provided, all datasets will be processed)")
    parser.add_argument("--indir", "-i", default="analysis_out", help="Input directory with data files")
    parser.add_argument("--outdir", "-o", default="plots", help="Output directory for plots")
    parser.add_argument("--all", "-a", action="store_true", help="Process all datasets in the directory (same as not providing basename)")
    
    args = parser.parse_args()
    
    # If --all is specified or no basename is provided, process all datasets
    if args.all or not args.basename:
        print(f"Processing all datasets in '{args.indir}'")
        results = process_multiple_datasets(indir=args.indir, outdir=args.outdir)
        print("\nAll datasets processed!")
        for basename, result in results.items():
            print(f"  {basename}: {result}")
    else:
        print(f"Loading dataset with basename '{args.basename}' from '{args.indir}'")
        dataset = load_dataset(args.basename, indir=args.indir)
        
        print("Generating plots...")
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        plot_connection_type_violins(
            dataset["pop_stats_df"], 
            outpath=str(outdir / f"{args.basename}_summary_connection_violins.png"), 
            use_log=True, 
            basename=args.basename
        )
        
        plot_violin_reg_and_histgrid(
            dataset["pop_stats_df"], 
            outpath=str(outdir / f"{args.basename}_summary_violin_reg_hist.png"), 
            use_log=True, 
            basename=args.basename
        )
        
        plot_clustered_heatmap(
            dataset["pop_stats_df"], 
            outpath=str(outdir / f"{args.basename}_clustermap.png"), 
            basename=args.basename
        )
        
        plot_combined_heatmaps(
            dataset["pop_stats_df"], 
            outpath=str(outdir / f"{args.basename}_combined_heatmaps.png"), 
            basename=args.basename
        )
        
        plot_ei_scatter_with_stacked(
            dataset["pop_stats_df"], 
            outpath=str(outdir / f"{args.basename}_ei_with_stacked.png"), 
            basename=args.basename
        )
        
        print(f"All plots saved to '{outdir}' directory")


if __name__ == "__main__":
    main()



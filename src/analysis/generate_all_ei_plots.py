"""Generate EI scatter variants and violin plots for all population CSVs.
Produces per-dataset subfolders under `plots/` and a simple `plots/index.html` gallery.
"""
from pathlib import Path
import importlib.util
import pandas as pd

import seaborn as sns
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / '55_0_analysis.py'
OUTROOT = ROOT / 'plots'
CSV_GLOB = 'analysis_out/*_population_stats.csv'

# gallery styling
sns.set_style('whitegrid')
sns.set_context('talk')

# load module
spec = importlib.util.spec_from_file_location('a55', str(ANALYSIS))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

OUTROOT.mkdir(exist_ok=True)

index_lines = ["<html><head><meta charset=\"utf-8\"><title>EI Plots Gallery</title></head><body>",
               "<h1>EI Plots Gallery</h1>"]


def _fallback_clustered_heatmap(df, outpath, method='spearman'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    cols = [c for c in df.columns if df[c].dtype.kind in 'fi']
    if len(cols) < 2:
        return {"outpath": outpath}
    corr = df[cols].corr(method=method).fillna(0)
    try:
        # larger clustermap and readable annotation font
        cg = sns.clustermap(corr, annot=True, annot_kws={'fontsize':4}, figsize=(14, 12),
                            cmap='coolwarm',center=0, vmin=-1, vmax=1, cbar_kws={"shrink":0.7}, linewidths=0.5)
        # improve tick label readability
        cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), fontsize=7)
        cg.fig.suptitle(f"{method.capitalize()} Correlation Clustermap", fontsize=12)
        cg.savefig(outpath, dpi=200)
        plt.close()
    except Exception:
        plt.figure(figsize=(14, 12))
        ax = plt.gca()
        sns.heatmap(corr, annot=True, annot_kws={'fontsize':4}, cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, cbar_kws={"shrink":0.7}, linewidths=0.5)
        plt.xticks(rotation=45, ha='right', fontsize=5)
        plt.yticks(fontsize=5)
        plt.title(f"{method.capitalize()} Correlation Heatmap", fontsize=12)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
    return {"outpath": outpath}


def _fallback_combined_heatmaps(df, outpath, methods=("spearman", "pearson")):
    import seaborn as sns
    import matplotlib.pyplot as plt
    cols = [c for c in df.columns if df[c].dtype.kind in 'fi']
    if len(cols) < 2:
        return {"outpath": outpath}
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(10*n, 8))
    if n == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        corr = df[cols].corr(method=method).fillna(0)
        sns.heatmap(corr, annot=True, annot_kws={'fontsize':4}, cmap='coolwarm', 
                    center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink":0.7}, linewidths=0.4)
        ax.set_title(f"{method.capitalize()} Correlation", fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=5)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return {"outpath": outpath}


# Removed _make_all_across_summary: combined heatmaps are sufficient, no separate all_across figure


def _combine_ei_images(dataset_dir, stem, outpath, parts=None):
    """Combine several EI variant PNGs side-by-side into a single PNG.
    `parts` is a list of Path objects or strings relative to dataset_dir.
    """
    from PIL import Image
    from pathlib import Path
    ds = Path(dataset_dir)
    if parts is None:
        parts = [f"{stem}_ei_highlight_kde.png", f"{stem}_ei_all_err_hexbin.png", f"{stem}_ei_no_density.png"]
    imgs = []
    for p in parts:
        pth = ds / p
        if pth.exists():
            try:
                imgs.append(Image.open(pth).convert('RGBA'))
            except Exception:
                try:
                    target1 = dataset_dir / f"{stem}_ei_highlight_kde.png"
                    # current analysis API uses counts_log1p; keep only the improved highlight+kde variant
                    r1 = mod.plot_ei_scatter_with_stacked(df, outpath=str(target1), basename=None, counts_log1p=True, top_n=10)
                    saved.append(r1['outpath'])
                except Exception as e:
                    print('Error highlight+kde', e)


def _compute_stats_for_errorbars(df, col, comp_map=None):
    import numpy as np
    import pandas as pd
    if comp_map is None:
        comp_map = {
            "total_out": ["cont_out", "elec_out"],
            "total_in": ["cont_in", "elec_in"],
            "cont_elec_in": ["cont_in", "elec_in"],
            "cont_elec_out": ["cont_out", "elec_out"],
            "inputs": ["exc_inputs", "inh_inputs"],
        }
    if col in comp_map:
        cols = [c for c in comp_map[col] if c in df.columns]
        if len(cols) == 0:
            return pd.Series(np.nan, index=df.index), pd.Series(0.0, index=df.index)
        vals = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        mean_sum = np.nansum(vals, axis=1)
        std_vals = np.nanstd(vals, axis=1)
        return pd.Series(mean_sum, index=df.index), pd.Series(std_vals, index=df.index)
    else:
        if col in df.columns:
            s = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            return s, pd.Series(0.0, index=df.index)
        else:
            return pd.Series(np.nan, index=df.index), pd.Series(0.0, index=df.index)


def _make_ei_errorbars(df, outpath, top_n=30, stem=None):
    """Create E/I errorbar figure matching scripts/plot_ei_with_errorbars.py layout.
    Produces 2 rows x 3 columns of scatter plots and a legend row below.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # compute derived totals if missing
    if 'cont_elec_in' not in df.columns and 'cont_in' in df.columns and 'elec_in' in df.columns:
        df['cont_elec_in'] = df['cont_in'] + df['elec_in']
    if 'cont_elec_out' not in df.columns and 'cont_out' in df.columns and 'elec_out' in df.columns:
        df['cont_elec_out'] = df['cont_out'] + df['elec_out']

    plot_configs = [
        {"title": "Synaptic (cont_in vs cont_out)", "x_col": "cont_in", "y_col": "cont_out"},
        {"title": "Overall (cont+elec_in vs cont+elec_out)", "x_col": "cont_elec_in", "y_col": "cont_elec_out"},
        {"title": "E/I Input Ratio (Continuous)", "x_col": "cont_inh_in", "y_col": "cont_exc_in"},
        {"title": "E/I Output Ratio (Continuous)", "x_col": "cont_inh_out", "y_col": "cont_exc_out"},
        {"title": "E/I Ratio (Overall)", "x_col": "cont_elec_in", "y_col": "cont_elec_out"},
        {"title": "I/E Ratio (Overall)", "x_col": "cont_elec_out", "y_col": "cont_elec_in"},
    ]

    top_in_idx = set(df.nlargest(top_n, 'cont_in').index) if 'cont_in' in df.columns else set()
    top_out_idx = set(df.nlargest(top_n, 'cont_out').index) if 'cont_out' in df.columns else set()
    highlight_idx = top_in_idx.union(top_out_idx)

    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[9, 9, 0.9, 0.15], hspace=0.6, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, i]) for i in range(3)]
    fig.suptitle(f'E/I Comparisons - {stem}' if stem else 'E/I Comparisons', fontsize=16)

    import seaborn as sns

    for i, cfg in enumerate(plot_configs):
        ax = axes[i]
        xcol = cfg['x_col']
        ycol = cfg['y_col']

        x_mean, x_std = _compute_stats_for_errorbars(df, xcol)
        y_mean, y_std = _compute_stats_for_errorbars(df, ycol)

        mask = ~(np.isnan(x_mean) | np.isnan(y_mean))
        if np.count_nonzero(mask) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            continue

        indices = df.index.to_list()
        is_high = np.array([idx in highlight_idx for idx in indices])

        normal_idx = np.array(indices)[mask & (~is_high)]
        high_idx = np.array(indices)[mask & is_high]

        # Draw a 2D KDE background when possible (helps show density for multi-region nets)
        try:
            x_vals_k = x_mean.loc[np.array(indices)[mask]].values
            y_vals_k = y_mean.loc[np.array(indices)[mask]].values
            # Only attempt KDE if we have more than a few unique points
            if len(x_vals_k) >= 5 and len(np.unique(x_vals_k)) > 1 and len(np.unique(y_vals_k)) > 1:
                # Use a perceptually-uniform, high-contrast colormap with slightly stronger fill
                sns.kdeplot(x=x_vals_k, y=y_vals_k, ax=ax, fill=False, 
                            cmap='magma', alpha=0.35, thresh=0.02, levels=6)
        except Exception:
            pass

        ax.scatter(x_mean.loc[normal_idx], y_mean.loc[normal_idx], s=30, alpha=0.6, color='steelblue', edgecolors='k', linewidth=0.3)
        ax.scatter(x_mean.loc[high_idx], y_mean.loc[high_idx], s=90, alpha=0.95, color='orange', edgecolors='black', linewidth=0.8, zorder=3)

        if len(high_idx) > 0:
            ax.errorbar(x_mean.loc[high_idx], y_mean.loc[high_idx], xerr=x_std.loc[high_idx], yerr=y_std.loc[high_idx],
                        fmt='none', ecolor='gray', alpha=0.8, capsize=3, zorder=2)

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(cfg['title'])
        ax.grid(True, alpha=0.3)

    # put legend in a single compact row to avoid excessive blank space below
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis('off')
    # Rotate x tick labels and reduce fontsize to avoid overlap with legend row
    for a in axes:
        plt.setp(a.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    # Use tight_layout but reserve significant space at the bottom for the legend row
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        try:
            # leave a very small bottom margin (6%) for the legend row to minimize blank space
            plt.tight_layout(rect=[0, 0.06, 1, 0.98])
        except Exception:
            pass

    final_path = outpath
    fig.savefig(final_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {"outpath": outpath}


def _make_ei_highlight_kde(df, outpath, top_n=30, stem=None):
    """Create E/I 2x3 figure with KDE background, colored regions and legend.
    Saves both PNG and PDF (PDF will be vector where possible).
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # compute derived totals if missing
    if 'cont_elec_in' not in df.columns and 'cont_in' in df.columns and 'elec_in' in df.columns:
        df['cont_elec_in'] = df['cont_in'] + df['elec_in']
    if 'cont_elec_out' not in df.columns and 'cont_out' in df.columns and 'elec_out' in df.columns:
        df['cont_elec_out'] = df['cont_out'] + df['elec_out']

    plot_configs = [
        {"title": "Synaptic (cont_in vs cont_out)", "x_col": "cont_in", "y_col": "cont_out"},
        {"title": "Overall (cont+elec_in vs cont+elec_out)", "x_col": "cont_elec_in", "y_col": "cont_elec_out"},
        {"title": "E/I Input Ratio (Continuous)", "x_col": "cont_inh_in", "y_col": "cont_exc_in"},
        {"title": "E/I Output Ratio (Continuous)", "x_col": "cont_inh_out", "y_col": "cont_exc_out"},
        {"title": "E/I Ratio (Overall)", "x_col": "cont_elec_in", "y_col": "cont_elec_out"},
        {"title": "I/E Ratio (Overall)", "x_col": "cont_elec_out", "y_col": "cont_elec_in"},
    ]

    # increase spacing between the two plot rows but keep legend area compact
    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[9, 9, 0.9, 0.15], hspace=0.5, wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, i]) for i in range(3)]
    fig.suptitle(f'E/I KDE & Highlights - {stem}' if stem else 'E/I KDE & Highlights', fontsize=16)

    # region colors
    if 'region' in df.columns:
        unique_regions = list(df['region'].astype(str).unique())
        cmap = plt.get_cmap('tab20')
        colors = {r: cmap(i % 20) for i, r in enumerate(unique_regions)}
    else:
        unique_regions = []
        colors = {}

    # determine highlights
    top_in_idx = set(df.nlargest(top_n, 'cont_in').index) if 'cont_in' in df.columns else set()
    top_out_idx = set(df.nlargest(top_n, 'cont_out').index) if 'cont_out' in df.columns else set()
    highlight_idx = top_in_idx.union(top_out_idx)

    for i, cfg in enumerate(plot_configs):
        ax = axes[i]
        xcol = cfg['x_col']
        ycol = cfg['y_col']

        if xcol not in df.columns or ycol not in df.columns:
            ax.text(0.5, 0.5, f"Data not available\n{xcol} vs {ycol}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cfg['title'])
            continue

        # dropna and compute arrays
        sub = df[[xcol, ycol, 'region']].copy() if 'region' in df.columns else df[[xcol, ycol]].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            continue

        xvals = sub[xcol].values
        yvals = sub[ycol].values

        # KDE background (filled contours) with downsampling for performance
        try:
            if len(xvals) >= 5 and len(np.unique(xvals)) > 1 and len(np.unique(yvals)) > 1:
                max_kde = 2000
                if len(xvals) > max_kde:
                    # reproducible downsample
                    rs = np.random.RandomState(0)
                    sel = rs.choice(np.arange(len(xvals)), size=max_kde, replace=False)
                    x_k = xvals[sel]
                    y_k = yvals[sel]
                else:
                    x_k = xvals
                    y_k = yvals
                sns.kdeplot(x=x_k, y=y_k, ax=ax, fill=False, cmap='magma', alpha=0.28, thresh=0.02, levels=9)
        except Exception:
            pass

        # scatter colored by region if present
        if 'region' in sub.columns:
            for r in unique_regions:
                rsub = sub[sub['region'].astype(str) == str(r)]
                if rsub.empty:
                    continue
                pts = rsub.index.to_list()
                is_high = [idx in highlight_idx for idx in pts]
                normal = rsub[~np.array(is_high)]
                high = rsub[np.array(is_high)]
                ax.scatter(normal[xcol], normal[ycol], s=30, alpha=0.7, color=colors[r], edgecolors='k', linewidth=0.3, label=r, zorder=3)
                if not high.empty:
                    ax.scatter(high[xcol], high[ycol], s=90, alpha=0.95, color=colors[r], edgecolors='black', linewidth=0.8, zorder=4)
        else:
            # single color
            pts = sub.index.to_list()
            is_high = [idx in highlight_idx for idx in pts]
            normal = sub[~np.array(is_high)]
            high = sub[np.array(is_high)]
            ax.scatter(normal[xcol], normal[ycol], s=30, alpha=0.7, color='steelblue', edgecolors='k', linewidth=0.3, zorder=3)
            if not high.empty:
                ax.scatter(high[xcol], high[ycol], s=90, alpha=0.95, color='orange', edgecolors='black', linewidth=0.8, zorder=4)

        # Add overall regression line (global across this subplot)
        try:
            from scipy.stats import pearsonr
            xv = sub[xcol].values
            yv = sub[ycol].values
            # remove NaNs
            mask_reg = ~(np.isnan(xv) | np.isnan(yv))
            xv = xv[mask_reg]
            yv = yv[mask_reg]
            if len(xv) > 2 and len(np.unique(xv)) > 1 and len(np.unique(yv)) > 1:
                with np.errstate(divide='ignore', invalid='ignore'):
                    z = np.polyfit(xv, yv, 1)
                    p = np.poly1d(z)
                    xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 200)
                    ax.plot(xs, p(xs), 'r--', linewidth=1.6, alpha=0.9, zorder=2)
                    try:
                        corr, pval = pearsonr(xv, yv)
                        ax.text(0.05, 0.95, f"R = {corr:.3f}\np = {pval:.2e}", transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7), zorder=5)
                    except Exception:
                        pass
        except Exception:
            pass

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(cfg['title'])
        ax.grid(True, alpha=0.3)

    # legend row
    # put legend in a single compact row to avoid excessive blank space below
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis('off')
    if unique_regions:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            legend_ax.legend(handles, labels, ncol=min(6, len(labels)), loc='center', frameon=False, fontsize=9)
    # Rotate x tick labels and reduce fontsize to avoid overlap with legend row
    for a in axes:
        plt.setp(a.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    # Use tight_layout but reserve significant space at the bottom for the legend row
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        try:
            plt.tight_layout(rect=[0, 0.06, 1, 0.98])
        except Exception:
            pass

    # save PNG and PDF (PDF will be vector where possible)
    out_png = outpath
    out_pdf = str(Path(outpath).with_suffix('.pdf'))
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    try:
        fig.savefig(out_pdf, bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)
    return {"outpath": outpath}

for p in sorted(ROOT.glob(CSV_GLOB)):
    stem = p.stem
    dataset_dir = OUTROOT / stem
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print('Processing', p)
    df = pd.read_csv(p)

    saved = []
    try:
        target1 = dataset_dir / f"{stem}_ei_highlight_kde.png"
        r1 = _make_ei_highlight_kde(df, str(target1), top_n=30, stem=stem)
        saved.append(r1['outpath'])
    except Exception as e:
        print('Error highlight+kde', e)

    # Removed creation of redundant EI variants: all_err_hexbin and no_density

    # Combine the three similar EI variants into a single side-by-side image
    try:
        # create an EI errorbar figure (kept as canonical errorbars file)
        err_target = dataset_dir / f"{stem}_ei_errorbars.png"
        try:
            e = _make_ei_errorbars(df, str(err_target), top_n=30, stem=stem)
        except Exception:
            e = {"outpath": None}
        if e.get('outpath'):
            # keep the errorbars image in the gallery (do not create a separate _ei_combined.png)
            saved.append(str(err_target))
    except Exception as e:
        print('Error creating EI errorbars', e)

    try:
        target4 = dataset_dir / f"{stem}_population_stats_violins.png"
        v = mod.plot_violin_reg_and_histgrid(df, outpath=str(target4), use_log=True, basename=stem)
        saved.append(v['outpath'])
    except Exception as e:
        print('Error population_stats violins', e)

    # Add clustered and combined heatmaps if available in the analysis module
    try:
        target5 = dataset_dir / f"{stem}_clustermap.png"
        # Always use fallback to ensure small annotation/tick fonts
        h1 = _fallback_clustered_heatmap(df, str(target5), method='spearman')
        saved.append(h1['outpath'])
    except Exception as e:
        print('Error clustermap', e)

    try:
        target6 = dataset_dir / f"{stem}_combined_heatmaps.png"
        # Always use fallback combined heatmaps with small fonts
        h2 = _fallback_combined_heatmaps(df, str(target6), methods=("spearman","pearson"))
        saved.append(h2['outpath'])
    except Exception as e:
        print('Error combined_heatmaps', e)

    # (all_across summary removed — combined heatmaps serve this purpose)

    index_lines.append(f"<h2>{stem}</h2>")
    for s in saved:
        rel = Path(s).relative_to(OUTROOT)
        index_lines.append(f"<div><img src=\"{rel.as_posix()}\" style=\"max-width:800px;display:block;margin-bottom:8px\"></div>")

index_lines.append('</body></html>')
(OUTROOT / 'index.html').write_text('\n'.join(index_lines))
print('Done; gallery at plots/index.html')

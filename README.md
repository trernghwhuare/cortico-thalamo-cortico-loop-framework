# Cortico-Thalamo-Cortical Loop Framework

A reproducible computational neuroscience framework that integrates six morphologically validated thalamic neuron classes (TCRil/TCRm/TCRc and nRTil/nRTm/nRTc) into large-scale cortico-thalamo-cortical (CTC) and M2M1S1 network reconstructions.

## Scientific Contribution

This framework demonstrates that thalamic projection classes act as structural control parameters, systematically modulating E/I balance, synchronization tendency, and network complexity in large-scale CTC networks. The integration of morphological diversity enables distinct, testable responses to visual, TMS-like, and nociceptive stimuli.

## Framework Architecture

The framework follows a 15-step modular workflow:

### Step 1: Thalamus Neurons construction
- Six thalamic cell class simulations (`src/morphology/00_sim_nRTc.py`, `src/morphology/01_sim_nRTm.py`, `src/morphology/02_sim_nRTil.py`, `src/morphology/03_sim_TCRc.py`, `src/morphology/04_sim_TCRm.py`, `src/morphology/05_sim_TCRil.py`)
- Validate newly generated NeuroML cell files (`data/morphologies/nRTc.cell.nml`,`data/morphologies/nRTil.cell.nml`, `data/morphologies/nRTm.cell.nml`,`data/morphologies/TCRc.cell.nml`, `data/morphologies/TCRil.cell.nml`, `data/morphologies/TCRil.cell.nml`)

### Step 2: cortical Neurons Cataloging & Clustering  
- Cell data extraction and pattern cataloging (`src/cell_catalog/01_extract_cell_data.py`, `src/cell_catalog/02_catalog_cells_by_pattern.py`)
- Clustering of CT, IT, and PT cell types (`src/cell_catalog/cluster_ct_cells.py`, `src/cell_catalog/cluster_it_cells.py`, `src/cell_catalog/cluster_pt_cells.py`)
- CSV outputs for cell catalogs and clusters (`src/cell_catalog/extract_member_segment_count.py`)

### Step 3: Network Construction
- CTC and M2M1S1 network generation (`src/network_construction/11_CTC_max+.py`, `src/network_construction/33_M2M1S1_max+.py`)
- Newly generated NeuroML network files (`data/networks/M2M1S1_max_plus.net.nml`, `data/networks/max_CTC_plus.net.nml`)
- Connection statistics and population analysis  (`src/analysis/77_0_extract_net_params.py`)

### Step 4-6: Network Analysis
- Comprehensive network analysis (`src/analysis/55_0_analysis.py`)
- Clustering coefficient, fractal dimension, and machine learning analysis (`src/analysis/55_1_CC.py`, `src/analysis/55_2_FD.py`, `src/analysis/55_3_MLP.py`, `src/analysis/55_1_clustering.py`)
- Connection ratio analysis (`src/analysis/55_2_connection_ratio.py`, `src/analysis/generate_all_ei_plots.py`)

### Step 7-8: Ground Truth Parameter Extraction
- GT parameter extraction from network structure (`src/ground_truth/66_1_extract_gt_params.py`)
- Hierarchical and SFDP visualization of networks modes in macro/meso scale (`src/ground_truth/66_1_gt_sfdp_meso.py`, `src/ground_truth/66_1_gt_sfdp_macro.py`, `src/ground_truth/66_1_gt_hierarchical.py`)

### Step 9: Dynamic Analysis & Comparisons
- Animation of network dynamics (`src/dynamics/69_5_gt_combined_ani.py`,`src/dynamics/69_6_gt_combined_ani_adv.py`)
- E/I vs homogeneous model comparisons (`src/dynamics/69_6_simple_ei_vs_homogeneous_comparison.py`)
- Statistical comparison results (`src/dynamics/69_6_sirs_statistical_comparison.py`)

### Step 10: Network Parameter Extraction
- Anatomy, circuit, layer, pathway, and region parameter extraction (`src/analysis/77_0_extract_net_params.py`)

### Step 11: Stimulation Simulations
- Pain, TMS, and visual stimuli simulations (`src/stimuli/77_1_pain_stimuli_simulation.py`, `src/stimuli/77_1_TMS_stimuli_simulation.py`, `src/stimuli/77_1_visual_stimuli_simulation.py`)
- Dynamic stimulation response animations (`src/stimuli/77_2_animate_stimulation_dynamics.py`)

### Step 12-13: Mean-Field Integration
- YAML parameter generation for NEST simulations (`src/mean_field/88_1_generate_MF_params.py`, `src/mean_field/88_5_nest_yaml_simulation.py`)
- Mean-field optimization (`src/mean_field/88_9_MF_optimization.py`)

### Step 14-15: PyNN Integration & Visualization
- Unified PyNN mean-field simulations (`src/pynn/99_1_pynn_MF_unified.py`, `src/pynn/99_3_pynn_MF_visual.py`, `src/pynn/99_5_pynn_MF_TMS.py`, `src/pynn/99_7_pynn_MF_pain.py`)
- Comprehensive plotting and voltage comparisons (`src/pynn/99_9_plot_pynn_comparisons.py`,`src/pynn/99_10_plot_pynn_voltage_comparisons.py`)

## Dependencies

- Python 3.x
- NEST Simulator
- pyNeuroML
- graph-tool
- pandas, numpy, matplotlib
- PyNN

## Usage

1. **Clone the repository**
2. **Install dependencies** from `requirements.txt`
3. **Run individual steps** or the complete pipeline
4. **Analyze results** in the `results/` directory
5. **Visualize findings** using the provided plotting scripts

## Repository Structure

```
cortico-thalamo-cortico-loop-framework/
├── src/                     # Source code organized by functional modules
│   ├── morphology/          # Thalamic neuron morphology simulations
│   ├── cell_catalog/        # Cortical neuron cataloging and clustering
│   ├── network_construction/ # Large-scale CTC and M2M1S1 network generation
│   ├── analysis/            # Network topology and connectivity analysis
│   ├── ground_truth/        # Ground truth parameter extraction and visualization
│   ├── dynamics/            # Network dynamics and statistical comparisons
│   ├── stimuli/             # Multi-modal stimulation simulations
│   ├── mean_field/          # Mean-field theory integration and optimization
│   └── pynn/                # PyNN simulations and visualization
├── data/                    # Input data and generated morphologies
│   ├── morphologies/        # Generated NeuroML cell morphology files
│   ├── catalog/             # Cell catalog data and clustering results
│   ├── networks/            # Generated NeuroML network files
│   ├── net_params/          # Extracted network parameters by anatomy/circuit/layer
│   ├── ground_truth/        # Ground truth parameter configurations
│   └── stimuli/             # Stimuli configuration files
├── configs/                 # Network configuration files
│   └── yaml/                # YAML configuration files for CTC and M2M1S1 networks
├── results/                 # Generated results and analysis outputs
│   ├── analysis_out/        # Network analysis statistics and summaries
│   ├── plots/               # Generated visualization plots
│   ├── MF_optimized/        # Mean-field optimization results
│   ├── PyNN_results/        # PyNN simulation outputs
│   └── yaml_nest/           # NEST-compatible YAML parameter files
├── notebooks/               # Jupyter notebooks for interactive demonstrations
├── paper/                   # Publication figures and supplementary materials
├── binder/                  # Binder environment configuration
├── _build/                  # Build artifacts and cached content
├── requirements.txt         # Python dependencies
├── myst.yml                 # MyST documentation configuration
├── paper.md                 # Main manuscript
├── paper.bib                # Bibliography references
└── supplementary_tables.tex # Supplementary tables in LaTeX format
```

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.



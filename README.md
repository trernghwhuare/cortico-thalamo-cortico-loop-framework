# Cortico-Thalamo-Cortical Loop Framework

A reproducible computational neuroscience framework that integrates six morphologically validated thalamic neuron classes (TCRil/TCRm/TCRc and nRTil/nRTm/nRTc) into large-scale cortico-thalamo-cortical (CTC) and M2M1S1 network reconstructions.

## Scientific Contribution

This framework demonstrates that thalamic projection classes act as structural control parameters, systematically modulating E/I balance, synchronization tendency, and network complexity in large-scale CTC networks. The integration of morphological diversity enables distinct, testable responses to visual, TMS-like, and nociceptive stimuli.

## Framework Architecture

The framework follows a 15-step modular workflow:

### Step 1: Morphology Generation
- Six thalamic cell class simulations (`00_sim_*.py`)
- NeuroML morphology files (`.cell.nml`)

### Step 2: Cell Cataloging & Clustering  
- Cell data extraction and pattern cataloging
- Clustering of CT, IT, and PT cell types
- CSV outputs for cell catalogs and clusters

### Step 3: Network Construction
- CTC and M2M1S1 network generation (`11_CTC_max+.py`, `33_M2M1S1_max+.py`)
- NeuroML network files (`.net.nml`)
- Connection statistics and population analysis

### Step 4-6: Network Analysis
- Comprehensive network analysis (`55_0_analysis.py`)
- Clustering coefficient, fractal dimension, and machine learning analysis
- Connection ratio analysis

### Step 7-8: Ground Truth Parameter Extraction
- GT parameter extraction from network structure
- Hierarchical and SFDP visualization (macro/meso scale)

### Step 9: Dynamic Analysis & Comparisons
- Animation of network dynamics
- E/I vs homogeneous model comparisons
- Statistical comparison results

### Step 10: Network Parameter Extraction
- Anatomy, circuit, layer, pathway, and region parameter extraction

### Step 11: Stimulation Simulations
- Pain, TMS, and visual stimuli simulations
- Dynamic stimulation response animations

### Step 12-13: Mean-Field Integration
- YAML parameter generation for NEST simulations
- Mean-field optimization and validation

### Step 14-15: PyNN Integration & Visualization
- Unified PyNN mean-field simulations
- Comprehensive plotting and voltage comparisons

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
├── data/                    # Input data and morphologies
├── src/                     # Source code organized by functionality
├── configs/                 # Configuration files
├── results/                 # Generated results and outputs
└── paper/                   # Publication figures and materials
```

## Citation

[Include your publication citation here]

## License

[Specify license information]
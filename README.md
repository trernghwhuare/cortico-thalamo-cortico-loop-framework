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
[1] Simone Russo et al. “Thalamic feedback shapes brain responses evoked by cortical stimulation in mice and humans”. In: Nature Communications 16 (1 2025), p. 3627. issn: 2041-1723. doi: 10.1038/s41467-025-58717-2. url: https://doi.org/10.1038/s41467-025-58717-2.

[2] Tomas Vega-Zuniga et al. “A thalamic hub-and-spoke network enables visual perception during action by coordinating visuomotor dynamics”. In: Nature Neuroscience 28 (3 2025), pp. 627–639. issn: 1546-1726.
doi: 10.1038/s41593-025-01874-w. url: https://doi.org/10.1038/s41593-025-01874-w.

[3] Robert C. Cannon et al. “LEMS: a language for expressing complex biological models in concise and hierarchical form and its use in underpinning NeuroML 2”. In: Frontiers in Neuroinformatics 8 (2014). doi: 10.3389/fninf.2014.00079.

[4] Michael Vella et al. “libNeuroML and PyLEMS: using Python to combine procedural and declarative modeling approaches in computational neuroscience.” In: Frontiers in neuroinformatics 8 (2014), p. 38. doi: 10.3389/fninf.2014.00038. 

[5] Marc-Oliver Gewaltig and Markus Diesmann. “NEST (NEural Simulation Tool)”. In: Scholarpedia 2.4 (2007), p. 1430. doi: 10.4249/scholarpedia.1430. url: http://www.scholarpedia.org/article/NEST_(NEural_Simulation_Tool).

[6] Tiago P. Peixoto. “The graph-tool python library”. In: figshare (2014). doi: 10.6084/m9.figshare.1164194. url: http://figshare.com/articles/graph_tool/1164194.

[7] Moritz Layer et al. “NNMT: Mean-Field Based Analysis Tools for Neuronal Network Models”. In: Frontiers in Neuroinformatics 16 (2022). issn: 1662-5196. doi: 10 . 3389 / fninf . 2022 . 835657. url: https :
//www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.835657.

[8] Thomas Parr et al. “Neuronal message passing using Mean-field, Bethe, and Marginal approximations”.In: Scientific Reports 9 (1 2019), p. 1889. issn: 2045-2322. doi: 10.1038/s41598-018-38246-3. url:
https://doi.org/10.1038/s41598-018-38246-3.

[9] Marianne J Bezaire et al. “Interneuronal mechanisms of hippocampal theta oscillations in a full-scale model of the rodent CA1 circuit”. In: eLife 5 (Dec. 2016). Ed. by Frances K Skinner, e18566. issn: 2050-084X. doi: 10.7554/eLife.18566. url: https://doi.org/10.7554/eLife.18566.

[10] Markram Henry et al. “Reconstruction and Simulation of Neocortical Microcircuitry.” In: Cell 163 (2 2015). issn: 1097-4172. doi: 10.1016/j.cell.2015.09.029. url: http://www.ncbi.nlm.nih.gov/pubmed/26451489.

[11] Srikanth Ramaswamy et al. “The neocortical microcircuit collaboration portal: a resource for rat somatosen-sory cortex”. In: Frontiers in Neural Circuits 9 (2015). issn: 1662-5110. doi: 10.3389/fncir.2015.00044. url: https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00044.

[12] Michael W. Reimann et al. “An algorithm to predict the connectome of neural microcircuits”. In: Frontiers in Computational Neuroscience 9 (2015). issn: 1662-5188. doi: 10 . 3389 / fncom . 2015 . 00120. url: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00120.

[13] Roger D. Traub et al. “Single-Column Thalamocortical Network Model Exhibiting Gamma Oscillations, Sleep Spindles, and Epileptogenic Bursts”. In: Journal of Neurophysiology 93 (4 2005). PMID: 15525801, pp. 2194–2232. doi: 10.1152/jn.00983.2004. eprint: https://doi.org/10.1152/jn.00983.2004. url: https://journals.physiology.org/doi/abs/10.1152/jn.00983.2004.

[14] Shepherd Gordon M G and Yamawaki Naoki. “Untangling the cortico-thalamo-cortical loop: cellular pieces of a knotty circuit puzzle.” In: Nat Rev Neurosci 22 (7 2021). issn: 1471-0048. doi: 10.1038/s41583-021-00459-3. url: http://www.ncbi.nlm.nih.gov/pubmed/33958775.

[15] Michael W. Reimann et al. “Morphological Diversity Strongly Constrains Synaptic Connectivity and Plasticity”. In: Cerebral Cortex 27 (0 June 2017), pp. 4570–4585. issn: 1047-3211. doi: 10.1093/cercor/bhx150. eprint: https://academic.oup.com/cercor/article-pdf/27/9/4570/24361383/bhx150.pdf. url: https://doi.org/10.1093/cercor/bhx150.

[16] Gal Eyal et al. “Rich cell-type-specific network topology in neocortical microcircuitry”. In: Nat Neurosci 20 (7 2017). issn: 1546-1726. doi: 10.1038/nn.4576. url: http://www.ncbi.nlm.nih.gov/pubmed/28581480.

[17] Julie A. Harris et al. “Hierarchical organization of cortical and thalamic connectivity”. In: Nature 575 (2019), pp. 195–202. issn: 1476-4687. doi: 10.1038/s41586-019-1716-z. url: http://www.ncbi.nlm.nih.gov/pubmed/26980805.

[18] Kenneth D. Harris and Gordon M. G. Shepherd. “The neocortical circuit: themes and variations”. In: Nature Neuroscience 18 (2015), pp. 170–181. issn: 1546-1726. doi: 10 . 1038 / nn . 3917. url: https ://pubmed.ncbi.nlm.nih.gov/25622573/.

[19] Charles J Wilson. “The Sensory Striatum”. In: Neuron 83 (5 2014), pp. 999–1001. issn: 0896-6273. doi: 10.1016/j.neuron.2014.08.025. url: https://doi.org/10.1016/j.neuron.2014.08.025.

[20] Minju Jeong et al. “Comparative three-dimensional connectome map of motor cortical projections in the mouse brain”. In: Scientific Reports 1 (2016), p. 20072. doi: 10 . 1038 / srep20072. url: https://doi.org/10.1038/srep20072.

[21] C. Bennett et al. “Higher-Order Thalamic Circuits Channel Parallel Streams of Visual Information in Mice”. In: Neuron 102 (2019), 477–492.e5. issn: 0896-6273 (Print) and 0896-6273. doi: 10.1016/j.neuron.2019.02.010. url: https://pmc.ncbi.nlm.nih.gov/articles/PMC8638696/.

[22] K. D. Alloway, M. L. Olson, and J. B. Smith. “Contralateral corticothalamic projections from MI whisker cortex: potential route for modulating hemispheric interactions”. In: J Comp Neurol 510 (2007), pp. 100–16. issn: 0021-9967 (Print) and 0021-9967. doi: 10.1002/cne.21782. url: https://pmc.ncbi.nlm.nih.gov/articles/PMC2504743/.

[23] J. Winnubst et al. “Reconstruction of 1,000 Projection Neurons Reveals New Cell Types and Organization of Long-Range Connectivity in the Mouse Brain”. In: Cell 179 (1 2019), 268–281.e13. issn: 0092-8674 (Print) and 0092-8674. doi: 10.1016/j.cell.2019.01.047. url: https://pubmed.ncbi.nlm.nih.gov/31495573/.

[24] E. G. Jones. “Viewpoint: the core and matrix of thalamic organization”. In: Neuroscience 85 (2 1998), pp. 331–345. issn: 0306-4522. doi: 10.1016/S0306-4522(97)00581-2. url: https://pubmed.ncbi.nlm.nih.gov/9622234/.

[25] C. E. Landisman and B. W. Connors. “VPM and PoM nuclei of the rat somatosensory thalamus: intrinsic neuronal properties and corticothalamic feedback”. In: Cereb Cortex 17 (12 2007), pp. 2853–65. issn:1047-3211. doi: 10.1093/cercor/bhm211. url: https://www.sci-hub.ru/10.1093/cercor/bhm025.

[26] Michael Vella et al. “libNeuroML and PyLEMS: using Python to combine procedural and declarativemodeling approaches in computational neuroscience”. In: Frontiers in Neuroinformatics 8 (2014). issn: 1662-5196. doi:10.3389/fninf.2014.00038. url: https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2014.00038.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License

Copyright (c) 2026 [Hua Cheng]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

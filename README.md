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

## Citation

@article{10.1016/j.celrep.2017.05.044,
    author = {Alexandra, Clemente-Perez
    and Stefanie Ritter, Makinson
    and Bryan, Higashikubo
    and Scott, Brovarney
    and Frances S, Cho
    and Alexander, Urry
    and Stephanie S, Holden
    and Matthew, Wimer
    and Csaba, Dávid
    and Lief E, Fenno
    and László, Acsády
    and Karl, Deisseroth
    and Jeanne T, Paz},
    title = {Distinct Thalamic Reticular Cell Types Differentially Modulate Normal and Pathological Cortical Rhythms.},
    journal = {Cell Rep},
    year = 2017,
    volume = 19,
    issue = 10,
    abstract = {Integrative brain functions depend on widely distributed, rhythmically coordinated computations. Through its long-ranging connections with cortex and most senses, the thalamus orchestrates the flow of cognitive and sensory information. Essential in this process, the nucleus reticularis thalami (nRT) gates different information streams through its extensive inhibition onto other thalamic nuclei, however, we lack an understanding of how different inhibitory neuron subpopulations in nRT function as gatekeepers. We dissociated the connectivity, physiology, and circuit functions of neurons within rodent nRT, based on parvalbumin (PV) and somatostatin (SOM) expression, and validated the existence of such populations in human nRT. We found that PV, but not SOM, cells are rhythmogenic, and that PV and SOM neurons are connected to and modulate distinct thalamocortical circuits. Notably, PV, but not SOM, neurons modulate somatosensory behavior and disrupt seizures. These results provide a conceptual framework for how nRT may gate incoming information to modulate brain-wide rhythms.},
    issn = {2211-1247},
    doi = {10.1016/j.celrep.2017.05.044},
    url = {http://www.ncbi.nlm.nih.gov/pubmed/28591583},
}
@article{10.1038/nn.4576,
    author={Eyal, Gal
    and Michael, London
    and Amir, Globerson
    and Srikanth, Ramaswamy
    and Michael W, Reimann
    and Eilif, Muller
    and Henry, Markram
    and Idan, Segev},
    title = {Rich cell-type-specific network topology in neocortical microcircuitry},
    journal = {Nat Neurosci},
    year = 2017,
    volume = 20,
    issue = 7,
    abstract = {Uncovering structural regularities and architectural topologies of cortical circuitry is vital for understanding neural computations. Recently, an experimentally constrained algorithm generated a dense network reconstruction of a ∼0.3-mm(3) volume from juvenile rat somatosensory neocortex, comprising ∼31,000 cells and ∼36 million synapses. Using this reconstruction, we found a small-world topology with an average of 2.5 synapses separating any two cells and multiple cell-type-specific wiring features. Amounts of excitatory and inhibitory innervations varied across cells, yet pyramidal neurons maintained relatively constant excitation/inhibition ratios. The circuit contained highly connected hub neurons belonging to a small subset of cell types and forming an interconnected cell-type-specific rich club. Certain three-neuron motifs were overrepresented, matching recent experimental results. Cell-type-specific network properties were even more striking when synaptic strength and sign were considered in generating a functional topology. Our systematic approach enables interpretation of microconnectomics 'big data' and provides several experimentally testable predictions.},
    issn = {1546-1726},
    doi = {10.1038/nn.4576},
    url = {http://www.ncbi.nlm.nih.gov/pubmed/28581480},
}

@article{10.3389/fncom.2015.00120,
    author = {Reimann, Michael W.  and King, James G.  and Muller, Eilif B.  and Ramaswamy, Srikanth  and Markram, Henry },
    title = {An algorithm to predict the connectome of neural microcircuits},
    journal = {Frontiers in Computational Neuroscience},
    volume = 9,
    year = 2015,
    url = {https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00120},
    doi = {10.3389/fncom.2015.00120},
    issn = {1662-5188},
    abstract = {Experimentally mapping synaptic connections, in terms of the issues and locations of their synapses and estimating connection probabilities is still not a tractable task, even for small volumes of tissue, In fact, the 6 layers of the neocortex contain thousands of unique types of synaptic connections between the many different types of neurons, of which only a handful have been characterized experimentally. Here we present a theoretical framework and a data-driven algorithmic strategy to digitally reconstruct the complete synaptic connectivity between the different types of neuron in a in a small well-defined volume of tissue – the micro scale connectome of a neural microcircuit. By enforcing a set of established principles of synaptic connectivity and leveraging interdependencies between fundamental properties of neural microcircuits to constrain the reconstructed connectivity, the algorithm yields three parameters per connection type that predict the anatomy of all types of biologically viable synaptic connections. The predictions reproduce a spectrum of experimental data on synaptic connectivity not used by the algorithm. We conclude that an algorithmic approach to the connectome can serve as a tool to accelerate experimental mapping, indicating the minimal dataset required to make useful predictions, identifying the datasets required to improve their accuracy, testing the feasibility of experimental measurements, and making it possible to test hypotheses of synaptic connectivity.}
}

@article{10.7554/eLife.18566,
    author = {Bezaire, Marianne J and Raikov, Ivan and Burk, Kelly and Vyas, Dhrumil and Soltesz, Ivan},
    article_type = {journal},
    title = {Interneuronal mechanisms of hippocampal theta oscillations in a full-scale model of the rodent CA1 circuit},
    editor = {Skinner, Frances K},
    volume = 5,
    year = 2016,
    month = {12},
    pub_date = {2016-12-23},
    pages = {e18566},
    citation = {eLife 2016;5:e18566},
    doi = {10.7554/eLife.18566},
    url = {https://doi.org/10.7554/eLife.18566},
    abstract = {The hippocampal theta rhythm plays important roles in information processing; however, the mechanisms of its generation are not well understood. We developed a data-driven, supercomputer-based, full-scale (1:1) model of the rodent CA1 area and studied its interneurons during theta oscillations. Theta rhythm with phase-locked gamma oscillations and phase-preferential discharges of distinct interneuronal types spontaneously emerged from the isolated CA1 circuit without rhythmic inputs. Perturbation experiments identified parvalbumin-expressing interneurons and neurogliaform cells, as well as interneuronal diversity itself, as important factors in theta generation. These simulations reveal new insights into the spatiotemporal organization of the CA1 circuit during theta oscillations.},
    keywords = {computational, inhibition, hippocampus, model network, oscillation, theta},
    journal = {eLife},
    issn = {2050-084X},
    publisher = {eLife Sciences Publications, Ltd},
}

@article{10.1093/cercor/bhx150,
    author = {Reimann, Michael W. and Horlemann, Anna-Lena and Ramaswamy, Srikanth and Muller, Eilif B. and Markram, Henry},
    title = {Morphological Diversity Strongly Constrains Synaptic Connectivity and Plasticity},
    journal = {Cerebral Cortex},
    volume = 27,
    issue = 0,
    pages = {4570-4585},
    year = 2017,
    month = {06},
    abstract = {Synaptic connectivity between neurons is naturally constrained by the anatomical overlap of neuronal arbors, the space on the axon available for synapses, and by physiological mechanisms that form synapses at a subset of potential synapse locations. What is not known is how these constraints impact emergent connectivity in a circuit with diverse morphologies. We investigated the role of morphological diversity within and across neuronal types on emergent connectivity in a model of neocortical microcircuitry. We found that the average overlap between the dendritic and axonal arbors of different types of neurons determines neuron-type specific patterns of distance-dependent connectivity, severely constraining the space of possible connectomes. However, higher order connectivity motifs depend on the diverse branching patterns of individual arbors of neurons belonging to the same type. Morphological diversity across neuronal types, therefore, imposes a specific structure on first order connectivity, and morphological diversity within neuronal types imposes a higher order structure of connectivity. We estimate that the morphological constraints resulting from diversity within and across neuron types together lead to a 10-fold reduction of the entropy of possible connectivity configurations, revealing an upper bound on the space explored by structural plasticity.},
    issn = {1047-3211},
    doi = {10.1093/cercor/bhx150},
    url = {https://doi.org/10.1093/cercor/bhx150},
    eprint = {https://academic.oup.com/cercor/article-pdf/27/9/4570/24361383/bhx150.pdf},
}

@article{10.3389/fncir.2015.00044,
    author = {Ramaswamy, Srikanth  and Courcol, Jean-Denis  and Abdellah, Marwan  and Adaszewski, Stanislaw R.  and Antille, Nicolas  and Arsever, Selim  and Atenekeng, Guy  and Bilgili, Ahmet  and Brukau, Yury  and Chalimourda, Athanassia  and Chindemi, Giuseppe  and Delalondre, Fabien  and Dumusc, Raphael  and Eilemann, Stefan  and Gevaert, Michael Emiel  and Gleeson, Padraig  and Graham, Joe W.  and Hernando, Juan B.  and Kanari, Lida  and Katkov, Yury  and Keller, Daniel  and King, James G.  and Ranjan, Rajnish  and Reimann, Michael W.  and Rössert, Christian  and Shi, Ying  and Shillcock, Julian C.  and Telefont, Martin  and Van Geit, Werner  and Villafranca Diaz, Jafet  and Walker, Richard  and Wang, Yun  and Zaninetta, Stefano M.  and DeFelipe, Javier  and Hill, Sean L.  and Muller, Jeffrey  and Segev, Idan  and Schürmann, Felix  and Muller, Eilif B.  and Markram, Henry },
    title = {The neocortical microcircuit collaboration portal: a resource for rat somatosensory cortex},
    journal = {Frontiers in Neural Circuits},
    volume = 9,
    year = 2015,
    issn = {1662-5110},
    doi = {10.3389/fncir.2015.00044},
    url = {https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2015.00044},

}

@article{10.1152/jn.00983.2004,
    author = {Traub, Roger D. and Contreras, Diego and Cunningham, Mark O. and Murray, Hilary and LeBeau, Fiona E. N. and Roopun, Anita and Bibbig, Andrea and Wilent, W. Bryan and Higley, Michael J. and Whittington, Miles A.},
    title = {Single-Column Thalamocortical Network Model Exhibiting Gamma Oscillations, Sleep Spindles, and Epileptogenic Bursts},
    journal = {Journal of Neurophysiology},
    volume = 93,
    issue = 4,
    pages = {2194-2232},
    year = 2005,
    doi = {10.1152/jn.00983.2004},
    note ={PMID: 15525801},
    url = {https://journals.physiology.org/doi/abs/10.1152/jn.00983.2004},
    eprint = {https://doi.org/10.1152/jn.00983.2004},
    abstract = { To better understand population phenomena in thalamocortical neuronal ensembles, we have constructed a preliminary network model with 3,560 multicompartment neurons (containing soma, branching dendrites, and a portion of axon). Types of neurons included superficial pyramids (with regular spiking [RS] and fast rhythmic bursting [FRB] firing behaviors); RS spiny stellates; fast spiking (FS) interneurons, with basket-type and axoaxonic types of connectivity, and located in superficial and deep cortical layers; low threshold spiking (LTS) interneurons, which contacted principal cell dendrites; deep pyramids, which could have RS or intrinsic bursting (IB) firing behaviors, and endowed either with nontufted apical dendrites or with long tufted apical dendrites; thalamocortical relay (TCR) cells; and nucleus reticularis (nRT) cells. To the extent possible, both electrophysiology and synaptic connectivity were based on published data, although many arbitrary choices were necessary. In addition to synaptic connectivity (by AMPA/kainate, NMDA, and GABAA receptors), we also included electrical coupling between dendrites of interneurons, nRT cells, and TCR cells, and—in various combinations—electrical coupling between the proximal axons of certain cortical principal neurons. Our network model replicates several observed population phenomena, including 1) persistent gamma oscillations; 2) thalamocortical sleep spindles; 3) series of synchronized population bursts, resembling electrographic seizures; 4) isolated double population bursts with superimposed very fast oscillations (>100 Hz, “VFO”); 5) spike-wave, polyspike-wave, and fast runs (about 10 Hz). We show that epileptiform bursts, including double and multiple bursts, containing VFO occur in rat auditory cortex in vitro, in the presence of kainate, when both GABAA and GABAB receptors are blocked. Electrical coupling between axons appears necessary (as reported previously) for persistent gamma and additionally plays a role in the detailed shaping of epileptogenic events. The degree of recurrent synaptic excitation between spiny stellate cells, and their tendency to fire throughout multiple bursts, also appears critical in shaping epileptogenic events. }
}

@article{10.1038/s41583-021-00459-3,
    author = {Gordon M G, Shepherd and Naoki, Yamawaki},
    title = {Untangling the cortico-thalamo-cortical loop: cellular pieces of a knotty circuit puzzle.},
    journal = {Nat Rev Neurosci},
    year = 2021,
    volume = 22,
    issue = 7,
    abstract = {Functions of the neocortex depend on its bidirectional communication with the thalamus, via cortico-thalamo-cortical (CTC) loops. Recent work dissecting the synaptic connectivity in these loops is generating a clearer picture of their cellular organization. Here, we review findings across sensory, motor and cognitive areas, focusing on patterns of cell type-specific synaptic connections between the major types of cortical and thalamic neurons. We outline simple and complex CTC loops, and note features of these loops that appear to be general versus specialized. CTC loops are tightly interlinked with local cortical and corticocortical (CC) circuits, forming extended chains of loops that are probably critical for communication across hierarchically organized cerebral networks. Such CTC-CC loop chains appear to constitute a modular unit of organization, serving as scaffolding for area-specific structural and functional modifications. Inhibitory neurons and circuits are embedded throughout CTC loops, shaping the flow of excitation. We consider recent findings in the context of established CTC and CC circuit models, and highlight current efforts to pinpoint cell type-specific mechanisms in CTC loops involved in consciousness and perception. As pieces of the connectivity puzzle fall increasingly into place, this knowledge can guide further efforts to understand structure-function relationships in CTC loops.},
    issn= {1471-0048},
    doi= {10.1038/s41583-021-00459-3},
    url= {http://www.ncbi.nlm.nih.gov/pubmed/33958775},
}

@article{10.1016/j.cell.2015.09.029,
    author = {Henry, Markram
    and Eilif, Muller
    and Srikanth, Ramaswamy
    and Michael W, Reimann
    and Marwan, Abdellah
    and Carlos Aguado, Sanchez
    and Anastasia, Ailamaki
    and Lidia, Alonso-Nanclares
    and Nicolas, Antille
    and Selim, Arsever
    and Guy Antoine Atenekeng, Kahou
    and Thomas K, Berger
    and Ahmet, Bilgili
    and Nenad, Buncic
    and Athanassia, Chalimourda
    and Giuseppe, Chindemi
    and Jean-Denis, Courcol
    and Fabien, Delalondre
    and Vincent, Delattre
    and Shaul, Druckmann
    and Raphael, Dumusc
    and James, Dynes
    and Stefan, Eilemann
    and Eyal, Gal
    and Michael Emiel, Gevaert
    and Jean-Pierre, Ghobril
    and Albert, Gidon
    and Joe W, Graham
    and Anirudh, Gupta
    and Valentin, Haenel
    and Etay, Hay
    and Thomas, Heinis
    and Juan B, Hernando
    and Michael, Hines
    and Lida, Kanari
    and Daniel, Keller
    and John, Kenyon
    and Georges, Khazen
    and Yihwa, Kim
    and James G, King
    and Zoltan, Kisvarday
    and Pramod, Kumbhar
    and Sébastien, Lasserre
    and Jean-Vincent, Le Bé
    and Bruno R C, Magalhães
    and Angel, Merchán-Pérez
    and Julie, Meystre
    and Benjamin Roy, Morrice
    and Jeffrey, Muller
    and Alberto, Muñoz-Céspedes
    and Shruti, Muralidhar
    and Keerthan, Muthurasa
    and Daniel, Nachbaur
    and Taylor H, Newton
    and Max, Nolte
    and Aleksandr, Ovcharenko
    and Juan, Palacios
    and Luis, Pastor
    and Rodrigo, Perin
    and Rajnish, Ranjan
    and Imad, Riachi
    and José-Rodrigo, Rodríguez
    and Juan Luis, Riquelme
    and Christian, Rössert
    and Konstantinos, Sfyrakis
    and Ying, Shi
    and Julian C, Shillcock
    and Gilad, Silberberg
    and Ricardo, Silva
    and Farhan, Tauheed
    and Martin, Telefont
    and Maria, Toledo-Rodriguez
    and Thomas, Tränkler
    and Werner, Van Geit
    and Jafet Villafranca, Díaz
    and Richard, Walker
    and Yun, Wang
    and Stefano M, Zaninetta
    and Javier, DeFelipe
    and Sean L, Hill
    and Idan, Segev
    and Felix, Schürmann},
    title = {Reconstruction and Simulation of Neocortical Microcircuitry.},
    journal = {Cell},
    year = 2015,
    volume = 163,
    issue = 2,
    abstract = {We present a first-draft digital reconstruction of the microcircuitry of somatosensory cortex of juvenile rat. The reconstruction uses cellular and synaptic organizing principles to algorithmically reconstruct detailed anatomy and physiology from sparse experimental data. An objective anatomical method defines a neocortical volume of 0.29 ± 0.01 mm(3) containing ~31,000 neurons, and patch-clamp studies identify 55 layer-specific morphological and 207 morpho-electrical neuron subtypes. When digitally reconstructed neurons are positioned in the volume and synapse formation is restricted to biological bouton densities and issues of synapses per connection, their overlapping arbors form ~8 million connections with ~37 million synapses. Simulations reproduce an array of in vitro and in vivo experiments without parameter tuning. Additionally, we find a spectrum of network states with a sharp transition from synchronous to asynchronous activity, modulated by physiological mechanisms. The spectrum of network states, dynamically reconfigured around this transition, supports diverse information processing strategies. PAPERCLIP: VIDEO abstract.},
    issn = {1097-4172},
    doi = {10.1016/j.cell.2015.09.029},
    url = {http://www.ncbi.nlm.nih.gov/pubmed/26451489},
}

@article{10.1038/s41586-019-1716-z,
    author = {Harris, Julie A.
    and Mihalas, Stefan
    and Hirokawa, Karla E.
    and Whitesell, Jennifer D.
    and Choi, Hannah
    and Bernard, Amy
    and Bohn, Phillip
    and Caldejon, Shiella
    and Casal, Linzy
    and Cho, Andrew
    and Feiner, Aaron
    and Feng, David
    and Gaudreault, Nathalie
    and Gerfen, Charles R.
    and Graddis, Nile
    and Groblewski, Peter A.
    and Henry, Alex M.
    and Ho, Anh
    and Howard, Robert
    and Knox, Joseph E.
    and Kuan, Leonard
    and Kuang, Xiuli
    and Lecoq, Jerome
    and Lesnar, Phil
    and Li, Yaoyao
    and Luviano, Jennifer
    and McConoughey, Stephen
    and Mortrud, Marty T.
    and Naeemi, Maitham
    and Ng, Lydia
    and Oh, Seung Wook
    and Ouellette, Benjamin
    and Shen, Elise
    and Sorensen, Staci A.
    and Wakeman, Wayne
    and Wang, Quanxin
    and Wang, Yun
    and Williford, Ali
    and Phillips, John W.
    and Jones, Allan R.
    and Koch, Christof
    and Zeng, Hongkui},
    title = {Hierarchical organization of cortical and thalamic connectivity},
    journal = {Nature},
    volume = 575,
    pages = {195-202},
    year = 2019,
    issn = {1476-4687},
    doi = {10.1038/s41586-019-1716-z},
    url = {http://www.ncbi.nlm.nih.gov/pubmed/26980805},
}

@article{10.1038/nn.3917,
    author = {Harris, Kenneth D. and Shepherd, Gordon M. G.},
    title = {The neocortical circuit: themes and variations},
    journal = {Nature Neuroscience},
    volume = 18,
    pages = {170-181},
    year = 2015,
    issn = {1546-1726},
    doi = {10.1038/nn.3917},
    url = {https://pubmed.ncbi.nlm.nih.gov/25622573/}
}
@article{10.1016/j.neuron.2014.08.025,
  author = {Wilson, Charles J},
  title = {The Sensory Striatum},
  journal = {Neuron},
  volume = 83,
  pages = {999-1001},
  issue = 5,
  issn = {0896-6273},
  year = 2014,
  doi = {10.1016/j.neuron.2014.08.025},
  url = {https://doi.org/10.1016/j.neuron.2014.08.025}
}

@article{10.1038/srep20072,
  author = {Jeong, Minju
    and Kim, Yongsoo
    and Kim, Jeongjin
    and Ferrante, Daniel D.
    and Mitra, Partha P.
    and Osten, Pavel
    and Kim, Daesoo},
  journal = {Scientific Reports},
  volume = 6,
  pages = 20072,
  title = {Comparative three-dimensional connectome map of motor cortical projections in the mouse brain},
  volume = 1,
  year = 2016,
  doi = {10.1038/srep20072},
  url = {https://doi.org/10.1038/srep20072},
}

@article{10.1016/j.neuron.2019.02.010,
  author = {Bennett, C.
    and Gale, S. D.
    and Garrett, M. E.
    and Newton, M. L.
    and Callaway, E. M.
    and Murphy, G. J.
    and Olsen, S. R.},
  journal = {Neuron},
  volume = 102,
  title = {Higher-Order Thalamic Circuits Channel Parallel Streams of Visual Information in Mice},
  issn = {0896-6273 (Print)
    and 0896-6273},
  pages = {477-492.e5},
  year = 2019,
  doi = {10.1016/j.neuron.2019.02.010},
  url = {https://pmc.ncbi.nlm.nih.gov/articles/PMC8638696/}
}

@article{10.1002/cne.21782,
  author = {Alloway, K. D.
    and Olson, M. L.
    and Smith, J. B.},
  journal = {J Comp Neurol},
  volume = 510,
  title = {Contralateral corticothalamic projections from MI whisker cortex: potential route for modulating hemispheric interactions},
  issn = {0021-9967 (Print)
    and 0021-9967},
  pages = {100-16},
  year = 2007,
  doi = {10.1002/cne.21782},
  url = {https://pmc.ncbi.nlm.nih.gov/articles/PMC2504743/}
}

@article{10.1016/j.cell.2019.01.047,
  author = {Winnubst, J.
    and Bas, E.
    and Ferreira, T. A.
    and Wu, Z.
    and Economo, M. N.
    and Edson, P.
    and Arthur, B. J.
    and Bruns, C.
    and Rokicki, K.
    and Schauder, D.
    and Olbris, D. J.
    and Murphy, S. D.
    and Ackerman, D. G.
    and Arshadi, C.
    and Baldwin, P.
    and Blake, R.
    and Elsayed, A.
    and Hasan, M.
    and Ramirez, D.
    and Dos Santos, B.
    and Weldon, M.
    and Zafar, A.
    and Dudman, J. T.
    and Gerfen, C. R.
    and Hantman, A. W.
    and Korff, W.
    and Sternson, S. M.
    and Spruston, N.
    and Svoboda, K.
    and Chandrashekar, J.},
  journal = {Cell},
  volume = 179,
  issue = 1,
  pages = {268-281.e13},
  title = {Reconstruction of 1,000 Projection Neurons Reveals New Cell Types and Organization of Long-Range Connectivity in the Mouse Brain},
  issn = {0092-8674 (Print)
    and 0092-8674},
  year = 2019,
  doi = {10.1016/j.cell.2019.01.047},
  url = {https://pubmed.ncbi.nlm.nih.gov/31495573/},
}

@article{10.1016/S0306-4522,
  author = {Jones, E. G.},
  journal = {Neuroscience},
  volume = 85,
  issue = 2,
  pages = {331-345},
  title = {Viewpoint: the core and matrix of thalamic organization},
  issn = {0306-4522},
  year = 1998,
  doi = {10.1016/S0306-4522(97)00581-2},
  url = {https://pubmed.ncbi.nlm.nih.gov/9622234/}
}

@article{10.1093/cercor/bhm211,
  author = {Landisman, C. E.
    and Connors, B. W.},
  journal = {Cereb Cortex},
  volume = 17,
  issue = 12,
  pages = {2853-65},
  title = {VPM and PoM nuclei of the rat somatosensory thalamus: intrinsic neuronal properties and corticothalamic feedback},
  issn = {1047-3211},
  year = 2007,
  doi = {10.1093/cercor/bhm211},
  url = {https://www.sci-hub.ru/10.1093/cercor/bhm025}
}


@article{10.3389/fninf.2014.00038,
  author = {Vella, Michael and Cannon, Robert C. and Crook, Sharon and Davison, Andrew P. and Ganapathy, Gautham and Robinson, Hugh P. C. and Silver, R. Angus and Gleeson, Padraig},
  title = {libNeuroML and PyLEMS: using Python to combine procedural and declarative modeling approaches in computational neuroscience},
  journal = {Frontiers in Neuroinformatics},
  volume = {8},
  issn = {1662-5196},
  abstract = {NeuroML is an XML-based model description language, which provides a powerful common data format for defining and exchanging models of neurons and neuronal networks. In the latest version of NeuroML, the structure and behavior of ion channel, synapse, cell, and network model descriptions are based on underlying definitions provided in LEMS, a domain-independent language for expressing hierarchical mathematical models of physical entities. While declarative approaches for describing models have led to greater exchange of model elements among software tools in computational neuroscience, a frequent criticism of XML-based languages is that they are difficult to work with directly. Here we describe two APIs (Application Programming Interfaces) written in Python (http://www.python.org), which simplify the process of developing and modifying models expressed in NeuroML and LEMS. The libNeuroML API provides a Python object model with a direct mapping to all NeuroML concepts defined by the NeuroML Schema, which facilitates reading and writing the XML equivalents. In addition, it offers a memory-efficient, array-based internal representation, which is useful for handling large-scale connectomics data. The libNeuroML API also includes support for performing common operations that are required when working with NeuroML documents. Access to the LEMS data model is provided by the PyLEMS API, which provides a Python implementation of the LEMS language, including the ability to simulate most models expressed in LEMS. Together, libNeuroML and PyLEMS provide a comprehensive solution for interacting with NeuroML models in a Python environment.},
  year = {2014},
  doi = {10.3389/fninf.2014.00038},
  url = {https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2014.00038},
}

@article{peixoto_graph-tool_2014,
  author = {Peixoto, Tiago P.},
  title = {The graph-tool python library},
  journal = {figshare},
  year = {2014},
  doi = {10.6084/m9.figshare.1164194},
  url = {http://figshare.com/articles/graph_tool/1164194},
}

@article{Gewaltig:NEST,
  author  = {Marc-Oliver Gewaltig and Markus Diesmann},
  title   = {NEST (NEural Simulation Tool)},
  journal = {Scholarpedia},
  year    = {2007},
  volume  = {2},
  pages   = {1430},
  number  = {4},
  doi     = {10.4249/scholarpedia.1430},
  url     = {http://www.scholarpedia.org/article/NEST_(NEural_Simulation_Tool)},
}

@article{Cannon2014,
  author    = {Robert C. Cannon and Padraig Gleeson and Sharon Crook and Gautham Ganapathy and Boris Marin and Eugenio Piasini and R. Angus Silver},
  title     = {{LEMS}: a language for expressing complex biological models in concise and hierarchical form and its use in underpinning {NeuroML} 2},
  doi       = {10.3389/fninf.2014.00079},
  volume    = {8},
  journal   = {Frontiers in Neuroinformatics},
  publisher = {Frontiers Media {SA}},
  year      = {2014},
}

@article{Vella2014,
  author       = {Vella, Michael and Cannon, Robert C. and Crook, Sharon and Davison, Andrew P. and Ganapathy, Gautham and Robinson, Hugh P. C. and Silver, R. Angus and Gleeson, Padraig},
  title        = {libNeuroML and PyLEMS: using Python to combine procedural and declarative modeling approaches in computational neuroscience.},
  doi          = {10.3389/fninf.2014.00038},
  pages        = {38},
  volume       = {8},
  journal      = {Frontiers in neuroinformatics},
  year         = {2014},
}

@article{10.3389/fninf.2022.835657,
  author       = {Layer, Moritz  and Senk, Johanna  and Essink, Simon  and van Meegen, Alexander  and Bos, Hannah  and Helias, Moritz },
  title        = {NNMT: Mean-Field Based Analysis Tools for Neuronal Network Models},
  journal      = {Frontiers in Neuroinformatics},
  volume       = {16},
  year         = {2022},
  url          = {https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.835657},
  doi          = {10.3389/fninf.2022.835657},
  issn         = {1662-5196},
  abstract     = {Mean-field theory of neuronal networks has led to numerous advances in our analytical and intuitive understanding of their dynamics during the past decades. In order to make mean-field
                  based analysis tools more accessible, we implemented an extensible, easy-to-use open-source
                  Python toolbox that collects a variety of mean-field methods for the leaky integrate-and-fire neuron
                  model. The Neuronal Network Mean-field Toolbox (NNMT) in its current state allows for estimating
                  properties of large neuronal networks, such as firing rates, power spectra, and dynamical stability
                  in mean-field and linear response approximation, without running simulations. In this article, we
                  describe how the toolbox is implemented, show how it is used to reproduce results of previous
                  studies, and discuss different use-cases, such as parameter space explorations, or mapping
                  different network models. Although the initial version of the toolbox focuses on methods for leaky
                  integrate-and-fire neurons, its structure is designed to be open and extensible. It aims to provide
                  a platform for collecting analytical methods for neuronal network model analysis, such that the
                  neuroscientific community can take maximal advantage of them.}
}

@article{10.1038/s41598-018-38246-3,
  author  = {Parr, Thomas and Markovic, Dimitrije and Kiebel, Stefan J. and Friston, Karl J.},
  title   = {Neuronal message passing using Mean-field, Bethe, and Marginal approximations},
  journal = {Scientific Reports},
  volume  = {9},
  issue   = {1},
  pages   = {1889},
  year    = {2019},
  url     = {https://doi.org/10.1038/s41598-018-38246-3},
  doi     = {10.1038/s41598-018-38246-3},
  issn    = {2045-2322}
}

@article{10.1038/s41467-025-58717-2,
  author  = {Russo, Simone and Claar, Leslie D. and Furregoni, Giulia and
            Marks, Lydia C. and Krishnan, Giri and Zauli, Flavia Maria and
            Hassan, Gabriel and Solbiati, Michela and d’Orio, Piergiorgio and
            Mikulan, Ezequiel and Sarasso, Simone and Rosanova, Mario and
            Sartori, Ivana and Bazhenov, Maxim and Pigorini, Andrea and
            Massimini, Marcello and Koch, Christof and Rembado, Irene},
  title   = {Thalamic feedback shapes brain responses evoked by cortical stimulation in mice and humans},
  journal = {Nature Communications},
  volume  = {16},
  issue   = {1},
  pages   = {3627},
  year    = {2025},
  url     = {https://doi.org/10.1038/s41467-025-58717-2},
  doi     = {10.1038/s41467-025-58717-2},
  issn    = {2041-1723}
}

@article{10.1038/s41593-025-01874-w,
  author  = {Vega-Zuniga, Tomas and
            Sumser, Anton and
            Symonova, Olga and
            Koppensteiner, Peter and
            Schmidt, Florian H. and
            Joesch, Maximilian},
  title   = {A thalamic hub-and-spoke network enables visual perception during action by coordinating visuomotor dynamics},
  journal = {Nature Neuroscience},
  volume  = {28},
  issue   = {3},
  pages   = {627-639},
  year    = {2025},
  url     = {https://doi.org/10.1038/s41593-025-01874-w},
  doi     = {10.1038/s41593-025-01874-w},
  issn    = {1546-1726}
}


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License

Copyright (c) 2026 [Author Name]

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

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time
import warnings
from itertools import cycle, islice
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_circles, make_classification, make_moons, make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from pyneuroml.pynml import read_neuroml2_file
from joblib import Parallel, delayed
# from sklearn.utils.parallel import Parallel, delayed
import json
import os
import glob

def main():
    json_files = glob.glob('/home/leo520/pynml/analysis_out/*_summary.json')
    # Create plots directory if it doesn't exist
    plots_dir = "/home/leo520/pynml/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    for json_file in json_files:
        print(f"Found JSON file: {json_file}")
        # Use the first JSON file found (you may want to modify this logic based on your needs)
        # json_file = json_files[0]
        with open(json_file, 'r') as f:
            stats = json.load(f)
            # Extract network ID from filename - remove last part after splitting by underscore
            filename = json_file.split('/')[-1]  # Get just the filename, not the full path
            basic = '_'.join(filename.split(".")[0].split("_")[0:-1])
            print(f" {basic}")
        # Changed from cell_counts to population_counts
        n_samples = stats['population_counts']['total_populations']
        print(f"n_samples (neural diversity): {n_samples}")

        ele_ee_conductances = stats['electrical_conductances']['EE']
        ele_ii_conductances = stats['electrical_conductances']['II']
        EE_contacts = stats['synaptic_contacts']['EE']
        EI_contacts = stats['synaptic_contacts']['EI']
        IE_contacts = stats['synaptic_contacts']['IE']
        II_contacts = stats['synaptic_contacts']['II']
        total_ee_connections = ele_ee_conductances + EE_contacts
        total_ii_connections = ele_ii_conductances + II_contacts
        print(f"EE_contacts : {stats['synaptic_contacts']['EE']}")
        print(f"EI_contacts: {stats['synaptic_contacts']['EI']}")
        print(f"IE_contacts: {stats['synaptic_contacts']['IE']}")
        print(f"II_contacts: {stats['synaptic_contacts']['II']}")
        print(f"ele_ee_conductances : {stats['electrical_conductances']['EE']}")
        print(f"ele_ii_conductances: {stats['electrical_conductances']['II']}")
       
        # noise represents the "neural network stability" 
        # High synaptic connections = more chemical signaling = more complex dynamics = more noise
        # High electrical connections = more direct signaling = more synchronized = less noise
        synaptic_activity = EI_contacts + EE_contacts - IE_contacts - II_contacts
        electrical_stability = ele_ee_conductances - ele_ii_conductances
        total_activity = synaptic_activity + electrical_stability
        
        # More synaptic activity = more complex dynamics (noise), more electrical = more stability (less noise)
        if total_activity > 0:
            # Inverse relationship - more electrical = less noise
            noise = min(0.1 + (synaptic_activity / total_activity) * 0.9, 0.95)
        else:
            noise = 0.05  # Default moderate noise
        print(f"network dynamics complexity (noise): {noise}")
        
        # factor represents the "network synchronization tendency"
        # High electrical = more synchronized = lower factor
        # High synaptic = more asynchronous = higher factor
        if total_activity > 0:
            # Direct relationship - more synaptic = higher factor
            factor = min(0.05 + (synaptic_activity / total_activity) * 0.9, 0.95)
        else:
            factor = 0.5  # Balanced synchronization
        print(f"synchronization tendency (factor): {factor}")
        
        # seed represents the "excitation/inhibition balance"
        # Based on the ratio of inhibitory to excitatory synaptic connections
        # In neuroscience, this is typically GABA (inhibitory) to AMPA/NMDA (excitatory) ratio
        # But in our case, we'll use inh_inputs to exc_inputs ratio as a proxy
        if total_ee_connections > 0:
            # E/I ratio - higher values mean more excitation relative to inhibition
            excitation_inhibition_ratio = total_ee_connections / total_ii_connections
            # Convert to seed value (0-41)
            seed = int((excitation_inhibition_ratio * 420) )
        else:
            seed = 30 if total_ii_connections > 0 else 0  # All inhibitory or no inputs
            
        random_state = np.random.RandomState(seed)
        print(f"E/I balance ratio (seed): {seed}")
        
        print(f"Network characteristics:")
        print(f"  - Diversity: {n_samples} populations")
        print(f"  - Dynamics complexity: {noise:.2f} (synaptic: {synaptic_activity}, electrical: {electrical_stability})")
        print(f"  - Synchronization: {factor:.2f} ({'more asynchronous' if factor > 0.5 else 'more synchronized'})")
        print(f"  - E/I balance: {excitation_inhibition_ratio:.2f} ({'excitatory dominated' if excitation_inhibition_ratio > 1 else 'inhibitory dominated' if excitation_inhibition_ratio < 1 else 'balanced'})")


        # Limit the number of samples to prevent memory issues
        max_samples = min(n_samples, 3000)  # Reduce to 3000 samples to prevent memory issues
        print(f"Using {max_samples} samples for clustering (down from {n_samples}) to prevent memory issues")

        noisy_circles = datasets.make_circles(n_samples=max_samples, factor=factor, noise=noise, random_state=random_state)
        noisy_moons = datasets.make_moons(n_samples=max_samples, noise=noise, random_state=random_state)
        blobs = datasets.make_blobs(n_samples=max_samples, random_state=seed)
        rng = np.random.RandomState(seed)
        no_structure = rng.rand(max_samples, 2), None

        # Anisotropicly distributed data
        random_state = random_state
        X, y = datasets.make_blobs(n_samples=max_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        aniso = (X_aniso, y)

        # blobs with varied variances
        varied = datasets.make_blobs(n_samples=max_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

        # ============
        # Set up cluster parameters
        # ============
        plt.figure(figsize=(9 * 2 + 3, 13))
        plt.subplots_adjust(
            left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
        )

        plot_num = 1

        default_base = {
            "quantile": 0.3,
            "eps": 0.3,
            "damping": 0.9,
            "preference": -200,
            "n_neighbors": 3,
            "n_clusters": 3,
            "min_samples": 7,
            "xi": 0.05,
            "min_cluster_size": 0.1,
            "allow_single_cluster": True,
            "hdbscan_min_cluster_size": 15,
            "hdbscan_min_samples": 3,
            "random_state": 42,
        }

        clustering_datasets = [
            (
                noisy_circles,
                {
                    "damping": 0.77,
                    "preference": -240,
                    "quantile": 0.2,
                    "n_clusters": 2,
                    "min_samples": 7,
                    "xi": 0.08,
                },
            ),
            (
                noisy_moons,
                {
                    "damping": 0.75,
                    "preference": -220,
                    "n_clusters": 2,
                    "min_samples": 7,
                    "xi": 0.1,
                },
            ),
            (
                varied,
                {
                    "eps": 0.18,
                    "n_neighbors": 2,
                    "min_samples": 7,
                    "xi": 0.01,
                    "min_cluster_size": 0.2,
                },
            ),
            (
                aniso,
                {
                    "eps": 0.15,
                    "n_neighbors": 2,
                    "min_samples": 7,
                    "xi": 0.1,
                    "min_cluster_size": 0.2,
                },
            ),
            (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
            (no_structure, {}),
        ]

        for i_dataset, (dataset, algo_params) in enumerate(clustering_datasets):
            # update parameters with dataset-specific values
            params_data = default_base.copy()
            params_data.update(algo_params)

            X, y = dataset

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # estimate bandwidth for mean shift with smaller subset
            bandwidth_X = X[:min(300, X.shape[0]), :]  # Use at most 300 samples for bandwidth estimation
            bandwidth = cluster.estimate_bandwidth(bandwidth_X, quantile=params_data["quantile"])

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(X, n_neighbors=params_data["n_neighbors"], include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # ============
            # Create cluster objects
            # ============
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(
                n_clusters=params_data["n_clusters"],
                random_state=params_data["random_state"],
            )
            ward = cluster.AgglomerativeClustering(
                n_clusters=params_data["n_clusters"], linkage="ward", connectivity=connectivity
            )
            spectral = cluster.SpectralClustering(
                n_clusters=params_data["n_clusters"],
                eigen_solver="arpack",
                affinity="nearest_neighbors",
                random_state=params_data["random_state"],
            )
            dbscan = cluster.DBSCAN(eps=params_data["eps"])
            hdbscan = cluster.HDBSCAN(
                min_samples=params_data["hdbscan_min_samples"],
                min_cluster_size=params_data["hdbscan_min_cluster_size"],
                allow_single_cluster=params_data["allow_single_cluster"],
            )
            optics = cluster.OPTICS(
                min_samples=params_data["min_samples"],
                xi=params_data["xi"],
                min_cluster_size=params_data["min_cluster_size"],
            )
            average_linkage = cluster.AgglomerativeClustering(
                linkage="average",
                metric="cityblock",
                n_clusters=params_data["n_clusters"],
                connectivity=connectivity,
            )
            birch = cluster.Birch(n_clusters=params_data["n_clusters"])
            gmm = mixture.GaussianMixture(
                n_components=params_data["n_clusters"],
                covariance_type="full",
                random_state=params_data["random_state"],
            )
            
            # Create clustering algorithms list
            clustering_algorithms = [
                ("MiniBatch\nKMeans", two_means),
                ("MeanShift", ms),
                ("Spectral\nClustering", spectral),
                ("Ward", ward),
                ("Agglomerative\nClustering", average_linkage),
                ("DBSCAN", dbscan),
                ("HDBSCAN", hdbscan),
                ("OPTICS", optics),
                ("BIRCH", birch),
                ("Gaussian\nMixture", gmm),
            ]
            
            # Add Affinity Propagation only for small datasets
            if max_samples <= 300:
                affinity_propagation = cluster.AffinityPropagation(
                    damping=params_data["damping"],
                    preference=params_data["preference"],
                    random_state=params_data["random_state"],
                )
                clustering_algorithms.insert(1, ("Affinity\nPropagation", affinity_propagation))

            for name, algorithm in clustering_algorithms:
                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the "
                        "connectivity matrix is [0-9]{1,2}"
                        " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding"
                        " may not work as expected.",
                        category=UserWarning,
                    )
                    
                    # Skip Affinity Propagation for large datasets to prevent memory issues
                    if name == "Affinity\nPropagation" and max_samples > 300:
                        print(f"Skipping {name} for dataset {basic} due to memory constraints")
                        t1 = time.time()
                        y_pred = np.zeros(X.shape[0], dtype=int)  # Dummy prediction with correct type
                    else:
                        algorithm.fit(X)
                        t1 = time.time()
                        if hasattr(algorithm, "labels_"):
                            y_pred = algorithm.labels_.astype(int)
                        else:
                            y_pred = algorithm.predict(X)

                plt.subplot(len(clustering_datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)

                colors = np.array(
                    list(
                        islice(
                            cycle(
                                [
                                    "#eb2a2d",
                                    "#157792",
                                    "#435636",
                                    "#c6d30f",
                                    "#df1d84",
                                    "#74E90D",
                                    "#a51fb9",
                                    "#221F1F",
                                    "#0EA3E3",
                                ]
                            ),
                            int(max(y_pred) + 1),
                        )
                    )
                )
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(
                    0.99,
                    0.01,
                    ("%.2fs" % (t1 - t0)).lstrip("0"),
                    transform=plt.gca().transAxes,
                    size=15,
                    horizontalalignment="right",
                )
                plot_num += 1
        plt.suptitle(f"Clustering algorithms on {basic} dataset", size=20, weight='bold')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
        plt.tight_layout()
        # Save to dedicated plots directory
        plot_filename = f"/home/leo520/pynml/plots/{basic}_clustering.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved clustering plot to: {plot_filename}")


if __name__ == '__main__':
    main()
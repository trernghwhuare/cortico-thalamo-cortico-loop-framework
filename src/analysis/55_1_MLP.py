# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from pyneuroml.pynml import read_neuroml2_file
import json
import os
import glob


h = 0.005  # step size in the mesh
alphas = np.logspace(-1, 1, 5)
RandomState = 42
classifiers = []
names = []
for alpha in alphas:
    classifiers.append(
        make_pipeline(
            StandardScaler(),
            MLPClassifier(
                activation="relu",
                solver="adam",
                batch_size=10,
                learning_rate='adaptive',
                learning_rate_init=0.0001,
                power_t=0.05,
                alpha=alpha,
                random_state=RandomState,
                momentum=0.6,
                max_iter=2000,
                early_stopping=True,
                hidden_layer_sizes=[5, 25],
                epsilon=1e-7,
            ),
        )
    )
    names.append(f"alpha {alpha:.2f}")

def main():
    json_files = glob.glob('/home/leo520/pynml/analysis_out/*_summary.json')
    plots_dir = "/home/leo520/pynml/plots"
    os.makedirs(plots_dir, exist_ok=True)

    for json_file in json_files:
        print(f"Found JSON file: {json_file}")
        # Use the first JSON file found (you may want to modify this logic based on your needs)
        # json_file = json_files[0]
        with open(json_file, 'r') as f:
            stats = json.load(f)
            filename = json_file.split('/')[-1]  # Get just the filename, not the full path
            basic = '_'.join(filename.split(".")[0].split("_")[0:-1])
            print(f" {basic}")
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
        synaptic_activity = EI_contacts + EE_contacts +IE_contacts + II_contacts
        electrical_stability = ele_ee_conductances + ele_ii_conductances
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


    
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=0,
            n_clusters_per_class=1
        )
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        datasets = [
            make_moons(noise=noise, random_state=0),
            make_circles(noise=noise, factor=factor, random_state=1),
            linearly_separable,
        ]

        figure = plt.figure(figsize=(17, 9))
        i = 1
        for X, y in datasets:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=random_state
            )
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            cm = plt.cm.tab20c
            cm_bright = ListedColormap(["#DEDB07", "#9B52BF"])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1

            for name, clf in zip(names, classifiers):
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
                else:
                    Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="black", s=25)
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="black", s=25)
                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                ax.set_title(name)
                ax.text(xx.max() - 0.3, yy.min() + 0.3, f"{score:.3f}".lstrip("0"), size=15, horizontalalignment="right")
                i += 1

        figure.subplots_adjust(left=0.02, right=0.98)
        plt.suptitle(f"Net: {basic}")
        # Save to dedicated plots directory
        plot_filename = f"/home/leo520/pynml/plots/{basic}_mlp_classification.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved MLP classification plot to: {plot_filename}")
        
if __name__ == '__main__':
    main()
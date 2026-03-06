# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.utils.parallel import Parallel, delayed
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pyneuroml.pynml import read_neuroml2_file
from joblib import Parallel, delayed
import json
import os
import glob


h = 0.01  # step size in the mesh

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(n_neighbors=6),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def main():
    json_files = glob.glob('/home/leo520/pynml/analysis_out/*_summary.json')
    
    # Create plots directory if it doesn't exist
    plots_dir = "/home/leo520/pynml/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    for json_file in json_files:
        print(f"Found JSON file: {json_file}")
        # json_file = json_files[0]
        with open(json_file, 'r') as f:
            stats = json.load(f)
            # Extract network ID from filename - remove last part after splitting by underscore
            filename = json_file.split('/')[-1]  # Get just the filename, not the full path
            basic = '_'.join(filename.split(".")[0].split("_")[0:-1])
            print(f"Network ID: {basic}")
        population_samples = stats['population_counts']['total_populations']
        ele_ee_conductances = stats['electrical_conductances']['EE']
        ele_ii_conductances = stats['electrical_conductances']['II']
        EE_contacts = stats['synaptic_contacts']['EE']
        EI_contacts = stats['synaptic_contacts']['EI']
        IE_contacts = stats['synaptic_contacts']['IE']
        II_contacts = stats['synaptic_contacts']['II']
        total_ee_connections = ele_ee_conductances + EE_contacts
        total_ii_connections = ele_ii_conductances + II_contacts
        print(f"population_samples: {stats['population_counts']['total_populations']}")
        print(f"exc_inputs: {stats['inputs']['exc']}")
        print(f"inh_inputs: {stats['inputs']['inh']}")
        print(f"EE_contacts: {stats['synaptic_contacts']['EE']}")
        print(f"EI_contacts: {stats['synaptic_contacts']['EI']}")
        print(f"IE_contacts: {stats['synaptic_contacts']['IE']}")
        print(f"II_contacts: {stats['synaptic_contacts']['II']}")

        n_samples = min(max(population_samples, 50), 7000)
        print(f"n_samples (neural diversity): {n_samples}")
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
        print(f"  - Diversity: {population_samples} populations")
        print(f"  - Dynamics complexity: {noise:.2f} (synaptic: {synaptic_activity}, electrical: {electrical_stability})")
        print(f"  - Synchronization: {factor:.2f} ({'asynchronous' if factor > 0.5 else 'synchronized'})")
        print(f"  - E/I balance: {excitation_inhibition_ratio:.2f} ({'excitatory dominated' if excitation_inhibition_ratio > 1 else 'inhibitory dominated' if excitation_inhibition_ratio < 1 else 'balanced'})")
        print(f"  - EE connections: {total_ee_connections} (syn: {EE_contacts}, ele: {ele_ee_conductances})")
        print(f"  - II connections: {total_ii_connections} (syn: {II_contacts}, ele: {ele_ii_conductances})")
        print("-----------------------------------------------------")

        X, y = make_classification(
            n_samples=n_samples,
            n_features=2, 
            n_redundant=0, 
            n_informative=2, 
            random_state=1, 
            n_clusters_per_class=1
        )
        rng = random_state
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        datasets = [
            make_moons(noise=noise, random_state=0),
            make_circles(noise=noise, factor=factor, random_state=1),
            linearly_separable,
        ]
        figure = plt.figure(figsize=(27, 9))
        i = 1
        # Create animations for each dataset instead of static plots
        for ds_cnt, ds in enumerate(datasets):
            # Preprocess dataset, split into training and test part
            X, y = ds
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=42
            )

            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            cm = plt.get_cmap('twilight')
            cm_bright = ListedColormap(["#FFC800", "#0000FF"])

            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            if ds_cnt == 0:
                ax.set_title("Input data")
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
            # Plot the testing points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            i += 1

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
                clf = make_pipeline(StandardScaler(), clf)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                DecisionBoundaryDisplay.from_estimator(
                    clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
                )
                # Plot the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
                # Plot the testing points
                ax.scatter(X_test[:, 0],X_test[:, 1],c=y_test,cmap=cm_bright,edgecolors="k",alpha=0.6)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks(())
                ax.set_yticks(())
                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(x_max - 0.3,y_min + 0.3,("%.2f" % score).lstrip("0"),size=15,horizontalalignment="right")
                i += 1

        plt.tight_layout()
        figure.subplots_adjust(left=0.02, right=0.98)
        plt.suptitle(f"Net: {basic}")
        plt.subplots_adjust(top=0.90)
        # Save to dedicated plots directory
        plot_filename = f"{plots_dir}/{basic}_cc_classification.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved CC classification plot to: {plot_filename}")


if __name__ == '__main__':
    main()
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils._testing import ignore_warnings
from pyneuroml.pynml import read_neuroml2_file
from joblib import Parallel, delayed
import json
import os
import glob


h = 0.02  # step size in the mesh

def get_name(estimator):
    name = estimator.__class__.__name__
    if name == "Pipeline":
        name = [get_name(est[1]) for est in estimator.steps]
        name = " + ".join(name)
    return name

classifiers = [
    (
        make_pipeline(StandardScaler(), LogisticRegression(random_state=0)),
        {"logisticregression__C": np.logspace(-1, 1, 3)},
    ),
    (
        make_pipeline(StandardScaler(), LinearSVC(random_state=0)),
        {"linearsvc__C": np.logspace(-1, 1, 3)},
    ),
    (
        make_pipeline(
            StandardScaler(),
            KBinsDiscretizer(
                encode="onehot", quantile_method="averaged_inverted_cdf", random_state=0
            ),
            LogisticRegression(random_state=0),
        ),
        {
            "kbinsdiscretizer__n_bins": np.arange(5, 8),
            "logisticregression__C": np.logspace(-1, 1, 3),
        },
    ),
    (
        make_pipeline(
            StandardScaler(),
            KBinsDiscretizer(
                encode="onehot", quantile_method="averaged_inverted_cdf", random_state=0
            ),
            LinearSVC(random_state=0),
        ),
        {
            "kbinsdiscretizer__n_bins": np.arange(5, 8),
            "linearsvc__C": np.logspace(-1, 1, 3),
        },
    ),
    (
        make_pipeline(
            StandardScaler(), GradientBoostingClassifier(n_estimators=5, random_state=0)
        ),
        {"gradientboostingclassifier__learning_rate": np.logspace(-2, 0, 5)},
    ),
    (
        make_pipeline(StandardScaler(), SVC(random_state=0)),
        {"svc__C": np.logspace(-1, 1, 3)},
    ),
]

names = [get_name(e).replace("StandardScaler + ", "") for e, _ in classifiers]

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
    
        datasets = [
            make_moons(n_samples=n_samples, noise=noise, random_state=0),
            make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=1),
            make_classification(
                n_samples=n_samples,
                n_features=2,
                n_redundant=0,
                n_informative=2,
                random_state=2,
                n_clusters_per_class=1,
            ),
        ]

        fig, axes = plt.subplots(
            nrows=len(datasets), ncols=len(classifiers) + 1, figsize=(21, 9)
        )

        cm = plt.cm.Spectral
        cm_bright = ListedColormap(["#f0de20", "#35E2BA"])

        # iterate over datasets
        for ds_cnt, (X, y) in enumerate(datasets):
            print(f"\ndataset {ds_cnt}\n---------")

            # split into training and test part
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, random_state=random_state
            )

            # create the grid for background colors
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            # plot the dataset first
            ax = axes[ds_cnt, 0]
            if ds_cnt == 0:
                ax.set_title("Input data")
            # plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
            # and testing points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
            )
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

            # iterate over classifiers
            for est_idx, (name, (estimator, param_grid)) in enumerate(zip(names, classifiers)):
                ax = axes[ds_cnt, est_idx + 1]

                clf = GridSearchCV(estimator=estimator, param_grid=param_grid)
                with ignore_warnings(category=ConvergenceWarning):
                    clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                print(f"{name}: {score:.2f}")

                # plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]*[y_min, y_max].
                if hasattr(clf, "decision_function"):
                    Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
                else:
                    Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

                # put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

                # plot the training points
                ax.scatter(
                    X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
                )
                # and testing points
                ax.scatter(
                    X_test[:, 0],
                    X_test[:, 1],
                    c=y_test,
                    cmap=cm_bright,
                    edgecolors="k",
                    alpha=0.6,
                )
                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())

                if ds_cnt == 0:
                    ax.set_title(name.replace(" + ", "\n"))
                ax.text(
                    0.95,
                    0.06,
                    (f"{score:.2f}").lstrip("0"),
                    size=15,
                    bbox=dict(boxstyle="round", alpha=0.8, facecolor="white"),
                    transform=ax.transAxes,
                    horizontalalignment="right",
                )

        plt.tight_layout()
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88)
        # Layer title (main suptitle)
        fig.suptitle(f"Network: {basic}", y=1.08, fontsize="xx-large")
    
        # Column group subtitles
        suptitles = [
            f"Network: {basic}",
            "Linear classifiers",
            "Feature discretization and linear classifiers",
            "Non-linear classifiers",
        ]
        col_indices = [0, 1, 3, 5]
        for i, suptitle in zip(col_indices, suptitles):
            # Get the axis for the first row of each group
            ax = axes[0, i]
            # Get the position of the axis in figure coordinates
            pos = ax.get_position()
            x = pos.x0 + pos.width / 2
            # Place the subtitle above the axis, in figure coordinates
            fig.text(
                x, 0.96, suptitle,
                ha="center", va="bottom", fontsize="x-large", weight="bold"
            )
        # Save to dedicated plots directory
        plot_filename = f"/home/leo520/pynml/plots/{basic}_fd_classification.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved FD classification plot to: {plot_filename}")

if __name__ == '__main__':
    main()
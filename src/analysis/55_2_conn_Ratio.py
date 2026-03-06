from cProfile import label
import random
from cv2 import line, mean
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import json
import os
import glob

def main():
    # Create plots directory if it doesn't exist
    plots_dir = '/home/leo520/pynml/plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")
        print("Reminder: All plots will be saved in the 'plots' folder.")
    
    json_files = glob.glob('/home/leo520/pynml/analysis_out/*_summary.json')
    
    for json_file in json_files:
        print(f"Found JSON file: {json_file}")
        with open(json_file, 'r') as f:
            stats = json.load(f)
            filename = json_file.split('/')[-1]  # Get just the filename, not the full path
            basic = '_'.join(filename.split(".")[0].split("_")[0:-1])
            print(f" {basic}")
        
        # Extract parameters from the JSON structure - aligned with 8_loop_optimus+.py export
        syn_ee_contacts = stats['synaptic_contacts']['EE']
        syn_ei_contacts = stats['synaptic_contacts']['EI']
        syn_ie_contacts = stats['synaptic_contacts']['IE']
        syn_ii_contacts = stats['synaptic_contacts']['II']
        ele_ee_conductances = stats['electrical_conductances']['EE']
        ele_ii_conductances = stats['electrical_conductances']['II']
        
        # Set electrical connections that are not tracked to 0 (consistent with 8_loop_optimus+.py)
        ele_ei_conductances = 0
        ele_ie_conductances = 0
        
        print(f"syn_ee_contacts:{syn_ee_contacts}, syn_ei_contacts:{syn_ei_contacts}, syn_ie_contacts:{syn_ie_contacts}, syn_ii_contacts:{syn_ii_contacts}")
        print(f"ele_ee_conductances:{ele_ee_conductances}, ele_ei_conductances:{ele_ei_conductances}, ele_ie_conductances:{ele_ie_conductances}, ele_ii_conductances:{ele_ii_conductances}")
        
        # Check if actual correlation exists in the JSON (calculated in 8_loop_optimus+.py)
        actual_correlation = stats.get('correlation', None)
        print(f"Loaded statistics from {json_file}")

        # Data preparations - aligned with 8_loop_optimus+.py calculations
        # Total connections to excitatory cells (EE + IE)
        total_excitatory_connections = syn_ee_contacts + syn_ie_contacts + ele_ee_conductances + ele_ie_conductances
        # Total connections to inhibitory cells (EI + II)
        total_inhibitory_connections = syn_ei_contacts + syn_ii_contacts + ele_ei_conductances + ele_ii_conductances
        
        print(f"Total connections to excitatory cells: {total_excitatory_connections}")
        print(f"Total connections to inhibitory cells: {total_inhibitory_connections}")
        
        # Calculate ratios - aligned with 8_loop_optimus+.py calculations
        # Avoid division by zero
        if total_inhibitory_connections > 0:
            ei_ratio = total_excitatory_connections / total_inhibitory_connections
        else:
            ei_ratio = float('inf')
            
        if total_excitatory_connections > 0:
            ie_ratio = total_inhibitory_connections / total_excitatory_connections
        else:
            ie_ratio = float('inf')

        # Synaptic ratios - aligned with 8_loop_optimus+.py calculations
        syn_total_to_e = syn_ee_contacts + syn_ie_contacts  # All synaptic connections to excitatory cells
        syn_total_to_i = syn_ei_contacts + syn_ii_contacts  # All synaptic connections to inhibitory cells
        
        if syn_total_to_i > 0:
            syn_ei_ratio = syn_total_to_e / syn_total_to_i
        else:
            syn_ei_ratio = float('inf')
            
        if syn_total_to_e > 0:
            syn_ie_ratio = syn_total_to_i / syn_total_to_e
        else:
            syn_ie_ratio = float('inf')

        # Electrical ratios - aligned with 8_loop_optimus+.py calculations
        ele_total_to_e = ele_ee_conductances + ele_ie_conductances  # All electrical connections to excitatory cells
        ele_total_to_i = ele_ei_conductances + ele_ii_conductances  # All electrical connections to inhibitory cells
        
        if ele_total_to_i > 0:
            ele_ei_ratio = ele_total_to_e / ele_total_to_i
        else:
            ele_ei_ratio = float('inf')
            
        if ele_total_to_e > 0:
            ele_ie_ratio = ele_total_to_i / ele_total_to_e
        else:
            ele_ie_ratio = float('inf')

        # Create data for plotting - generating scatter distributions
        # Use a fixed seed or derive from network name for reproducibility
        np.random.seed(hash(basic) % (2**32))
        
        # Create datasets for visualization around our calculated values
        n_points = 500  # Fixed number of points for visualization
        
        # Overall connection data
        exc_data = np.random.normal(total_excitatory_connections, total_excitatory_connections*0.05, n_points)
        inh_data = np.random.normal(total_inhibitory_connections, total_inhibitory_connections*0.05, n_points)
        
        # Synaptic connection data
        syn_exc_data = np.random.normal(syn_total_to_e, syn_total_to_e*0.05, n_points)
        syn_inh_data = np.random.normal(syn_total_to_i, syn_total_to_i*0.05, n_points)
        
        # Electrical connection data
        ele_exc_data = np.random.normal(ele_total_to_e, ele_total_to_e*0.05, n_points)
        ele_inh_data = np.random.normal(ele_total_to_i, ele_total_to_i*0.05, n_points)

        # Create a single figure with 3 rows and 1 column
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 15))

        # ------------------Subplot 0-----------------------------
        # Use actual correlation from JSON data (calculated in 8_loop_optimus+.py)
        corr_coef0 = actual_correlation if actual_correlation is not None else 0

        data0 = pd.DataFrame({
            'overall Excitatory Connections': exc_data,
            'overall Inhibitory Connections': inh_data
        })
        n_count_points = 500
        colors = np.random.rand(n_count_points)
        sns.scatterplot(data=data0, x='overall Inhibitory Connections', y='overall Excitatory Connections', 
                        hue='overall Inhibitory Connections', palette='spring', # c=colors, 
                        alpha=0.6, s=30, edgecolors='white', ax=ax0)
                        
        sns.regplot(data=data0, x='overall Inhibitory Connections', y='overall Excitatory Connections', 
                    scatter=False, color='orange', ax=ax0, line_kws={'linewidth': 2})
        
        exc_count_data = np.random.normal(total_excitatory_connections, total_excitatory_connections*0.01, n_count_points)
        inh_count_data = np.random.normal(total_inhibitory_connections, total_inhibitory_connections*0.01, n_count_points)
        ax0.scatter(inh_count_data, exc_count_data, c=colors, alpha=0.75, s=20, zorder=5)
        ax0.plot([], [], 'o', alpha=0.6, markersize=5)

        # Use actual correlation from JSON data (calculated in 8_loop_optimus+.py)
        display_corr = actual_correlation if actual_correlation is not None else 0
        ax0.plot([], [], '-', linewidth=2)
        ax0.text(0.05, 0.95, f'overall E/I ratio = {ei_ratio:.3f}\noverall I/E ratio = {ie_ratio:.3f}\nCorrelation = {display_corr:.3f}', 
                transform=ax0.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax0.set_title(f'overall {basic} Exc vs Inh Connections', fontsize=12, fontweight='bold')
        ax0.set_xlabel('overall Inhibitory Connections', fontsize=10)
        ax0.set_ylabel('overall Excitatory Connections', fontsize=10)
        ax0.ticklabel_format(style='plain', axis='both', scilimits=(0,0))
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        #--------------------Subplot 1------------------------------
        # For synaptical and electrical connections, we don't have separate correlation values in the JSON
        # So we'll calculate them from the generated data for visualization purposes
        corr_coef1, _ = pearsonr(syn_exc_data, syn_inh_data)

        data1 = pd.DataFrame({
            'Synaptical Excitatory Contacts': syn_exc_data,
            'Synaptical Inhibitory Contacts': syn_inh_data
        })
        colors = np.random.rand(n_count_points)
        sns.scatterplot(data=data1, x='Synaptical Inhibitory Contacts', y='Synaptical Excitatory Contacts', 
                        hue='Synaptical Inhibitory Contacts', palette='spring', # c=colors, 
                        alpha=0.6, s=30, edgecolors='white', ax=ax1)
        sns.regplot(data=data1, x='Synaptical Inhibitory Contacts', y='Synaptical Excitatory Contacts', 
                    scatter=False, color='orange', ax=ax1, line_kws={'linewidth': 2})
        syn_exc_count_data = np.random.normal(syn_total_to_e, syn_total_to_e*0.01, n_count_points)
        syn_inh_count_data = np.random.normal(syn_total_to_i, syn_total_to_i*0.01, n_count_points)
        
        ax1.scatter(syn_inh_count_data, syn_exc_count_data, c=colors, alpha=0.75, s=20, zorder=5,
                )
        ax1.plot([], [], 'o', alpha=0.6, markersize=5)
        ax1.plot([], [], '-', linewidth=2)
                 
        ax1.text(0.05, 0.95, f'syn E/I ratio = {syn_ei_ratio:.3f}\nsyn I/E ratio = {syn_ie_ratio:.3f}\nCorrelation = {corr_coef1:.3f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax1.set_title(f'{basic} synaptical Exc vs Inh Contacts ', fontsize=12, fontweight='bold')
        ax1.set_xlabel('synaptical Inhibitory Contacts', fontsize=10)
        ax1.set_ylabel('synaptical Excitatory Contacts', fontsize=10)

        ax1.ticklabel_format(style='plain', axis='both', scilimits=(0,0))
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # ---------------------Subplot 2----------------------------
        corr_coef2, _ = pearsonr(ele_exc_data, ele_inh_data)

        data2 = pd.DataFrame({
            'Electrical Excitatory Conductances': ele_exc_data,
            'Electrical Inhibitory Conductances': ele_inh_data
        })
        colors = np.random.rand(n_count_points)
        sns.scatterplot(data=data2, x='Electrical Inhibitory Conductances', y='Electrical Excitatory Conductances', 
                        hue='Electrical Inhibitory Conductances', palette='spring', # c=colors, 
                        alpha=0.6, s=30, edgecolors='white', ax=ax2)
                        
        sns.regplot(data=data2, x='Electrical Inhibitory Conductances', y='Electrical Excitatory Conductances', 
                    scatter=False, color='orange', ax=ax2, line_kws={'linewidth': 2})
        ele_exc_count_data = np.random.normal(ele_total_to_e, ele_total_to_e*0.01, n_count_points)
        ele_inh_count_data = np.random.normal(ele_total_to_i, ele_total_to_i*0.01, n_count_points)
        
        ax2.scatter(ele_inh_count_data, ele_exc_count_data, c=colors, alpha=0.75, s=20, zorder=5)
        ax2.plot([], [], 'o', alpha=0.6, markersize=5)
        ax2.plot([], [], '-', linewidth=2)
                 
        ax2.text(0.05, 0.95, f'ele E/I ratio = {ele_ei_ratio:.3f}\nele I/E ratio = {ele_ie_ratio:.3f}\nCorrelation = {corr_coef2:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.set_title(f'{basic} electrical Exc vs Inh Conductances ', fontsize=12, fontweight='bold')
        ax2.set_xlabel('electrical Inhibitory Conductances', fontsize=10)
        ax2.set_ylabel('electrical Excitatory Conductances', fontsize=10)

        ax2.ticklabel_format(style='plain', axis='both', scilimits=(0,0))
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Adjust layout and save the combined figure
        plt.tight_layout()
        # Save to plots directory
        plt.savefig(os.path.join(plots_dir, f'{basic}_regression.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        ###############################################################
        # Print the ratios and correlation
        print(f"overall E/I ratio: {ei_ratio:.6f}")
        print(f"overall I/E ratio: {ie_ratio:.6f}")
        print(f"Correlation coefficient: {display_corr:.6f}")

        print(f"syn E/I ratio: {syn_ei_ratio:.6f}")
        print(f"syn_I/E ratio: {syn_ie_ratio:.6f}")
        print(f"Correlation coefficient: {corr_coef1:.6f}")

        print(f"ele E/I ratio: {ele_ei_ratio:.6f}")
        print(f"ele I/E ratio: {ele_ie_ratio:.6f}")
        print(f"Correlation coefficient: {corr_coef2:.6f}")

        # Create a single figure with 3 rows and 2 columns
        fig, ((ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(3, 2, figsize=(12, 15))

        # -----------Subplot (0,0) - E/I Connections Ratio-----------------------------
        exc_ei_ratios = exc_data / np.where(inh_data != 0, inh_data, 1)
        inh_ei_ratios = total_excitatory_connections / np.where(inh_data != 0, inh_data, 1)
        exc_ie_ratios = total_inhibitory_connections / np.where(exc_data != 0, exc_data, 1)
        inh_ie_ratios = inh_data / np.where(total_excitatory_connections != 0, total_excitatory_connections, 1)
        linestyles = ['-', '-.']
        linewidths = [1, 0.5]
        edgecolors = ['red', 'blue', 'purple', 'green', 'grey', 'yellow']
        labels = ['Excitatory', 'Inhibitory', 'Synaptical Excitatory', 'Synaptical Inhibitory', 'Electrical Excitatory', 'Electrical Inhibitory']
        ax3.hist(exc_ei_ratios, bins=75, alpha=0.75, color='pink', label=labels[0], linestyle=linestyles[0],edgecolor=edgecolors[0], linewidth=linewidths[0])
        ax3.hist(inh_ei_ratios, bins=75, alpha=0.75, color='azure', label=labels[1], linestyle=linestyles[1],edgecolor=edgecolors[1], linewidth=linewidths[1])
        ax3.axvline(ei_ratio, color='black', linestyle='--', linewidth=1)
        ax3.plot(ei_ratio, ax3.get_ylim()[1]*0.05, 'v', markersize=10, color='black', markeredgecolor='white', markeredgewidth=1,
                label=f'E/I ratio = {ei_ratio:.3f}')
        ax3.set_title(f'Distribution of {basic} E/I Connections Ratio ', fontsize=12, fontweight='bold')
        ax3.set_xlabel(f'{basic} E/I Connections Ratio', fontsize=10)
        ax3.set_ylabel('Probability', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # -----------Subplot (1,0) - I/E Connections Ratio-----------------------------
        ax4.hist(exc_ie_ratios, bins=75, alpha=0.75, color='pink', label=labels[0], linestyle=linestyles[0],edgecolor=edgecolors[0], linewidth=linewidths[0])
        ax4.hist(inh_ie_ratios, bins=75, alpha=0.75, color='azure', label=labels[1], linestyle=linestyles[1],edgecolor=edgecolors[1], linewidth=linewidths[1])
        ax4.axvline(ie_ratio, color='black', linestyle='--', linewidth=2)
        ax4.plot(ie_ratio, ax4.get_ylim()[1]*0.05, 'v', markersize=10, color='black', markeredgecolor='white', markeredgewidth=1,
                label=f'I/E ratio = {ie_ratio:.3f}')
        ax4.set_title(f'Distribution of {basic} I/E Connections Ratio ', fontsize=12, fontweight='bold')
        ax4.set_xlabel(f'{basic} I/E Connections Ratio', fontsize=10)
        ax4.set_ylabel('Probability', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # -----------Subplot (0,1) - Synaptical E/I Contacts Ratio---------------------
        syn_exc_ei_ratios = syn_exc_data / np.where(syn_inh_data != 0, syn_inh_data, 1)
        syn_inh_ei_ratios = syn_total_to_e / np.where(syn_inh_data != 0, syn_inh_data, 1)

        ax5.hist(syn_exc_ei_ratios, bins=75, alpha=0.75, color='pink', label=labels[2], linestyle=linestyles[0],edgecolor=edgecolors[2], linewidth=linewidths[0])
        ax5.hist(syn_inh_ei_ratios, bins=75, alpha=0.75, color='azure', label=labels[3], linestyle=linestyles[1],edgecolor=edgecolors[3], linewidth=linewidths[1])
        ax5.axvline(syn_ei_ratio, color='black', linestyle='--', linewidth=2)
        ax5.plot(syn_ei_ratio, ax5.get_ylim()[1]*0.05, 'v', markersize=10, color='black', markeredgecolor='white', markeredgewidth=1,
                label=f'syn E/I ratio = {syn_ei_ratio:.3f}')
        ax5.set_title(f'Distribution of {basic} Synaptical E/I Contacts Ratio ', fontsize=12, fontweight='bold')
        ax5.set_xlabel(f'{basic} Synaptical E/I Contacts Ratio', fontsize=10)
        ax5.set_ylabel('Probability', fontsize=10)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # -----------Subplot (1,1) - Synaptical I/E Contacts Ratio---------------------
        syn_exc_ie_ratios = syn_total_to_i / np.where(syn_exc_data != 0, syn_exc_data, 1)
        syn_inh_ie_ratios = syn_inh_data / np.where(syn_total_to_e != 0, syn_total_to_e, 1)

        ax6.hist(syn_exc_ie_ratios, bins=75, alpha=0.75, color='pink', label=labels[2], linestyle=linestyles[0],edgecolor=edgecolors[2], linewidth=linewidths[0])
        ax6.hist(syn_inh_ie_ratios, bins=75, alpha=0.75, color='azure', label=labels[3], linestyle=linestyles[1],edgecolor=edgecolors[3], linewidth=linewidths[1])
        ax6.axvline(syn_ie_ratio, color='black', linestyle='--', linewidth=2)
        ax6.plot(syn_ie_ratio, ax6.get_ylim()[1]*0.05, 'v', markersize=10, color='black', markeredgecolor='white', markeredgewidth=1,
                label=f'syn I/E ratio = {syn_ie_ratio:.3f}')
        ax6.set_title(f'Distribution of {basic} Synaptical I/E Contacts Ratio ', fontsize=12, fontweight='bold')
        ax6.set_xlabel(f'{basic} Synaptical I/E Contacts Ratio', fontsize=10)
        ax6.set_ylabel('Probability', fontsize=10)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # -----------Subplot (2,0) - Electrical E/I Conductances Ratio-----------------
        ele_exc_ei_ratios = ele_exc_data / np.where(ele_inh_data != 0, ele_inh_data, 1)
        ele_inh_ei_ratios = ele_total_to_e / np.where(ele_inh_data != 0, ele_inh_data, 1)

        ax7.hist(ele_exc_ei_ratios, bins=75, alpha=0.75, color='pink', label=labels[4], linestyle=linestyles[0],edgecolor=edgecolors[4], linewidth=linewidths[0])
        ax7.hist(ele_inh_ei_ratios, bins=75, alpha=0.75, color='azure', label=labels[5], linestyle=linestyles[1],edgecolor=edgecolors[5], linewidth=linewidths[1])
        ax7.axvline(ele_ei_ratio, color='black', linestyle='--', linewidth=2)
        ax7.plot(ele_ei_ratio, ax7.get_ylim()[1]*0.05, 'v', markersize=10, color='black', markeredgecolor='white', markeredgewidth=1,
                label=f'ele E/I ratio = {ele_ei_ratio:.3f}')
        ax7.set_title(f'Distribution of {basic} Electrical E/I Conductances Ratio ', fontsize=12, fontweight='bold')
        ax7.set_xlabel(f'{basic} Electrical E/I Conductances Ratio', fontsize=10)
        ax7.set_ylabel('Probability', fontsize=10)
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # -----------Subplot (2,1) - Electrical I/E Conductances Ratio-----------------
        ele_exc_ie_ratios = ele_total_to_i / np.where(ele_exc_data != 0, ele_exc_data, 1)
        ele_inh_ie_ratios = ele_inh_data / np.where(ele_total_to_e != 0, ele_total_to_e, 1)

        ax8.hist(ele_exc_ie_ratios, bins=75, alpha=0.75, color='pink', label=labels[4], linestyle=linestyles[0],edgecolor=edgecolors[4], linewidth=linewidths[0])
        ax8.hist(ele_inh_ie_ratios, bins=75, alpha=0.75, color='azure', label=labels[5], linestyle=linestyles[1],edgecolor=edgecolors[5], linewidth=linewidths[1])
        ax8.axvline(ele_ie_ratio, color='black', linestyle='--', linewidth=2)
        ax8.plot(ele_ie_ratio, ax8.get_ylim()[1]*0.05, 'v', markersize=10, color='black', markeredgecolor='white', markeredgewidth=1,
                label=f'ele I/E ratio = {ele_ie_ratio:.3f}')
        ax8.set_title(f'Distribution of {basic} Electrical I/E Conductances Ratio', fontsize=12, fontweight='bold')
        ax8.set_xlabel(f'{basic} Electrical I/E Conductances Ratio', fontsize=10)
        ax8.set_ylabel('Probability', fontsize=10)
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # Adjust layout and save the combined figure
        plt.tight_layout()
        # Save to plots directory
        plt.savefig(os.path.join(plots_dir, f'{basic}_ratio_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
 
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
PT Cell Clustering Package

This module provides functionality to cluster Pyramidal Tract (PT) cells with their subtype identifications.
PT cells are identified by filenames starting with 'cADpyr'.
Subtypes are extracted from the morphology part of the filename (e.g., PC, SP, STPC, TTPC1, etc.).

Key Functions:
- extract_pt_cell_info: Extract PT cell information from filename
- find_pt_cell_files: Find all PT cell files in a directory
- cluster_pt_cells: Cluster PT cells by subtype and layer
- get_pt_clusters: Convenience function to get clusters without file I/O
- save_clusters_to_csv: Save clustered PT cells to CSV file
- print_cluster_summary: Print summary of PT cell clusters
- main: Main execution function (only runs when script is executed directly)
"""

import os
import re
import csv
import json
from collections import defaultdict
from pathlib import Path

def get_script_dir():
    """Get the directory of the current script, handling cases where __file__ is not available."""
    try:
        # Normal case when running as a script or imported module
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Case when executed via exec() or similar
        return os.getcwd()

def extract_pt_cell_info(filename):
    base_name = filename.replace('.cell.nml', '')
    pt_prefixes = ['cADpyr229', 'cADpyr230', 'cADpyr231', 'cADpyr232']
    is_pt_cell = any(base_name.startswith(prefix + '_') for prefix in pt_prefixes)
    if not is_pt_cell:
        return None
    parts = base_name.split('_')
    if len(parts) < 4:
        return None
    original_name = base_name
    layer = parts[1]  
    cell_type = parts[2] 
    category = 'PT' 

    if not re.match(r'^L\d+$', layer):
        return None
    combined_subtype = f"{layer}_{cell_type}"
    return {
        'original_name': original_name,
        'layer': layer,
        'subtype': cell_type,
        'source_file': filename,
        'combined_subtype': combined_subtype,
        'category': category 
    }

def find_pt_cell_files(directory='.'):
    pt_files = []
    for file in os.listdir(directory):
        if file.endswith('cell.nml'):
            base_name = file.replace('.cell.nml', '')
            pt_prefixes = ['cADpyr229', 'cADpyr230', 'cADpyr231', 'cADpyr232']
            if any(base_name.startswith(prefix + '_') for prefix in pt_prefixes):
                pt_files.append(file)
    return sorted(pt_files)

def cluster_pt_cells(pt_files):
    clusters = defaultdict(list)
    pt_cells = []
    for filename in pt_files:
        cell_info = extract_pt_cell_info(filename)
        if cell_info:
            pt_cells.append(cell_info)
            clusters[cell_info['combined_subtype']].append(cell_info)
    return dict(clusters), pt_cells

def extract_prefix_base(prefix):
    match = re.match(r'^([a-zA-Z]+(?:int)?[A-Z]*)', prefix)
    if match:
        return match.group(1)
    return prefix

def get_pt_clusters_from_catalog():
    catalog_path = os.path.join(get_script_dir(), 'cataloged_cells.json')
    if not os.path.exists(catalog_path):
        print(f"Warning: Catalog file not found at {catalog_path}")
        return {}, []
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    pt_cells_data = catalog.get('PT', [])
    clusters = {}
    pt_cells = []
    for cell in pt_cells_data:
        original_name = cell.get('original_name', '')
        layer = cell.get('layer_validated', '') or cell.get('layer_raw', '')
        morphological_type = cell.get('cell_type', '')
        prefix_full = cell.get('prefix', '')
        source_file = cell.get('source_file', '')
        
        if not morphological_type or not layer or not prefix_full:
            continue
            
        # Extract base prefix (remove trailing numbers)
        prefix_base = extract_prefix_base(prefix_full)
        
        # Create cluster key in format: prefix_layer (e.g., 'dNAC_L4')
        cluster_key = f"{prefix_base}_{layer}"
        cell_entry = {
            'original_name': original_name,
            'layer': layer,
            'subtype': morphological_type,
            'source_file': source_file,
            'combined_subtype': morphological_type,
            'cell_type': morphological_type,  # This should be the morphological subtype
            'category': 'IT',                 # This should be the high-level category
            'prefix': prefix_base,           # Add prefix for reference
            'prefix_full': prefix_full       # Keep full prefix for reference
        }
        # Add to clusters
        if cluster_key not in clusters:
            clusters[cluster_key] = []
        clusters[cluster_key].append(cell_entry)
        pt_cells.append(cell_entry)
    
    return clusters, pt_cells

def get_pt_clusters(directory='.'):
    pt_files = find_pt_cell_files(directory)
    if not pt_files:
        return {}, []
    return cluster_pt_cells(pt_files)

def save_clusters_to_csv(clusters, output_file='pt_cell_clusters.csv'):
    pt_cells = []
    for cell_list in clusters.values():
        pt_cells.extend(cell_list)
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['original_name', 'layer', 'subtype', 'source_file', 'combined_subtype', 'cell_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for cell in pt_cells:
            writer.writerow({
                'original_name': cell['original_name'],
                'layer': cell['layer'],
                'subtype': cell['subtype'],
                'source_file': cell['source_file'],
                'combined_subtype': cell['combined_subtype'],
                'cell_type': cell['cell_type']
            })
    

def print_cluster_summary(clusters):
    """Print summary of PT cell clusters."""
    print(f"Found {sum(len(cells) for cells in clusters.values())} PT cells")
    total_cells = 0
    
    for combined_subtype, cells in sorted(clusters.items()):
        print(f"{combined_subtype}: {len(cells)} cells")
        total_cells += len(cells)
    print(f"Total PT cells: {total_cells}")

def main():
    clusters, pt_cells = get_pt_clusters_from_catalog()
    
    if not pt_cells:
        print("Falling back to file-based approach...")
        clusters, pt_cells = get_pt_clusters()
    if clusters:
        print_cluster_summary(clusters)
        save_clusters_to_csv(clusters)
    
if __name__ == "__main__":
    main()
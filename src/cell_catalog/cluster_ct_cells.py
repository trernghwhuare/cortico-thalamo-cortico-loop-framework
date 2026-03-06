#!/usr/bin/env python3
"""
Cluster CT (Corticothalamic) cells from ct_clusters.csv.

This script reads CT cells from the ct_clusters.csv file and organizes them into clusters
based on their prefix (e.g., bSTUT213, cSTUT189, dSTUT214)
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

def extract_ct_cell_info(filename):
    base_name = filename.replace('.cell.nml', '')
    ct_prefixes = ['bSTUT213', 'cSTUT189', 'dSTUT214']
    is_ct_cell = any(base_name.startswith(prefix + '_') for prefix in ct_prefixes)
    if not is_ct_cell:
        return None
    parts = base_name.split('_')
    if len(parts) < 4:
        return None
    
    original_name = base_name
    layer = parts[1]  # e.g., L23, L4, L5, L6
    cell_type = parts[2]  # e.g., BP, ChC, LBC, MC, NBC, SBC, BTC (morphological subtype)
    category = 'CT'
    
    # Validate layer format (should start with L followed by numbers)
    if not re.match(r'^L\d+$', layer):
        return None
    
    combined_subtype = f"{layer}_{cell_type}"
    
    return {
        'original_name': original_name,
        'layer': layer,
        'cell_type': cell_type,  # This contains morphological subtype like MC, ChC, BP
        'source_file': filename,
        'combined_subtype': combined_subtype,
        'category': category  # This contains 'IT'
    }


def find_ct_cell_files(directory='.'):
    """Find all CT cell files ending with 'cell.nml'."""
    ct_files = []
    try:
        for file in os.listdir(directory):
            if file.endswith('cell.nml'):
                # Check if it starts with any known CT cell prefix
                base_name = file.replace('.cell.nml', '')
                ct_prefixes = ['bSTUT213', 'cSTUT189', 'dSTUT214']
                if any(base_name.startswith(prefix + '_') for prefix in ct_prefixes):
                    ct_files.append(file)
    except FileNotFoundError:
        print(f"Directory {directory} not found")
    return sorted(ct_files)

def cluster_ct_cells(ct_files):
    """Cluster CT cells by subtype and layer."""
    clusters = defaultdict(list)
    ct_cells = []
    for filename in ct_files:
        cell_info = extract_ct_cell_info(filename)
        if cell_info:
            ct_cells.append(cell_info)
            clusters[cell_info['combined_subtype']].append(cell_info)
    return dict(clusters), ct_cells
def extract_prefix_base(prefix):
    match = re.match(r'^([a-zA-Z]+(?:int)?[A-Z]*)', prefix)
    if match:
        return match.group(1)
    return prefix

def get_ct_clusters_from_catalog():
    catalog_path = os.path.join(get_script_dir(), 'cataloged_cells.json')
    if not os.path.exists(catalog_path):
        print(f"Warning: Catalog file not found at {catalog_path}")
        return {}, []
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    ct_cells_data = catalog.get('CT', [])
    clusters = {}
    ct_cells = []
    for cell in ct_cells_data:
        # Extract required fields from catalog
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
        
        # Create cell entry with proper field assignments
        # According to project specification:
        # - cell_type should be the morphological subtype (MC, ChC, BP, etc.)
        # - category should be the high-level classification (IT)
        cell_entry = {
            'original_name': original_name,
            'layer': layer,
            'subtype': morphological_type,
            'source_file': source_file,
            'combined_subtype': morphological_type,
            'cell_type': morphological_type,  # This should be the morphological subtype
            'category': 'CT',                 # This should be the high-level category
            'prefix': prefix_base,           # Add prefix for reference
            'prefix_full': prefix_full       # Keep full prefix for reference
        }
        
        # Add to clusters
        if cluster_key not in clusters:
            clusters[cluster_key] = []
        clusters[cluster_key].append(cell_entry)
        ct_cells.append(cell_entry)
    
    return clusters, ct_cells

def get_ct_clusters(directory='.'):
    """
    Get CT cell clusters from a directory.
    This is kept for backward compatibility but catalog method is preferred.
    """
    ct_files = find_ct_cell_files(directory)
    if not ct_files:
        return {}, []
    
    return cluster_ct_cells(ct_files)

def save_clusters_to_csv(clusters, output_file='ct_cell_clusters.csv'):
    """Save clustered CT cells to CSV file."""
    # Flatten all cells from all clusters
    ct_cells = []
    for cell_list in clusters.values():
        ct_cells.extend(cell_list)
    
    if not ct_cells:
        print("No CT cells to save.")
        return
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['original_name', 'layer', 'subtype', 'source_file', 'combined_subtype', 'cell_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for cell in ct_cells:
            writer.writerow({
                'original_name': cell['original_name'],
                'layer': cell['layer'],
                'subtype': cell['subtype'],
                'source_file': cell['source_file'],
                'combined_subtype': cell['combined_subtype'],
                'cell_type': cell['cell_type']
            })


def print_cluster_summary(clusters):
    """Print a summary of CT cell clusters."""
    print(f"Found {len(clusters)} CT cell subtypes:")
    total_cells = 0
    for combined_subtype, cells in sorted(clusters.items()):
        print(f"  {combined_subtype}: {len(cells)} cells")
        total_cells += len(cells)
    print(f"Total CT cells: {total_cells}")

def main():
    
    clusters, ct_cells = get_ct_clusters_from_catalog()
    
    if not ct_cells:
        # Fallback to file-based approach if catalog fails
        print("Falling back to file-based approach...")
        clusters, ct_cells = get_ct_clusters()
    
    if clusters:
        print_cluster_summary(clusters)
        save_clusters_to_csv(clusters)
        print("Clusters saved to 'ct_cell_clusters.csv'")
    else:
        print("No CT cells found.")

if __name__ == "__main__":
    main()
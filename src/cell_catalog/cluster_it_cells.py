#!/usr/bin/env python3
"""
IT Cell Clustering Package
Key Functions:
- extract_it_cell_info: Extract IT cell information from filename
- find_it_cell_files: Find all IT cell files in a directory
- cluster_it_cells: Cluster IT cells by subtype and layer
- get_it_clusters: Convenience function to get clusters without file I/O
- save_clusters_to_csv: Save clustered IT cells to CSV file
- print_cluster_summary: Print summary of IT cell clusters
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


def extract_it_cell_info(filename):
    base_name = filename.replace('.cell.nml', '')
    it_prefixes = ['bAC217','bIR215', 'bNAC219','dNAC222', 'cNAC187', 'cACint209','cIR216']
    is_it_cell = any(base_name.startswith(prefix + '_') for prefix in it_prefixes)
    if not is_it_cell:
        return None
    
    # Split by underscores
    parts = base_name.split('_')
    if len(parts) < 4:
        return None
    
    # Extract components
    original_name = base_name
    layer = parts[1]  # e.g., L23, L4, L5, L6
    cell_type = parts[2]  # e.g., BP, ChC, LBC, MC, NBC, SBC, BTC (morphological subtype)
    category = 'IT'
    
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


def find_it_cell_files(directory='.'):
    """Find all IT cell files ending with 'cell.nml'."""
    it_files = []
    try:
        for file in os.listdir(directory):
            if file.endswith('cell.nml'):
                # Check if it starts with any known IT cell prefix
                base_name = file.replace('.cell.nml', '')
                it_prefixes = ['dNAC222', 'cNAC187', 'cACint209', 'bIR215', 'cIR216']
                if any(base_name.startswith(prefix + '_') for prefix in it_prefixes):
                    it_files.append(file)
    except FileNotFoundError:
        print(f"Directory {directory} not found")
    return sorted(it_files)


def cluster_it_cells(it_files):
    """Cluster IT cells by subtype and layer."""
    clusters = defaultdict(list)
    all_cells = []
    
    for filename in it_files:
        cell_info = extract_it_cell_info(filename)
        if cell_info:
            all_cells.append(cell_info)
            clusters[cell_info['combined_subtype']].append(cell_info)
    
    return dict(clusters), all_cells


def extract_prefix_base(prefix):
    match = re.match(r'^([a-zA-Z]+(?:int)?[A-Z]*)', prefix)
    if match:
        return match.group(1)
    return prefix


def get_it_clusters_from_catalog():
    catalog_path = os.path.join(get_script_dir(), 'cataloged_cells.json')
    if not os.path.exists(catalog_path):
        print(f"Warning: Catalog file not found at {catalog_path}")
        return {}, []
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)
    it_cells_data = catalog.get('IT', [])
    clusters = {}
    all_cells = []
    
    for cell in it_cells_data:
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
            'category': 'IT',                 # This should be the high-level category
            'prefix': prefix_base,           # Add prefix for reference
            'prefix_full': prefix_full       # Keep full prefix for reference
        }
        
        # Add to clusters
        if cluster_key not in clusters:
            clusters[cluster_key] = []
        clusters[cluster_key].append(cell_entry)
        all_cells.append(cell_entry)
    
    return clusters, all_cells


def get_it_clusters(directory='.'):
    """
    Get IT cell clusters from a directory.
    This is kept for backward compatibility but catalog method is preferred.
    """
    it_files = find_it_cell_files(directory)
    if not it_files:
        return {}, []
    
    return cluster_it_cells(it_files)


def save_clusters_to_csv(clusters, output_file='it_cell_clusters.csv'):
    """Save clustered IT cells to CSV file."""
    # Flatten all cells from all clusters
    all_cells = []
    for cell_list in clusters.values():
        all_cells.extend(cell_list)
    
    if not all_cells:
        print("No IT cells to save.")
        return
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['original_name', 'layer', 'subtype', 'source_file', 'combined_subtype', 'cell_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for cell in all_cells:
            writer.writerow({
                'original_name': cell['original_name'],
                'layer': cell['layer'],
                'subtype': cell['subtype'],
                'source_file': cell['source_file'],
                'combined_subtype': cell['combined_subtype'],
                'cell_type': cell['cell_type']
            })


def print_cluster_summary(clusters):
    """Print a summary of IT cell clusters."""
    print(f"Found {len(clusters)} IT cell subtypes:")
    total_cells = 0
    for combined_subtype, cells in sorted(clusters.items()):
        print(f"  {combined_subtype}: {len(cells)} cells")
        total_cells += len(cells)
    print(f"Total IT cells: {total_cells}")


def main():
    """Main function to demonstrate IT cell clustering."""
    print("Clustering IT cells...")
    
    # Try catalog-based approach first (this should work with cataloged_cells.json)
    clusters, all_cells = get_it_clusters_from_catalog()
    
    if not all_cells:
        # Fallback to file-based approach if catalog fails
        print("Falling back to file-based approach...")
        clusters, all_cells = get_it_clusters()
    
    if clusters:
        print_cluster_summary(clusters)
        save_clusters_to_csv(clusters)
        print("Clusters saved to 'it_cell_clusters.csv'")
    else:
        print("No IT cells found.")


if __name__ == "__main__":
    main()
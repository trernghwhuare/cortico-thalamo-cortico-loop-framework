#!/usr/bin/env python3
"""
Script to extract cell data (ID, axonal counts, dendrites counts, member segment count) 
from .cell.nml files in a specified directory.

Uses XML parsing to:
- Extract cell ID from the cell element
- Count <include> elements within segment groups:
  * Primary method: Count includes in segmentGroup id="axon_group" and "dendrite_group"
  * Fallback method: Count segments by name patterns (Seg*_axon_*, Seg*_dend_*)
- Count member segments (<member segment=)

Handles both thalamic cells (with axon_group/dendrite_group) and cortical cells 
(with axon_/dend_ segment groups).
"""

import os
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_cell_nml(file_path):
    """
    Parse a .cell.nml file and extract cell information using XML parsing.
    
    Args:
        file_path (str): Path to the .cell.nml file
        
    Returns:
        dict: Dictionary containing cell_id, axonal_count, dendrites_count, member_segment_count
    """
    try:
        # Define NeuroML namespace
        ns = {'nml': 'http://www.neuroml.org/schema/neuroml2'}
        
        # Parse XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find the cell element
        cell = root.find('.//nml:cell', ns)
        if cell is None:
            print(f"Warning: No cell element found in {file_path}")
            return None
            
        cell_id = cell.get('id')
        if not cell_id:
            cell_id = cell.get('name', 'unknown')
        
        # Find morphology section
        morphology = cell.find('.//nml:morphology', ns)
        if morphology is None:
            print(f"Warning: No morphology element found in {file_path}")
            return None
        
        # Initialize counts
        axonal_count = 0
        dendrites_count = 0
        
        # Look for main segment groups that contain axon/dendrite segments
        # For cortical cells: axonal, basal, apical
        # For thalamus cells: axon_group, dendrite_group (if they exist)
        
        # Check for axonal group first (cortical style)
        axonal_group = morphology.find('.//nml:segmentGroup[@id="axonal"]', ns)
        if axonal_group is not None:
            # Count include elements in axonal group
            axonal_count = len(axonal_group.findall('.//nml:include', ns))
            print(f"DEBUG: Found axonal group with {axonal_count} includes in {os.path.basename(file_path)}")
        
        # If no axonal group found, check for axon_group (thalamus style)
        if axonal_count == 0:
            axon_group = morphology.find('.//nml:segmentGroup[@id="axon_group"]', ns)
            if axon_group is not None:
                axonal_count = len(axon_group.findall('.//nml:include', ns))
                print(f"DEBUG: Found axon_group with {axonal_count} includes in {os.path.basename(file_path)}")
        
        # Check for basal group (dendritic)
        basal_group = morphology.find('.//nml:segmentGroup[@id="basal"]', ns)
        if basal_group is not None:
            basal_count = len(basal_group.findall('.//nml:include', ns))
            dendrites_count += basal_count
            print(f"DEBUG: Found basal group with {basal_count} includes in {os.path.basename(file_path)}")
        
        # Check for apical group (dendritic)  
        apical_group = morphology.find('.//nml:segmentGroup[@id="apical"]', ns)
        if apical_group is not None:
            apical_count = len(apical_group.findall('.//nml:include', ns))
            dendrites_count += apical_count
            print(f"DEBUG: Found apical group with {apical_count} includes in {os.path.basename(file_path)}")
        
        # If no basal/apical groups found, check for dendrite_group (thalamus style)
        if dendrites_count == 0:
            dendrite_group = morphology.find('.//nml:segmentGroup[@id="dendrite_group"]', ns)
            if dendrite_group is not None:
                dendrites_count = len(dendrite_group.findall('.//nml:include', ns))
                print(f"DEBUG: Found dendrite_group with {dendrites_count} includes in {os.path.basename(file_path)}")
        
        # If still no counts found, fall back to original method as backup
        if axonal_count == 0 and dendrites_count == 0:
            # Original logic: look for individual axon_ and dend_ groups
            axon_groups = morphology.findall('.//nml:segmentGroup[starts-with(@id, "axon_")]', ns)
            dend_groups = morphology.findall('.//nml:segmentGroup[starts-with(@id, "dend_")]', ns)
            
            # Count members in each group
            for group in axon_groups:
                axonal_count += len(group.findall('.//nml:member', ns))
            for group in dend_groups:
                dendrites_count += len(group.findall('.//nml:member', ns))
            
            if axonal_count > 0 or dendrites_count > 0:
                print(f"DEBUG: Used fallback method - axonal: {axonal_count}, dendrites: {dendrites_count} in {os.path.basename(file_path)}")
        
        # Count total member segments (all <member segment= elements)
        member_segment_count = len(morphology.findall('.//nml:member', ns))
        
        return {
            'cell_id': cell_id,
            'axonal_count': axonal_count,
            'dendrites_count': dendrites_count,
            'member_segment_count': member_segment_count
        }
        
    except ET.ParseError as e:
        print(f"XML Parse error processing {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_cell_data.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    # Find all .cell.nml files recursively
    cell_files = list(Path(directory).rglob("*.cell.nml"))
    
    print(f"Found {len(cell_files)} .cell.nml files:")
    for f in cell_files:
        print(f"  - {f.name}")
    
    if not cell_files:
        print(f"No .cell.nml files found in {directory}")
        sys.exit(0)
    
    extracted_data = []
    
    for cell_file in cell_files:
        print(f"\nProcessing {cell_file.name}...")
        cell_data = parse_cell_nml(str(cell_file))
        if cell_data:
            # Add source file information
            cell_data['source_file'] = cell_file.name
            extracted_data.append(cell_data)
    
    # Write to CSV file
    output_file = "extracted_cell_data.csv"
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['cell_id', 'source_file', 'axonal_count', 'dendrites_count', 'member_segment_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in extracted_data:
            writer.writerow(row)
    
    print(f"\nExtracted data from {len(extracted_data)} cells")
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()
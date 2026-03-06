#!/usr/bin/env python3
"""
Script to distinguish and catalog PT, IT, CT, and TC cells based on their naming patterns.

This script analyzes cell names from files and categorizes them into:
- PT cells (Pyramidal Tract)
- IT cells (Intratelencephalic) 
- CT cells (Corticothalamic)
- TC cells (Thalamocortical)

Based on patterns observed in the hardcoded listings from 33_max_M1+.py
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CellCategorizer:
    """Categorizes cells based on their naming patterns."""
    
    def __init__(self):
        # Define patterns for each cell type based on the hardcoded examples
        
        # IT cell patterns - various prefixes with specific cell types
        self.it_prefixes = {
            'bAC', 'bIR', 'bNAC', 'cACint', 'cIR', 'cNAC', 'dNAC'
        }
        
        self.it_cell_types = {
            'BP', 'BTC', 'DBC', 'MC', 'NBC', 'LBC', 'NGC', 'SBC', 
            'HAC', 'DAC', 'SLAC', 'NGCDA', 'ChC', 'DLAC', 'NGCSA'
        }
        
        # PT cell patterns - cADpyr prefix with specific cell types
        self.pt_prefix = 'cADpyr'
        self.pt_cell_types = {
            'PC', 'SP', 'SS', 'BPC', 'IPC', 'TPC', 'UTPC', 'STPC', 'TTPC1', 'TTPC2'
        }
        
        # CT cell patterns - STUT prefixes with specific cell types
        self.ct_prefixes = {'bSTUT', 'cSTUT', 'dSTUT'}
        self.ct_cell_types = {
            'BP', 'BTC', 'DBC', 'MC', 'NBC', 'LBC', 'NGC', 'NGCDA'
        }
        
        # TC cell patterns - specific thalamic cell names with subtypes
        self.tc_patterns = {
            'TCRil': 'intralaminar',
            'nRTil': 'intralaminar', 
            'TCR': 'matrix',
            'TCRm': 'matrix',
            'nRTm': 'matrix',
            'nRT': 'core',
            'TCRc': 'core',
            'nRTc': 'core'
        }
        
        # Valid cortical layers
        self.valid_layers = {'L1', 'L23', 'L4', 'L5', 'L6'}
    
    def identify_layer(self, layer_str: str) -> Optional[str]:
        """
        Identify and validate cortical layer from string.
        
        Args:
            layer_str: Layer string (e.g., "L23", "L4", etc.)
            
        Returns:
            Validated layer string or None if invalid
        """
        if not layer_str:
            return None
            
        # Normalize layer string (remove any extra characters, ensure uppercase L)
        normalized = layer_str.strip()
        if normalized.startswith('l') or normalized.startswith('L'):
            # Ensure it's uppercase L followed by digits
            normalized = 'L' + normalized[1:].upper()
        
        # Check if it matches valid layer patterns
        if normalized in self.valid_layers:
            return normalized
        
        # Handle potential variations
        if normalized == 'L2/3':
            return 'L23'
        elif normalized == 'L2_3':
            return 'L23'
        elif normalized == 'Layer23':
            return 'L23'
        elif normalized.startswith('L2') or normalized.startswith('L3'):
            # If it's L2 or L3 separately, map to L23 as per common convention
            return 'L23'
        
        return None
    
    def identify_tc_subtype(self, tc_name: str) -> Optional[str]:
        """
        Identify thalamic cell subtype for TC cells.
        
        Args:
            tc_name: TC cell name (e.g., "TCRil", "nRT", etc.)
            
        Returns:
            Thalamic subtype ('intralaminar', 'matrix', 'core') or None
        """
        return self.tc_patterns.get(tc_name, None)
    
    def categorize_cell(self, cell_name: str) -> Optional[str]:
        """
        Categorize a single cell name into PT, IT, CT, or TC.
        
        Args:
            cell_name: The full cell name (e.g., "bAC217_L4_BP_8c9cdc6683_0_0")
            
        Returns:
            Category string ('PT', 'IT', 'CT', 'TC') or None if not matched
        """
        # Split by underscore to get the main identifier
        parts = cell_name.split('_')
        if not parts:
            return None
            
        main_part = parts[0]
        
        # Check for TC cells first (simplest pattern)
        if main_part in self.tc_patterns:
            return 'TC'
        
        # Check for PT cells
        if main_part.startswith(self.pt_prefix):
            # Extract the cell type part (after layer info)
            # Format: cADpyr229_L23_PC_8ef1aa6602_0_0
            if len(parts) >= 3:
                cell_type = parts[2]
                if cell_type in self.pt_cell_types:
                    return 'PT'
        
        # Check for CT cells
        for ct_prefix in self.ct_prefixes:
            if main_part.startswith(ct_prefix):
                if len(parts) >= 3:
                    cell_type = parts[2]
                    if cell_type in self.ct_cell_types:
                        return 'CT'
        
        # Check for IT cells
        for it_prefix in self.it_prefixes:
            if main_part.startswith(it_prefix):
                if len(parts) >= 3:
                    cell_type = parts[2]
                    if cell_type in self.it_cell_types:
                        return 'IT'
        
        return None
    
    def parse_cell_info(self, cell_name: str) -> Dict:
        """Extract detailed information from cell name including proper layer/thalamic identification."""
        parts = cell_name.split('_')
        info = {
            'original_name': cell_name,
            'category': self.categorize_cell(cell_name),
            'prefix': None,
            'layer_raw': None,
            'layer_validated': None,
            'thalamic_subtype': None,
            'cell_type': None,
            'id_suffix': None,
            'layer_category': None,  # Additional layer grouping info for cortical cells
            'region_type': None      # 'cortical' or 'thalamic'
        }
        
        if not parts:
            return info
            
        main_part = parts[0]
        info['prefix'] = main_part
        
        # Handle TC cells (thalamic) differently
        if info['category'] == 'TC':
            info['region_type'] = 'thalamic'
            info['thalamic_subtype'] = self.identify_tc_subtype(main_part)
            # TC cells don't have cortical layers
            info['layer_validated'] = None
            info['layer_category'] = None
        else:
            # Handle cortical cells (PT, IT, CT)
            info['region_type'] = 'cortical'
            # Extract layer and cell type if available
            if len(parts) >= 3:
                info['layer_raw'] = parts[1]
                info['layer_validated'] = self.identify_layer(parts[1])
                info['cell_type'] = parts[2]
                
                # Add layer category for easier filtering
                if info['layer_validated']:
                    if info['layer_validated'] == 'L1':
                        info['layer_category'] = 'supragranular_upper'
                    elif info['layer_validated'] in ['L23']:
                        info['layer_category'] = 'supragranular'
                    elif info['layer_validated'] == 'L4':
                        info['layer_category'] = 'granular'
                    elif info['layer_validated'] in ['L5', 'L6']:
                        info['layer_category'] = 'infragranular'
        
        # Extract ID suffix if available (for cortical cells)
        if len(parts) >= 4 and info['category'] != 'TC':
            info['id_suffix'] = parts[3]
            
        return info
    
    def categorize_by_layer_and_type(self, cell_name: str) -> Dict:
        """
        Provide comprehensive categorization including both cell type and layer/thalamic subtype.
        
        Returns:
            Dictionary with category, layer/thalamic info, and combined classification
        """
        info = self.parse_cell_info(cell_name)
        combined_category = None
        
        if info['category'] == 'TC':
            if info['thalamic_subtype']:
                combined_category = f"{info['category']}_{info['thalamic_subtype']}"
            else:
                combined_category = info['category']
        elif info['category'] and info['layer_validated']:
            combined_category = f"{info['category']}_{info['layer_validated']}"
        elif info['category']:
            combined_category = info['category']
            
        info['combined_category'] = combined_category
        return info


def find_cell_files(directory: str = '.', extensions: Optional[List[str]] = None) -> List[str]:
    """Find all potential cell definition files in the directory."""
    # Only process files ending with 'cell.nml'
    cell_files = []
    cell_files.extend(Path(directory).glob('*cell.nml'))
    # cell_files.extend(Path(directory).glob('**/*cell.nml'))
    
    return [str(f) for f in cell_files]


def extract_cell_names_from_file(filepath: str) -> List[str]:
    """Extract cell names from a file based on common patterns."""
    cell_names = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Skip binary files
        return cell_names
    except Exception:
        return cell_names
    
    # Look for patterns that match cell names
    # Pattern: word_characters_layer_celltype_hash_numbers
    cell_pattern = r'\b\w+_\w+_[A-Z][A-Za-z0-9]*_[a-f0-9]{10}_\d+_\d+\b'
    matches = re.findall(cell_pattern, content)
    cell_names.extend(matches)
    
    # Also look for TC patterns (simpler format)
    tc_pattern = r'\b(TCRil|nRTil|TCR|TCRm|nRTm|nRT|TCRc|nRTc)\b'
    tc_matches = re.findall(tc_pattern, content)
    cell_names.extend(tc_matches)
    
    return list(set(cell_names))  # Remove duplicates


def main():
    """Main function to catalog cells from local files."""
    categorizer = CellCategorizer()
    
    print("Searching for cell definition files...")
    cell_files = find_cell_files()
    print(f"Found {len(cell_files)} potential files to scan")
    
    all_cells = {}
    categorized_counts = {'PT': 0, 'IT': 0, 'CT': 0, 'TC': 0, 'Unknown': 0}
    layer_counts = {'L1': 0, 'L23': 0, 'L4': 0, 'L5': 0, 'L6': 0, 'Unknown': 0}
    thalamic_subtype_counts = {'intralaminar': 0, 'matrix': 0, 'core': 0, 'Unknown': 0}
    combined_counts = {}
    
    # Collect all individual cells for CSV output
    individual_cells = []
    
    # Process each file
    for filepath in cell_files:
        cell_names = extract_cell_names_from_file(filepath)
        if not cell_names:
            continue
            
        # print(f"Processing {filepath}: found {len(cell_names)} potential cell names")
        
        for cell_name in cell_names:
            category = categorizer.categorize_cell(cell_name)
            cell_info = categorizer.categorize_by_layer_and_type(cell_name)
            
            # Store individual cell data for CSV
            cell_record = {
                'original_name': cell_name,
                'category': category if category else 'Unknown',
                'region_type': cell_info['region_type'],
                'layer_validated': cell_info['layer_validated'] if cell_info['layer_validated'] else '',
                'layer_category': cell_info['layer_category'] if cell_info['layer_category'] else '',
                'thalamic_subtype': cell_info['thalamic_subtype'] if cell_info['thalamic_subtype'] else '',
                'combined_category': cell_info['combined_category'] if cell_info['combined_category'] else '',
                'source_file': os.path.basename(filepath)
            }
            individual_cells.append(cell_record)
            
            if category:
                categorized_counts[category] += 1
                if category not in all_cells:
                    all_cells[category] = []
                all_cells[category].append(cell_info)
                
                # Count by layer (cortical cells only)
                if cell_info['region_type'] == 'cortical':
                    if cell_info['layer_validated']:
                        layer_counts[cell_info['layer_validated']] += 1
                    else:
                        layer_counts['Unknown'] += 1
                # Count by thalamic subtype (TC cells only)
                elif cell_info['region_type'] == 'thalamic':
                    if cell_info['thalamic_subtype']:
                        thalamic_subtype_counts[cell_info['thalamic_subtype']] += 1
                    else:
                        thalamic_subtype_counts['Unknown'] += 1
                
                # Count by combined category
                combined_cat = cell_info['combined_category']
                if combined_cat:
                    if combined_cat not in combined_counts:
                        combined_counts[combined_cat] = 0
                    combined_counts[combined_cat] += 1
            else:
                categorized_counts['Unknown'] += 1
                if cell_info['region_type'] == 'cortical':
                    if cell_info['layer_validated']:
                        layer_counts[cell_info['layer_validated']] += 1
                    else:
                        layer_counts['Unknown'] += 1
                elif cell_info['region_type'] == 'thalamic':
                    if cell_info['thalamic_subtype']:
                        thalamic_subtype_counts[cell_info['thalamic_subtype']] += 1
                    else:
                        thalamic_subtype_counts['Unknown'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("CELL CATEGORIZATION SUMMARY")
    print("="*60)
    for category, count in categorized_counts.items():
        if count > 0:
            print(f"{category}: {count} ")
    
    print("\n" + "="*60)
    print("CORTICAL LAYER DISTRIBUTION")
    print("="*60)
    cortical_layer_found = False
    for layer, count in layer_counts.items():
        if count > 0:
            print(f"{layer}: {count} ")
            cortical_layer_found = True
    if not cortical_layer_found:
        print("No cortical cells found")
    
    print("\n" + "="*60)
    print("THALAMIC SUBTYPE DISTRIBUTION")
    print("="*60)
    thalamic_subtype_found = False
    for subtype, count in thalamic_subtype_counts.items():
        if count > 0:
            print(f"{subtype}: {count} ")
            thalamic_subtype_found = True
    if not thalamic_subtype_found:
        print("No thalamic cells found")
    
    print("\n" + "="*60)
    print("COMBINED CATEGORY DISTRIBUTION")
    print("="*60)
    for combined_cat, count in sorted(combined_counts.items()):
        print(f"{combined_cat}: {count} ")
    
    # Save detailed results
    output_file = 'cataloged_cells.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_cells, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to {output_file}")
    
    # Save individual cells to CSV
    csv_file = 'cell_catalog.csv'
    import csv
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['original_name', 'category', 'region_type', 'layer_validated', 'layer_category', 
                     'thalamic_subtype', 'combined_category', 'source_file']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cell in individual_cells:
            writer.writerow(cell)
    
    print(f"Individual cell catalog saved to {csv_file}")
    
    # Also create a simple CSV-like summary
    summary_file = 'cell_catalog_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Category\tCount\n")
        for category, count in categorized_counts.items():
            if count > 0:
                f.write(f"{category}\t{count}\n")
        f.write("\nCortical_Layer\tCount\n")
        for layer, count in layer_counts.items():
            if count > 0:
                f.write(f"{layer}\t{count}\n")
        f.write("\nThalamic_Subtype\tCount\n")
        for subtype, count in thalamic_subtype_counts.items():
            if count > 0:
                f.write(f"{subtype}\t{count}\n")
        f.write("\nCombined_Category\tCount\n")
        for combined_cat, count in sorted(combined_counts.items()):
            f.write(f"{combined_cat}\t{count}\n")
    
    print(f"Summary saved to {summary_file}")
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Catalog cells by type from local files")
    parser.add_argument('--test', action='store_true', help='Run tests with hardcoded examples')
    parser.add_argument('--directory', default='.', help='Directory to search for cell files')
    
    args = parser.parse_args()

    main()
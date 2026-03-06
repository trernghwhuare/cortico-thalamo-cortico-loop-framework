import pandas as pd
import os
import warnings

# Global cache to store the loaded data for repeated calls
_SEGMENT_COUNT_CACHE = None
_CSV_FILE_MISSING_WARNING_SHOWN = False

def _load_segment_data():
    """Internal function to load segment count data from CSV file."""
    global _SEGMENT_COUNT_CACHE, _CSV_FILE_MISSING_WARNING_SHOWN
    if _SEGMENT_COUNT_CACHE is not None:
        return _SEGMENT_COUNT_CACHE
    
    csv_path = os.path.join(os.path.dirname(__file__), 'extracted_cell_data.csv')
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        if not _CSV_FILE_MISSING_WARNING_SHOWN:
            warning_msg = (
                f"\nWARNING: CSV file '{csv_path}' not found!\n"
                "The extract_member_segment_count module requires 'extracted_cell_data.csv' to function properly.\n"
                "To generate this file:\n"
                "1. Ensure you have .cell.nml files in your workspace\n"
                "2. Run: python 01_extract_cell_data.py <directory_containing_cell_nml_files>\n"
                "3. This will create 'extracted_cell_data.csv' with actual segment counts\n\n"
                "Until then, the function will return fallback values (999 for cortical cells, 1000 for others).\n"
            )
            warnings.warn(warning_msg, UserWarning)
            _CSV_FILE_MISSING_WARNING_SHOWN = True
        
        # Return empty dict to trigger fallback behavior
        _SEGMENT_COUNT_CACHE = {}
        return _SEGMENT_COUNT_CACHE
    
    try:
        df = pd.read_csv(csv_path)
        # Create a dictionary mapping cell_id to member_segment_count
        _SEGMENT_COUNT_CACHE = dict(zip(df['cell_id'], df['member_segment_count']))
        return _SEGMENT_COUNT_CACHE
    except Exception as e:
        raise RuntimeError(f"Error loading segment data from {csv_path}: {str(e)}")

def get_actual_segment_count(cell_id):
    """
    Get the actual segment count for a given cell ID.
    
    Args:
        cell_id (str): The cell ID to look up
        
    Returns:
        int: The actual segment count for the cell, or fallback values if CSV not available:
             - 999 for cortical cells (PT, CT, IT types that typically lack segment ID 1)
             - 1000 for other cells (thalamic, etc.)
             
    Note:
        For proper functionality, run '01_extract_cell_data.py' to generate 'extracted_cell_data.csv'
        from your .cell.nml morphology files.
    """
    segment_data = _load_segment_data()
    
    # If we have actual data, use it
    if cell_id in segment_data:
        return int(segment_data[cell_id])
    
    # Fallback logic when CSV is not available
    # Determine if this is likely a cortical cell based on common prefixes
    cortical_prefixes = [
        'cADpyr', 'bSTUT', 'cSTUT', 'dSTUT',  # PT and CT cells
        'bAC', 'bIR', 'bNAC', 'dNAC', 'cNAC', 'cACint', 'cIR',  # IT cells
    ]
    
    is_cortical = any(cell_id.startswith(prefix) for prefix in cortical_prefixes)
    
    if is_cortical:
        # Cortical cells typically have fewer segments and lack segment ID 1
        return 999
    else:
        # Other cells (thalamic, etc.) typically have more complete segment sets
        return 1000

def get_all_segment_counts():
    """
    Get all segment counts as a dictionary.
    
    Returns:
        dict: Dictionary mapping cell_id to member_segment_count
              Returns empty dict if CSV file is not available.
    """
    return _load_segment_data().copy()

def is_csv_available():
    """
    Check if the extracted_cell_data.csv file is available.
    
    Returns:
        bool: True if CSV file exists and can be loaded, False otherwise
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'extracted_cell_data.csv')
    return os.path.exists(csv_path)
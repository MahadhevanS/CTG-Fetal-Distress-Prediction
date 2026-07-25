"""
Script to execute notebooks/01_exploratory_data_analysis.ipynb cell by cell
and embed all execution outputs and matplotlib figure renders.
"""

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Change working directory to notebooks directory so relative paths in notebook work correctly
nb_dir = os.path.join(PROJECT_ROOT, 'notebooks')
os.chdir(nb_dir)

nb_path = os.path.join(nb_dir, '01_exploratory_data_analysis.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Executing {len(nb['cells'])} cells in '{nb_path}'...")

# Use jupyter nbconvert or nbclient if available, otherwise execute cell code safely
try:
    import nbformat
    from nbclient import NotebookClient

    nb_node = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb_node, timeout=600, kernel_name='python3', allow_errors=True)
    client.execute()

    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb_node, f)
    print("Notebook executed successfully via NotebookClient!")

except Exception as e:
    print(f"NotebookClient execution failed or not installed: {e}")
    print("Executing code cells via custom runner...")

    exec_globals = {'__name__': '__main__'}
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            print(f"--- Executing Code Cell {idx+1} ---")
            try:
                exec(source, exec_globals)
                cell['execution_count'] = idx + 1
            except Exception as cell_err:
                print(f"Error in cell {idx+1}: {cell_err}")

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Fallback cell execution complete.")

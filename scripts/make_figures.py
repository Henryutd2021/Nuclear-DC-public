"""Render the paper figures by executing notebooks/figures.ipynb from the repository root.

    python scripts/make_figures.py                       # all figures
    python scripts/make_figures.py --figure fig6_policy_sensitivity

PDFs are written to figures/.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
notebook = ROOT / 'notebooks/figures.ipynb'
SETUP_CELLS = range(0, 5)
FIGURE_CELLS = {
    'fig2_inputs_overview': 6,
    'fig3_baseline_economics': 8,
    'fig4_operation_value': 10,
    'fig5_boundary_atlas': 12,
    'fig6_policy_sensitivity': 14,
    'figS1_size_matching': 16,
    'figS2_cooling_response': 18,
}
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--figure', choices=list(FIGURE_CELLS), action='append',
                    help='Render only the named figure; repeat to select several figures.')
args = parser.parse_args()
selected = set(FIGURE_CELLS.values()) if not args.figure else {FIGURE_CELLS[n] for n in args.figure}
env = {'__name__': '__figures_notebook__'}
for i, cell in enumerate(json.loads(notebook.read_text())['cells']):
    if cell['cell_type'] == 'code' and (i in SETUP_CELLS or i in selected):
        source = ''.join(cell.get('source', []))
        exec(compile(source, f'{notebook.name}:cell{i}', 'exec'), env)  # noqa: S102

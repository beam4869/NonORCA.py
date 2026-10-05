"""Render the CCUS manuscript figures with the recovered plotting functions.

Copies archived CSVs into outputs before plotting, because the original
functions also write projection-source CSVs. Includes the condition-2 grouped
projection function that is not invoked by its original script's main().
"""
from pathlib import Path
import argparse
import importlib.util
import shutil

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'experiments/ccus/scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/ccus_figures')
    output = parser.parse_args().output.resolve()
    source = ROOT/'experiments/ccus/results'
    if output == source or source in output.parents:
        parser.error('Choose an output directory outside the archived results.')
    data = output/'source_data'
    data.mkdir(parents=True,exist_ok=True)
    for path in source.glob('*.csv'):
        shutil.copyfile(path, data/path.name)
    one = load('ccus_figure_condition1','plot_revised_pareto_demonstrations.py')
    two = load('ccus_figure_condition2','plot_supply_chain_condition2_pareto.py')
    for module in [one,two]:
        module.RESULT_DIR = data
        module.FIGURE_DIR = output
    one.main()
    frontier,reduced,lower,ranges,summary,orca = two.load_data()
    two.plot_style_matched_frontier_orca(frontier,orca)
    two.plot_grouped_projections(frontier,reduced,lower,ranges,summary)
    print('CCUS figures written to',output)


if __name__ == '__main__':
    main()

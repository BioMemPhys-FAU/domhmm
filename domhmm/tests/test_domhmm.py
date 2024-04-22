"""
Unit and regression test for the domhmm package.
"""

# Import package, test suite, and other packages as needed
import domhmm
import sys
import os
import MDAnalysis as mda

def test_domhmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "domhmm" in sys.modules


def test_run():
    """Demo testing to try run """
    directory = os.getcwd()
    path2xtc = f"{directory}/data/md_center_mol_last2mus.xtc"
    path2tpr = f"{directory}/data/mem.tpr"
    uni = mda.Universe(path2tpr, path2xtc)

    # CHOL is out for this run because it doesn't have two tails. Not standard phospholipid
    membrane_select = "resname DPPC DIPC"
    tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
             "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}
    domhmm.PropertyCalculation(universe_or_atomgroup=uni,
                               leaflet_kwargs={"select": "name PO4", "pbc": True},
                               membrane_select= membrane_select,
                               tails=tails)\
        .run(start=0, stop=100)

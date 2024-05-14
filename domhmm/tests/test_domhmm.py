"""
Unit and regression test for the domhmm package.
"""

# Import package, test suite, and other packages as needed
import domhmm
import sys
import MDAnalysis as mda

def test_domhmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "domhmm" in sys.modules


def test_run():
    """Demo testing to try run """
    path2xtc = "domhmm/tests/data/md_center_mol_last2mus.xtc"
    path2tpr = "domhmm/tests/data/mem.tpr"
    uni = mda.Universe(path2tpr, path2xtc)

    membrane_select = "resname DPPC DIPC CHOL"
    heads = {"DPPC": "PO4",
             "DIPC": "PO4"}
    tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
             "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}
    sterols = {"CHOL": ["ROH", "C1"]}

    domhmm.PropertyCalculation(universe_or_atomgroup=uni,
                               leaflet_kwargs={"select": "name PO4", "pbc": True},
                               membrane_select=membrane_select,
                               heads=heads,
                               sterols=sterols,
                               tails=tails)\
        .run(start=0, stop=100)

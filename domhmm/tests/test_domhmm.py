"""
Unit and regression test for the domhmm package.
"""

# Import package, test suite, and other packages as needed
import domhmm
import pytest
import sys
import MDAnalysis as mda

def test_domhmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "domhmm" in sys.modules


def test_mdanalysis_logo_length(mdanalysis_logo_text):
    """Example test using a fixture defined in conftest.py"""
    logo_lines = mdanalysis_logo_text.split("\n")
    assert len(logo_lines) == 46, "Logo file does not have 46 lines!"


def test_run():
    """Demo testing to try run """
    path2xtc = "data/md_center_mol_last2mus.xtc"
    path2tpr = "data/mem.tpr"
    uni = mda.Universe(path2tpr, path2xtc)
    # TODO Parameters of PropertyCalculation
    #   * leaflet_kwargs:
    #       * Is select option correct
    #       * Is pbc (periodic boundary condtion) is correct
    #   * heads & tails list
    #       * why and how to select
    #       * error if not given

    # CHOL is out for this run because it doesn't have two tails. Not standard phospholipid
    membrane_select = "resname DPPC DIPC"
    heads = {"DPPC": ["NC3", "PO4"],
             "DIPC": ["NC3", "PO4"]}
    tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
             "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}
    domhmm.PropertyCalculation(universe_or_atomgroup=uni,
                               leaflet_kwargs={"select": "name PO4", "pbc": True},
                               membrane_select= membrane_select,
                               heads= heads,
                               tails=tails)\
        .run(start=10, stop=20)

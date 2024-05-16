"""
Unit and regression test for the domhmm package.
"""

# Import package, test suite, and other packages as needed
import pytest
import os
import MDAnalysis as mda
from ..analysis import base


class TestBase:
    @pytest.fixture(scope="class")
    def universe(self):
        test_dir = os.path.dirname(__file__)
        path2xtc = os.path.join(test_dir, "data/md_center_mol_last2mus.xtc")
        path2tpr = os.path.join(test_dir, "data/mem.tpr")
        uni = mda.Universe(path2tpr, path2xtc)
        return uni

    @pytest.fixture(scope="class")
    def analysis(self, universe):
        membrane_select = "resname DPPC DIPC CHOL"
        heads = {"DPPC": "PO4",
                 "DIPC": "PO4"}
        tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
                 "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}
        sterols = {"CHOL": ["ROH", "C1"]}

        return base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                        leaflet_kwargs={"select": "name PO4", "pbc": True},
                                        membrane_select=membrane_select,
                                        heads=heads,
                                        sterols=sterols,
                                        tails=tails)

    def test_run(self, universe):
        """Demo testing for base class """

        membrane_select = "resname DPPC DIPC CHOL"
        heads = {"DPPC": "PO4",
                 "DIPC": "PO4"}
        tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
                 "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}
        sterols = {"CHOL": ["ROH", "C1"]}

        base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                 leaflet_kwargs={"select": "name PO4", "pbc": True},
                                 membrane_select=membrane_select,
                                 heads=heads,
                                 sterols=sterols,
                                 tails=tails)

    def test_check_parameters(self, analysis):
        assert analysis.membrane_unique_resids.size == 720
        assert (analysis.unique_resnames == ['DPPC', 'DIPC', 'CHOL']).all()
        assert len(analysis.leaflet_selection) == 2
        assert analysis.sterols_tail.keys() == {"CHOL"}
        assert analysis.n_leaflets == 2


    def test_get_leaflets(self, analysis):
        leaflet_selection = analysis.get_leaflets()
        assert len(leaflet_selection) == 2
        assert leaflet_selection.keys() == {'0', '1'}
        assert leaflet_selection['0'].n_atoms == 355
        assert leaflet_selection['1'].n_atoms == 365

    def test_get_resids(self, analysis):
        residue_ids = analysis.get_resids()
        assert len(residue_ids) == 3
        assert residue_ids.keys() == {'DPPC', 'DIPC', 'CHOL'}
        assert residue_ids['DPPC'].shape == (302,)
        assert residue_ids['DIPC'].shape == (202,)
        assert residue_ids['CHOL'].shape == (216,)

    def test_get_leaflet_sterols(self, analysis):
        sterols_tail = analysis.get_leaflet_sterols()
        assert len(sterols_tail) == 1
        assert sterols_tail.keys() == {'CHOL'}
        assert sterols_tail['CHOL'].n_atoms == 432

    def test_get_leaflet_tails(self, analysis):
       resid_tails_selection = analysis.get_leaflet_tails()
       assert len(resid_tails_selection) == 2
       assert resid_tails_selection.keys() == {0, 1}
       assert resid_tails_selection[0].n_atoms == 2016
       assert resid_tails_selection[1].n_atoms == 2016

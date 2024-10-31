"""
Unit and regression test for the domhmm package.
"""

import os

import MDAnalysis as mda
import pytest

from ..analysis import base


class TestBase:
    @pytest.fixture(scope="class")
    def universe(self):
        """
        MDA universe of test environment
        """
        test_dir = os.path.dirname(__file__)
        path2xtc = os.path.join(test_dir, "data/md_center_mol_last2mus.xtc")
        path2tpr = os.path.join(test_dir, "data/mem.tpr")
        uni = mda.Universe(path2tpr, path2xtc)
        return uni

    @staticmethod
    def base_test_inputs():
        """
        General inputs for all DomHMM PropertyCalculation class
        """
        membrane_select = "resname DPPC DIPC CHOL"
        heads = {"DPPC": "PO4",
                 "DIPC": "PO4"}
        tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
                 "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}
        sterol_heads = {"CHOL": "ROH"}
        sterol_tails = {"CHOL": ["ROH", "C1"]}
        return membrane_select, heads, tails, sterol_heads, sterol_tails

    @pytest.fixture(scope="class")
    def analysis(self, universe):
        """
        Standard analysis options
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.base_test_inputs()

        return base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                        leaflet_kwargs={"select": "name PO4", "pbc": True},
                                        membrane_select=membrane_select,
                                        leaflet_select="auto",
                                        heads=heads,
                                        sterol_heads=sterol_heads,
                                        sterol_tails=sterol_tails,
                                        tails=tails)

    def test_gmm_hmm_input(self, universe):
        """
        Analysis option with GMM and HMM arguments
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.base_test_inputs()

        gmm_kwargs = {"tol": 1E-4, "init_params": 'k-means++', "verbose": 0,
                      "max_iter": 10000, "n_init": 20,
                      "warm_start": False, "covariance_type": "full"}

        hmm_kwargs = {"verbose": False, "tol": 1E-4, "n_iter": 2000,
                      "algorithm": "viterbi", "covariance_type": "full",
                      "init_params": "st", "params": "stmc"}

        analysis = base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                            leaflet_kwargs={"select": "name PO4", "pbc": True},
                                            gmm_kwargs=gmm_kwargs,
                                            hmm_kwargs=hmm_kwargs,
                                            membrane_select=membrane_select,
                                            leaflet_select="auto",
                                            heads=heads,
                                            sterol_heads=sterol_heads,
                                            sterol_tails=sterol_tails,
                                            tails=tails)

        assert analysis.membrane_unique_resids.size == 720
        assert (analysis.unique_resnames == ['DPPC', 'DIPC', 'CHOL']).all()
        assert analysis.sterol_tails_selection.keys() == {"CHOL"}
        assert analysis.n_leaflets == 2

    def test_leaflet_select_exceptions(self, universe):
        """
        Testing leaflet_select parameter's exceptions
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.base_test_inputs()
        # Catch errors for wrong leaflet selection option
        # Test None input
        leaflet_select = None
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select=leaflet_select,
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tails=tails)
        # Test wrong string input (anything except "auto")
        leaflet_select = "Wrong String"
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select=leaflet_select,
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tails=tails)
        # Test single element list (two needed for both upper and lower leaflet)
        leaflet_select = ["Single Leaflet"]
        with pytest.raises(AssertionError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select=leaflet_select,
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tails=tails)
        # Test wrong type list (either string or AtomGroup is required)
        leaflet_select = [[1], [2]]
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select=leaflet_select,
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tails=tails)
        # Test wrong MDA query strings for both leaflet
        leaflet_select = ["Wrong MDA", "Query"]
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select=leaflet_select,
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tails=tails)
        # Test sterol involment (only lipids in leaflet selection)
        leaflet_select = ["resname CHOL", "resname CHOL"]
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select=leaflet_select,
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tails=tails)

    def test_leaflet_select_parameter(self, universe):
        """
        Testing leaflet_select parameter
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.base_test_inputs()
        # With MDA Query String
        leaflet_select = ["resid 1:252", "resid 361:612"]
        base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                 leaflet_kwargs={"select": "name PO4", "pbc": True},
                                 membrane_select=membrane_select,
                                 leaflet_select=leaflet_select,
                                 heads=heads,
                                 sterol_heads=sterol_heads,
                                 sterol_tails=sterol_tails,
                                 tails=tails)
        # With MDA Atom Group
        upper_leaflet = universe.select_atoms("resid 1:252")
        lower_leaflet = universe.select_atoms("resid 361:612")
        leaflet_select = [upper_leaflet, lower_leaflet]
        base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                 leaflet_kwargs={"select": "name PO4", "pbc": True},
                                 membrane_select=membrane_select,
                                 leaflet_select=leaflet_select,
                                 heads=heads,
                                 sterol_heads=sterol_heads,
                                 sterol_tails=sterol_tails,
                                 tails=tails)

    def test_tmd_protein_list_exceptions(self, universe):
        """
        Testing tmd_protein_list parameter's exceptions
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.base_test_inputs()
        # Test tmd_protein_list format
        tmd_protein = "Wrong Format"
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select="auto",
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tmd_protein_list=tmd_protein,
                                     tails=tails)
        # Test wrong element format (should be only dicts)
        tmd_protein = ["Wrong Format"]
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select="auto",
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tmd_protein_list=tmd_protein,
                                     tails=tails)
        # Test wrong key names ("0" for upper, "1" for lower)
        tmd_protein = [{"2": [], "3": []}]
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select="auto",
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tmd_protein_list=tmd_protein,
                                     tails=tails)
        # Test wrong values (AtomGroup or string query only)
        tmd_protein = [{"0": 0, "1": 1}]
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select="auto",
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tmd_protein_list=tmd_protein,
                                     tails=tails)
        # Test wrong MDA query
        tmd_protein = [{"0": "Wrong Query", "1": "Wrong Query"}]
        with pytest.raises(ValueError):
            base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                     leaflet_kwargs={"select": "name PO4", "pbc": True},
                                     membrane_select=membrane_select,
                                     leaflet_select="auto",
                                     heads=heads,
                                     sterol_heads=sterol_heads,
                                     sterol_tails=sterol_tails,
                                     tmd_protein_list=tmd_protein,
                                     tails=tails)

    def test_tmd_protein_list_parameter(self, universe):
        """
        Testing tmd_protein_list parameter
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.base_test_inputs()
        # Test tmd_protein_list with string query
        tmd_protein_list = [{"0": "resid 1:3", "1": "resid 1:3"}]
        base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                 leaflet_kwargs={"select": "name PO4", "pbc": True},
                                 membrane_select=membrane_select,
                                 leaflet_select="auto",
                                 heads=heads,
                                 sterol_heads=sterol_heads,
                                 sterol_tails=sterol_tails,
                                 tmd_protein_list=tmd_protein_list,
                                 tails=tails)
        # Test tmd_protein_list with AtomGroup
        demo_protein = universe.select_atoms("resid 1:3")
        tmd_protein_list = [{"0": demo_protein, "1": demo_protein}]
        base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                 leaflet_kwargs={"select": "name PO4", "pbc": True},
                                 membrane_select=membrane_select,
                                 leaflet_select="auto",
                                 heads=heads,
                                 sterol_heads=sterol_heads,
                                 sterol_tails=sterol_tails,
                                 tmd_protein_list=tmd_protein_list,
                                 tails=tails)
        # Test multiple protein in tmd_protein_list
        tmd_protein_list = [{"0": "resid 1:3", "1": "resid 1:3"}, {"0": "resid 1:3", "1": "resid 1:3"}]
        base.LeafletAnalysisBase(universe_or_atomgroup=universe,
                                 leaflet_kwargs={"select": "name PO4", "pbc": True},
                                 membrane_select=membrane_select,
                                 leaflet_select="auto",
                                 heads=heads,
                                 sterol_heads=sterol_heads,
                                 sterol_tails=sterol_tails,
                                 tmd_protein_list=tmd_protein_list,
                                 tails=tails)

    def test_check_parameters(self, analysis):
        """
        Checking initial parameters
        """
        assert analysis.membrane_unique_resids.size == 720
        assert (analysis.unique_resnames == ['DPPC', 'DIPC', 'CHOL']).all()
        assert analysis.sterol_tails_selection.keys() == {"CHOL"}
        assert analysis.n_leaflets == 2

    def test_get_leaflets(self, analysis):
        """
        Unit test of get_leaflets function
        """
        leaflet_selection = analysis.get_leaflets()
        assert len(leaflet_selection) == 2
        assert leaflet_selection.keys() == {'0', '1'}
        assert leaflet_selection['0'].n_atoms == 252
        assert leaflet_selection['1'].n_atoms == 252

    def test_get_resids(self, analysis):
        """
        Unit test of get_resids function
        """
        residue_ids = analysis.get_resids()
        assert len(residue_ids) == 3
        assert residue_ids.keys() == {'DPPC', 'DIPC', 'CHOL'}
        assert residue_ids['DPPC'].shape == (302,)
        assert residue_ids['DIPC'].shape == (202,)
        assert residue_ids['CHOL'].shape == (216,)

    def test_get_leaflet_sterols(self, analysis):
        """
        Unit test of get_leaflet_sterols function
        """
        sterols_tail = analysis.get_leaflets_sterol()
        assert len(sterols_tail) == 2
        assert sterols_tail.keys() == {'0', '1'}
        assert sterols_tail['0'].n_atoms == 355
        assert sterols_tail['1'].n_atoms == 365

    def test_get_leaflet_tails(self, analysis):
        """
        Unit test of get_leaflet_tails function
        """
        resid_tails_selection = analysis.get_lipid_tails()
        assert len(resid_tails_selection) == 2
        assert resid_tails_selection.keys() == {0, 1}
        assert resid_tails_selection[0].n_atoms == 2016
        assert resid_tails_selection[1].n_atoms == 2016

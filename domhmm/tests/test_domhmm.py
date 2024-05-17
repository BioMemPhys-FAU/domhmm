"""
Unit and regression test for the domhmm package.
"""

# Import package, test suite, and other packages as needed
import domhmm
import sys
import pytest
import os
import pickle
import MDAnalysis as mda


class TestDomhmm:
    @pytest.fixture(scope="class")
    def universe(self):
        test_dir = os.path.dirname(__file__)
        path2xtc = os.path.join(test_dir, "data/md_center_mol_last2mus.xtc")
        path2tpr = os.path.join(test_dir, "data/mem.tpr")
        uni = mda.Universe(path2tpr, path2xtc)
        return uni

    @pytest.fixture(scope="function")
    def analysis(self, universe):
        membrane_select = "resname DPPC DIPC CHOL"
        heads = {"DPPC": "PO4",
                 "DIPC": "PO4"}
        tails = {"DPPC": [["C1B", "C2B", "C3B", "C4B"], ["C1A", "C2A", "C3A", "C4A"]],
                 "DIPC": [["C1B", "D2B", "D3B", "C4B"], ["C1A", "D2A", "D3A", "C4A"]]}
        sterols = {"CHOL": ["ROH", "C1"]}

        return domhmm.PropertyCalculation(universe_or_atomgroup=universe,
                                          leaflet_kwargs={"select": "name PO4", "pbc": True},
                                          membrane_select=membrane_select,
                                          heads=heads,
                                          sterols=sterols,
                                          tails=tails)

    def test_domhmm_imported(self):
        """Sample test, will always pass so long as import statement worked"""
        assert "domhmm" in sys.modules

    @staticmethod
    def result_parameter_check(analysis):
        assert analysis.results['GMM']['DPPC'].converged_
        assert analysis.results['GMM']['DIPC'].converged_
        assert analysis.results['GMM']['CHOL'].converged_
        assert len(analysis.results['HMM_Pred']) == 3
        assert analysis.results['HMM_Pred'].keys() == {'DPPC', 'DIPC', 'CHOL'}
        assert analysis.results['HMM_Pred']['DPPC'].shape == (302, 100)
        assert analysis.results['HMM_Pred']['DIPC'].shape == (202, 100)
        assert analysis.results['HMM_Pred']['CHOL'].shape == (216, 100)
        assert len(analysis.results['Getis_Ord']) == 2

    def test_run(self, analysis):
        """Demo testing to try run """
        print(analysis.results)
        analysis.run(start=0, stop=100)
        self.result_parameter_check(analysis)

    # TODO tests
    def test_calc_order_parameter(self):
        pass

    def test_area_per_lipid_vor(self, analysis):
        print(analysis)
        boxdim = analysis.universe.trajectory.ts.dimensions[0:3]
        upper_vor, upper_apl = analysis.area_per_lipid_vor(leaflet=0, boxdim=boxdim, frac=analysis.frac)
        lower_vor, lower_apl = analysis.area_per_lipid_vor(leaflet=1, boxdim=boxdim, frac=analysis.frac)
        try:
            test_dir = os.path.dirname(__file__)
            with open(os.path.join(test_dir, "data/first_upper_vor.pickle"), "rb") as f:
                test_upper_vor = pickle.load(f)
            with open(os.path.join(test_dir, "data/first_upper_apl.pickle"), "rb") as f:
                test_upper_apl = pickle.load(f)
            with open(os.path.join(test_dir, "data/lower_lower_vor.pickle"), "rb") as f:
                test_lower_vor = pickle.load(f)
            with open(os.path.join(test_dir, "data/first_lower_apl.pickle"), "rb") as f:
                test_lower_apl = pickle.load(f)
        except FileNotFoundError:
            print("Test data files for area per lipid are not found.")
        assert (test_upper_vor.points == upper_vor.points).all()
        assert (test_upper_apl == upper_apl).all()
        assert (test_lower_vor.points == lower_vor.points).all()
        assert (test_lower_apl == lower_apl).all()


    def test_weight_matrix(self):
        pass

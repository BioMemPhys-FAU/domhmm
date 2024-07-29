"""
Unit and regression test for the domhmm package.
"""
import os
import pickle
import sys

import MDAnalysis as mda
import numpy as np
import pytest

# Import package, test suite, and other packages as needed
import domhmm

error_tolerance = 0.001


class TestDomhmm:

    @pytest.fixture(scope="function")
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
    def domhmm_test_inputs():
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

    @pytest.fixture(scope="function")
    def analysis(self, universe):
        """
        Standard analysis options
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.domhmm_test_inputs()
        return domhmm.PropertyCalculation(universe_or_atomgroup=universe,
                                          leaflet_kwargs={"select": "name PO4", "pbc": True},
                                          membrane_select=membrane_select,
                                          leaflet_select="auto",
                                          heads=heads,
                                          sterol_heads=sterol_heads,
                                          sterol_tails=sterol_tails,
                                          tails=tails,
                                          result_plots=True)

    @pytest.fixture(scope="function")
    def analysis_reuse_hmm_model(self, universe):
        """
        Analysis option with reusing of HMM models
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.domhmm_test_inputs()
        test_dir = os.path.dirname(__file__)
        with open(os.path.join(test_dir, "data/symmetric_hmm.pickle"), "rb") as f:
            trained_hmm = pickle.load(f)
        return domhmm.PropertyCalculation(universe_or_atomgroup=universe,
                                          leaflet_kwargs={"select": "name PO4", "pbc": True},
                                          membrane_select=membrane_select,
                                          leaflet_select="auto",
                                          heads=heads,
                                          sterol_heads=sterol_heads,
                                          sterol_tails=sterol_tails,
                                          tails=tails,
                                          result_plots=True,
                                          trained_hmms=trained_hmm)

    @pytest.fixture(scope="function")
    def analysis_asymmetric(self, universe):
        """
        Analysis option with asymmetric membrane simulation
        """
        membrane_select, heads, tails, sterol_heads, sterol_tails = self.domhmm_test_inputs()
        return domhmm.PropertyCalculation(universe_or_atomgroup=universe,
                                          leaflet_kwargs={"select": "name PO4", "pbc": True},
                                          membrane_select=membrane_select,
                                          leaflet_select="auto",
                                          heads=heads,
                                          sterol_heads=sterol_heads,
                                          sterol_tails=sterol_tails,
                                          tails=tails,
                                          verbose=True,
                                          result_plots=True,
                                          asymmetric_membrane=True)

    @pytest.fixture(scope="class")
    def apl_results(self):
        """
        Area per lipid test data loading
        """
        result = {}
        try:
            test_dir = os.path.dirname(__file__)
            with open(os.path.join(test_dir, "data/first_upper_vor.pickle"), "rb") as f:
                result["test_upper_vor"] = pickle.load(f)
            with open(os.path.join(test_dir, "data/first_upper_apl.pickle"), "rb") as f:
                result["test_upper_apl"] = pickle.load(f)
            with open(os.path.join(test_dir, "data/first_upper_pbc_idx.pickle"), "rb") as f:
                result["test_upper_pbc_idx"] = pickle.load(f)
            with open(os.path.join(test_dir, "data/first_lower_vor.pickle"), "rb") as f:
                result["test_lower_vor"] = pickle.load(f)
            with open(os.path.join(test_dir, "data/first_lower_apl.pickle"), "rb") as f:
                result["test_lower_apl"] = pickle.load(f)
            with open(os.path.join(test_dir, "data/first_lower_pbc_idx.pickle"), "rb") as f:
                result["test_lower_pbc_idx"] = pickle.load(f)
        except FileNotFoundError:
            print("Test data files for area per lipid are not found.")
        return result

    @pytest.fixture(scope="class")
    def weight_results(self):
        """
        Weight matrix test data loading
        """
        result = {}
        try:
            test_dir = os.path.dirname(__file__)
            with open(os.path.join(test_dir, "data/first_upper_weight.pickle"), "rb") as f:
                result["test_upper_weight"] = pickle.load(f)
            with open(os.path.join(test_dir, "data/first_lower_weight.pickle"), "rb") as f:
                result["test_lower_weight"] = pickle.load(f)
        except FileNotFoundError:
            print("Test data files for area per lipid are not found.")
        return result

    @pytest.fixture(scope="class")
    def order_parameters_results(self):
        """
        Order parameters test data loading
        """
        try:
            test_dir = os.path.dirname(__file__)
            with open(os.path.join(test_dir, "data/first_order_parameters.pickle"), "rb") as f:
                result_dict = pickle.load(f)
        except FileNotFoundError:
            print("Test data files for area per lipid are not found.")
        return result_dict

    @staticmethod
    def result_parameter_check(analysis, test_type):
        """
        Result parameter checking for each type of analysis tests
        """
        if test_type == "analysis":
            assert analysis.results['GMM']['DPPC'].converged_
            assert analysis.results['GMM']['DIPC'].converged_
            assert analysis.results['GMM']['CHOL'].converged_
            assert analysis.results['HMM_Pred']['DPPC'].shape == (302, 100)
            assert analysis.results['HMM_Pred']['DIPC'].shape == (202, 100)
            assert analysis.results['HMM_Pred']['CHOL'].shape == (216, 100)
        elif test_type == "analysis_asymmetric":
            for leaflet in range(2):
                assert analysis.results['GMM']['DPPC'][leaflet].converged_
                assert analysis.results['GMM']['DIPC'][leaflet].converged_
                assert analysis.results['GMM']['CHOL'][leaflet].converged_
            assert analysis.results['HMM_Pred']['DPPC'].shape == (302, 20)
            assert analysis.results['HMM_Pred']['DIPC'].shape == (202, 20)
            assert analysis.results['HMM_Pred']['CHOL'].shape == (216, 20)
        elif test_type == "analysis_reuse_hmm_model":
            assert analysis.results['HMM_Pred']['DPPC'].shape == (302, 100)
            assert analysis.results['HMM_Pred']['DIPC'].shape == (202, 100)
            assert analysis.results['HMM_Pred']['CHOL'].shape == (216, 100)
        assert len(analysis.results['HMM_Pred']) == 3
        assert analysis.results['HMM_Pred'].keys() == {'DPPC', 'DIPC', 'CHOL'}
        assert len(analysis.results['Getis_Ord']) == 4

    def test_domhmm_imported(self):
        """
        Sample test, will always pass so long as import statement worked
        """
        assert "domhmm" in sys.modules

    def test_run(self, analysis):
        """
        Demo run with standard options
        """
        analysis.run(start=0, stop=100)
        self.result_parameter_check(analysis, "analysis")

    def test_run_reuse_hmm_model(self, analysis_reuse_hmm_model):
        """
        Demo run with reusing of HMM models option
        """
        analysis_reuse_hmm_model.run(start=0, stop=100)
        self.result_parameter_check(analysis_reuse_hmm_model, "analysis_reuse_hmm_model")

    def test_run_asymmetric(self, analysis_asymmetric):
        """
        Demo run with asymmetric membrane option
        """
        analysis_asymmetric.run(start=0, stop=100, step=5)
        self.result_parameter_check(analysis_asymmetric, "analysis_asymmetric")

    def test_calc_order_parameter(self, analysis, order_parameters_results):
        """
        Unit test of calculation of order parameter function
        """
        result = []
        for chain, tail in analysis.resid_tails_selection.items():
            s_cc = analysis.calc_order_parameter(tail)
            result.append(s_cc)
        for i, (resname, tail) in enumerate(analysis.sterol_tails_selection.items()):
            s_cc = analysis.calc_order_parameter(tail)
            result.append(s_cc)
        assert np.allclose(order_parameters_results["SCC_0"], result[0], error_tolerance)
        assert np.allclose(order_parameters_results["SCC_1"], result[1], error_tolerance)
        assert np.allclose(order_parameters_results["CHOL"], result[2], error_tolerance)

    def test_area_per_lipid_vor(self, analysis, apl_results):
        """
        Unit test of calculation of area per lipid function
        """
        boxdim = analysis.universe.trajectory.ts.dimensions[0:3]
        analysis.leaflet_selection_no_sterol = analysis.get_leaflets()
        analysis.leaflet_selection = analysis.get_leaflets_sterol()
        upper_vor, upper_apl, upper_pbc_idx = analysis.area_per_lipid_vor(leaflet=0, boxdim=boxdim, frac=analysis.frac)
        lower_vor, lower_apl, lower_pbc_idx = analysis.area_per_lipid_vor(leaflet=1, boxdim=boxdim, frac=analysis.frac)
        assert np.allclose(apl_results["test_upper_vor"].points, upper_vor.points, error_tolerance)
        assert np.allclose(apl_results["test_upper_apl"], upper_apl, error_tolerance)
        assert np.allclose(apl_results["test_upper_pbc_idx"], upper_pbc_idx, error_tolerance)
        assert np.allclose(apl_results["test_lower_vor"].points, lower_vor.points, error_tolerance)
        assert np.allclose(apl_results["test_lower_apl"], lower_apl, error_tolerance)
        assert np.allclose(apl_results["test_lower_pbc_idx"], lower_pbc_idx, error_tolerance)

    def test_weight_matrix(self, analysis, apl_results, weight_results):
        """
        Unit test of calculation of weight matrix function
        """
        analysis.leaflet_selection_no_sterol = analysis.get_leaflets()
        analysis.leaflet_selection = analysis.get_leaflets_sterol()
        upper_weight = analysis.weight_matrix(apl_results["test_upper_vor"], pbc_idx=apl_results["test_upper_pbc_idx"],
                                              leaflet=0)
        lower_weight = analysis.weight_matrix(apl_results["test_lower_vor"], pbc_idx=apl_results["test_lower_pbc_idx"],
                                              leaflet=1)
        assert np.allclose(weight_results["test_upper_weight"], upper_weight, error_tolerance)
        assert np.allclose(weight_results["test_lower_weight"], lower_weight, error_tolerance)

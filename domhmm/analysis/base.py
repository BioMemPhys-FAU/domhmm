"""
DomHMM --- :mod:`domhmm.analysis.LeafletAnalysisBase`
===========================================================

This module contains the :class:`LeafletAnalysisBase` class.

"""

# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
import sklearn
from MDAnalysis.analysis import distances
# ----MDANALYSIS---- #
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.leaflet import LeafletFinder
# if TYPE_CHECKING:
from MDAnalysis.core.universe import Universe, AtomGroup


class LeafletAnalysisBase(AnalysisBase):
    """
    LeafletAnalysisBase class.

    This class marks the start point for each analysis method. 

    It connects to the MDAnalysis core library, setups the MDAnalysis universe, and provides coordinates for each
    membrane leaflet.

    Parameters
    ----------
    universe_or_atomgroup: :class:`~MDAnalysis.core.universe.Universe` or :class:`~MDAnalysis.core.groups.AtomGroup`
        Universe or group of atoms to apply this analysis to.
        If a trajectory is associated with the atoms,
        then the computation iterates over the trajectory.
    membrane_select: str
        Membrane selection query for analysis of simulation trajectories.
    gmm_kwargs: Optional[Dict]
        Optional parameter for Gaussian mixture model function.
    hmm_kwargs: Optional[Dict]
        Optional parameter for Hidden Markov model function.
    leaflet_kwargs: Optional[dict]
        dictionary containing additional arguments for the MDAnalysis LeafletFinder
    leaflet_select: Union["auto", List[AtomGroup], List[str]]
        Leaflet selection options for lipids which can be automatic by finding Leafletfinder, atomgroup, string query or
         list
    heads: Dict[str, Any]
        dictionary containing residue name and atom selection for lipid head groups
    tails: Dict[str, Any]
        dictionary containing residue name and atom selection for lipid tail groups
    sterol_heads: Dict[str, Any]
        dictionary containing residue name and atom selection for sterol head groups
    sterol_tails: Dict[str, Any]
         dictionary containing residue name and atom selection for sterol tail groups
         (head as first, tail beginning as second)
    tmd_protein_list: Union["auto", List[AtomGroup], List[str]]
         Transmembrane domain protein list to include area per lipid calculation
    frac: float
        fraction of box length in x and y outside the unit cell considered for Voronoi calculation
    p_value: float
        p_value for z_score calculation
    lipid_leaflet_rate: Union[None, int]
        Frame rate for checking lipids leaflet assignments via LeafletFinder
    sterol_leaflet_rate: int
        Frame rate for checking sterols leaflet assignments via LeafletFinder
    asymmetric_membrane: bool
        Asymmetric membrane option to train models by separated data w.r.t. leaflets
    verbose: bool
        Debug option to print step progress, warnings and errors
    result_plots: bool
        Plotting intermediate result option
    save_plots : bool
        Option for saving intermediate plots in pdf format
    trained_hmms: dict
        User-specific HMM (e.g., pre-trained on another simulation)
    do_clustering: bool
        Perform the hierarchical clustering for each frame
    n_init_hmm: int
        Number of repeats for HMM model trainings

    Attributes
    ----------
    universe_or_atomgroup: :class:`~MDAnalysis.core.universe.Universe`
        The universe to which this analysis is applied
    atomgroup: :class:`~MDAnalysis.core.groups.AtomGroup`
        The atoms to which this analysis is applied
    results: :class:`~MDAnalysis.analysis.base.Results`
        results of calculation are stored here, after calling
        :meth:`.run`
    start: Optional[int]
        The first frame of the trajectory used to compute the analysis
    stop: Optional[int]
        The frame to stop at for the analysis
    step: Optional[int]
        Number of frames to skip between each analyzed frame
    n_frames: int
        Number of frames analysed in the trajectory
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`.run`
    """

    def __init__(
            self,
            universe_or_atomgroup: Union["Universe", "AtomGroup"],
            membrane_select: str = "all",
            gmm_kwargs: Union[None, dict] = None,
            hmm_kwargs: Union[None, dict] = None,
            leaflet_kwargs: Dict[str, Any] = {},
            leaflet_select: Union[None, "AtomGroup", str, list] = None,
            heads: Dict[str, Any] = {},
            tails: Dict[str, Any] = {},
            sterol_heads: Dict[str, Any] = {},
            sterol_tails: Dict[str, Any] = {},
            tmd_protein_list: Union[None, "AtomGroup", str, list] = None,
            frac: float = 0.5,
            p_value: float = 0.05,
            lipid_leaflet_rate: Union[None, int] = None,
            sterol_leaflet_rate: int = 1,
            asymmetric_membrane: bool = False,
            verbose: bool = False,
            result_plots: bool = False,
            trained_hmms: Dict[str, Any] = {},
            n_init_hmm: int = 2,
            save_plots: bool = False,
            do_clustering: bool = True,
            curved_apl_cutoff: float = 30,
            **kwargs
    ):
        # the below line must be kept to initialize the AnalysisBase class!
        super().__init__(universe_or_atomgroup.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can be used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results
        self.resid_tails_selection = {}
        self.universe = universe_or_atomgroup.universe
        self.membrane = universe_or_atomgroup.select_atoms(membrane_select)
        self.membrane_unique_resids = np.unique(self.membrane.resids)
        self.resids_index_map = {resid: idx for idx, resid in enumerate(self.membrane_unique_resids)}
        self.index_resid_map = {idx: resid for idx, resid in enumerate(self.membrane_unique_resids)}
        self.heads = heads
        self.tails = tails
        self.sterol_heads = sterol_heads
        self.sterol_tails = sterol_tails
        self.lipid_leaflet_rate = lipid_leaflet_rate
        self.sterol_leaflet_rate = sterol_leaflet_rate
        self.frac = frac
        self.p_value = p_value
        self.asymmetric_membrane = asymmetric_membrane
        self.verbose = verbose
        self.result_plots = result_plots
        self.tmd_protein = None
        self.n_init_hmm = n_init_hmm
        self.save_plots = save_plots
        self.do_clustering = do_clustering
        self.cutoff = curved_apl_cutoff

        assert heads.keys() == tails.keys(), "Heads and tails don't contain same residue names"

        self.leaflet_kwargs = leaflet_kwargs
        self.n_leaflets = 0

        if gmm_kwargs is None:
            self.gmm_kwargs = {"tol": 1E-4, "init_params": 'random_from_data', "verbose": 0,
                               "max_iter": 10000, "n_init": 20,
                               "warm_start": False, "covariance_type": "full"}
        else:
            self.gmm_kwargs = gmm_kwargs

        if hmm_kwargs is None:
            self.hmm_kwargs = {"verbose": False, "tol": 1E-4, "n_iter": 2000,
                               "algorithm": "viterbi", "covariance_type": "full",
                               "init_params": "st", "params": "stmc"}
        else:
            self.hmm_kwargs = hmm_kwargs

        # -----------------------------------------------------------------LEAFLETS----------------------------------- #

        if leaflet_select is None:
            # No information about leaflet assignment is provided. Raise an error and exit.
            raise ValueError(
                "No leaflet assigned! Please provide a list containing either two MDAnalysis.AtomGroup objects, "
                "two valid MDAnalysis selection strings, or 'auto' to trigger automatic leaflet assignment.")

        elif isinstance(leaflet_select, list):
            # If the argument leaflet_select is a list, the user specified their own leaflet selection. The code
            # checks if there are exactly two groups and raises an assertion if not. It then stores the provided
            # AtomGroup in a dictionary, or creates the AtomGroup with a provided selection string and stores it in
            # the dictionary.

            assert len(leaflet_select) == 2, (
                f"A bilayer is required. {len(leaflet_select)} entries found in leaflet_select list.")

            # Initialize empty dictionary to store AtomGroups
            self.leaflet_selection_no_sterol = {}

            # Iterate over both entries
            for i in range(2):
                # MDAnalysis.AtomGroup was provided as input
                if isinstance(leaflet_select[i], AtomGroup):
                    self.leaflet_selection_no_sterol[str(i)] = leaflet_select[i]
                # Character string was provided as input, assume it contains a selection for an MDAnalysis.AtomGroup
                elif isinstance(leaflet_select[i], str):
                    # Try to create a MDAnalysis.AtomGroup, raise a ValueError if not selection group could be provided
                    try:
                        self.leaflet_selection_no_sterol[str(i)] = self.universe.select_atoms(leaflet_select[i])
                    except Exception as e:
                        raise ValueError("Please provide a valid MDAnalysis selection string!") from e

                    # TODO: Check for atom number in upper and lower leaflet and raise a Warning
                else:
                    raise ValueError("Please provide an MDAnalysis.AtomGroup or a valid MDAnalysis selection string!")

                # Iterate over sterol compounds and check if it is part of the phospholipid leaflet assignment
                for rsn, atoms in self.sterol_heads.items():
                    if rsn in self.leaflet_selection_no_sterol[str(i)].residues.resnames:
                        raise ValueError(
                            f"Sterol {rsn} should not be part of the initial leaflet identification! Sterols will be "
                            f"assigned automatically.")
                    else:
                        pass

        elif leaflet_select.lower() == "auto":
            # 'auto' should trigger an automated leaflet assignment pipeline
            # (e.g., LeafletFinder provided by MDAnalysis)
            self.leaflet_selection_no_sterol = self.get_leaflets()

        else:
            # An unknown argument is provided for leaflet_select
            raise ValueError(
                "No leaflet assigned! Please provide a list containing either two MDAnalysis.AtomGroup objects, "
                "two valid MDAnalysis selection strings, or 'auto' to trigger automatic leaflet assignment.")

        # -----------------------------------------------------------------Transmembrane Domains --------------------- #
        if isinstance(tmd_protein_list, list):
            # Initialize empty dictionary to store AtomGroups
            self.tmd_protein = {"0": [], "1": []}
            for each in tmd_protein_list:
                if not isinstance(each, dict):
                    raise ValueError(
                        "Entry for each TDM protein should be a dictionary in the format {'0': ..., '1': ...} "
                        "where 0 for lower leaflet and 1 for upper leaflet.")
                for leaflet, query in each.items():
                    if leaflet not in ["0", "1"]:
                        raise ValueError(
                            "Entry for each TDM protein should be a dictionary in the format {'0': ..., '1': ...} "
                            "where 0 for lower leaflet and 1 for upper leaflet.")
                    if isinstance(query, AtomGroup):
                        # Take center of geometry of three positions
                        self.tmd_protein[leaflet].append(query)
                        # cog = np.mean(query.positions, axis=0)
                        # self.tmd_protein[leaflet].append(cog)
                    # Character string was provided as input, assume it contains a selection for an MDAnalysis.AtomGroup
                    elif isinstance(query, str):
                        # Try to create a MDAnalysis.AtomGroup, raise a ValueError if not selection group could be
                        # provided
                        try:
                            self.tmd_protein[leaflet].append(self.universe.select_atoms(query))
                            # cog = np.mean(self.universe.select_atoms(query).positions, axis=0)
                            # self.tmd_protein[leaflet].append(cog)
                        except Exception as e:
                            raise ValueError("Please provide a valid MDAnalysis selection string!") from e
                    else:
                        raise ValueError(
                            "TDM Protein list should contain AtomGroup from MDAnalysis universe or a string "
                            "query for MDAnalysis selection.")
            # self.tmd_protein["0"] = np.array(self.tmd_protein["0"])
            # self.tmd_protein["1"] = np.array(self.tmd_protein["1"])
        elif tmd_protein_list is not None:
            # An unknown argument is provided for tdm_protein_list
            raise ValueError(
                "Please provide tdm_protein_list in list format such as [{'0': upper leaflet related 3 atom, "
                "'1': lower leaflet related 3 atom }, {'0': ..., '1': ...}]. Every dictionary stands for an "
                "individual transmembrane protein.")
        # -----------------------------------------------------------------HMMs--------------------------------------- #

        # Check for user-specified trained HMM
        if not any(trained_hmms):
            # Carry on if there is no trained HMM provided -> Will train HMM(s) later on
            self.trained_hmms = trained_hmms

        else:
            # User-specified trained HMM provided, check for consistency with expected format

            # Check for assymmetric membrane
            if self.asymmetric_membrane:
                # Asymmetric membrane functionality was triggered! Assuming same/different lipid types per leaflet
                # Structure of the expected dictionary: {ResnameA: {0: HMM0A, 1: HMM1A}, ResnameB: {0: HMM0B,
                # 1: HMM1B}, ResnameC: {0: None, 1: HMM1B}, ...}

                # Iterate over each entry and check for validity of input
                for lipid, hmms in zip(trained_hmms.keys(), trained_hmms.values()):

                    # Expected format of object "hmms" right now: {0: HMM0X, 1: HMM1X}

                    assert lipid in self.membrane.residues.resnames, f"{lipid} not found in membrane. Maybe a typo?"

                    # Check if correct number of HMMs per lipid is given
                    assert len(
                        hmms.keys()) == 2, (f'Too many/less HMMs provided for lipid type {lipid}.'
                                            + "\nPlease provide exactly two ({0:...,1:...}) or do not trigger "
                                              "'asymmetric_membrane'!\nIf a lipid is not present in one leaflet, "
                                              "use 'None'.")

                    # Check for correct labelling of leaflets
                    assert (0 in hmms.keys()) and (
                            1 in hmms.keys()), "Provide suitable keys (i.e., 0, 1) for the leaflets!"

                    # Iterate over leaflets
                    for leaflet in range(2):

                        # Expected format of object "hmms[leaflet]" right now: HMMYX

                        # If a lipid is not present in one of the leaflets a None is expected instead of a trained HMM
                        if hmms[leaflet] is None:
                            # If no HMM was provided then this lipid should be not part of this leaflet. Sterols are
                            # expected to flip, therefore always two HMM should be provided for sterol types...
                            assert (lipid not in self.leaflet_selection_no_sterol[
                                str(leaflet)].residues.resnames and lipid not in self.sterol_heads.keys()), \
                                (f"Found lipid {lipid} in leaflet {leaflet}, but no HMM was found!\nPlease provide a "
                                 f"valid HMM for this lipid in this leaflet!\nNote, for sterols always two HMMs are "
                                 f"expected!")

                            # If everything is fine carry on with next leaflet/lipid
                            continue

                        # Check if lipid is in this leaflet or if lipid is a sterol
                        assert lipid in self.leaflet_selection_no_sterol[
                            str(leaflet)].residues.resnames or lipid in self.sterol_heads.keys(), \
                            (f"Could not find lipid/sterol {lipid} in leaflet {leaflet}. Provide only HMMs for lipids "
                             f"that are present in this leaflet!")

                        # If all checks passed, we can assume that "something" should be and is present. Now,
                        # check the validity of the provided object
                        try:
                            # Try to sample something from HMM to check if it is fitted
                            hmms[leaflet].sample(n_samples=1)
                        except Exception as hmmerror:
                            raise ValueError(
                                f"HMM check failed with {hmmerror}! Could not sample a single point from the provided "
                                f"HMM for lipid {lipid} in leaflet {leaflet}. Check your model!")

                # If everthing works until here, it is assumed that all provided HMMs are valid and can be used later on
                self.trained_hmms = trained_hmms

            else:
                # Symmetric membrane is assumed!
                # Assuming same lipid types per leaflet
                # Structure of the expected dictionary: {ResnameA: HMMA, ResnameB: HMMB, ResnameC: HMMC, ...}

                # Iterate over each entry and check for validity of input
                for lipid, hmms in zip(trained_hmms.keys(), trained_hmms.values()):

                    assert lipid in self.membrane.residues.resnames, f"{lipid} not found in membrane. Maybe a typo?"

                    # If all checks passed, we can assume that "something" should be and is present. Now, check
                    # the validity of the provided object
                    try:
                        # Try to sample something from HMM to check if it is fitted
                        hmms.sample(n_samples=1)
                    except sklearn.exceptions.NotFittedError as hmmerror:
                        raise ValueError(
                            f"HMM check failed with {hmmerror}! Could not sample a single point from the provided HMM "
                            f"for lipid {lipid} in leaflet {leaflet}. Check your model!")

                # If everthing works until here, it is assumed that all provided HMMs are valid and can be used later on
                self.trained_hmms = trained_hmms

        # --------------------------------------HOUSE KEEPING-------------------------------------- #

        # Save unique residue names
        _, idx = np.unique(self.membrane.resnames, return_index=True)
        self.unique_resnames = self.membrane.resnames[np.sort(idx)]

        # Get residues ids in upper and lower leaflet -> self.leaflet_resids
        self.residue_ids = self.get_resids()

        # Get dictionary with selection of head- and tailgroups in upper and lower leaflets
        self.resid_tails_selection = self.get_lipid_tails()

        self.sterol_tails_selection = self.get_sterol_tails()

        self.all_heads = self.get_heads()

    def get_leaflets(self):

        """
        Automatically assign non-sterol compounds to the upper and lower leaflet of a bilayer using the
        MDAnalysis.LeafletFinder
        """

        # Call LeafletFinder to get upper and lower leaflets
        leafletfinder = LeafletFinder(self.universe, **self.leaflet_kwargs)

        # Check for two leaflets
        self.n_leaflets = len(leafletfinder.groups())
        assert self.n_leaflets == 2, f"Bilayer is required. {self.n_leaflets} are found."

        # Init empty dict to store AtomGroups -> That would be not necessary, but I want the same variables for the
        # user specified case or the automatic case
        leaflet_selection = {}

        # Iterate over found groups
        for idx, leafgroup in enumerate(leafletfinder.groups_iter()):
            leaflet_selection[str(idx)] = leafgroup

        return leaflet_selection

    def get_leaflets_sterol(self):

        """
        Assign sterol compounds to the upper and lower leaflet of a bilayer using a distance cut-off.
        """

        # Copy dict for leaflet selection without sterols, only the AtomGroups in the copied dict should be updated
        leaflet_selection = self.leaflet_selection_no_sterol.copy()

        # Iterate over each type of sterol in the membrane
        for rsn, head in self.sterol_heads.items():
            sterol = self.universe.select_atoms(f"resname {rsn} and name {head}")
            upper_sterol = distances.distance_array(reference=sterol,
                                                    configuration=self.leaflet_selection_no_sterol['0'],
                                                    box=self.universe.trajectory.ts.dimensions)
            lower_sterol = distances.distance_array(reference=sterol,
                                                    configuration=self.leaflet_selection_no_sterol['1'],
                                                    box=self.universe.trajectory.ts.dimensions)

            # ...determining the minimum distance to each leaflet for each cholesterol,...
            upper_sterol = np.min(upper_sterol, axis=1)
            lower_sterol = np.min(lower_sterol, axis=1)

            # ...the assignment is finished by checking for which leaflet the minimum distance is smallest.
            upper_sterol = sterol[upper_sterol < lower_sterol]
            lower_sterol = sterol.difference(upper_sterol)

            # Merge the atom selections for the phospholipids and cholesterol. "+" just adds the second selection on
            # top of the former one.
            leaflet_selection['0'] += upper_sterol
            leaflet_selection['1'] += lower_sterol

        leaflet_selection['0'] = leaflet_selection['0'][np.argsort(leaflet_selection['0'].resids)]
        leaflet_selection['1'] = leaflet_selection['1'][np.argsort(leaflet_selection['1'].resids)]

        return leaflet_selection

    def get_resids(self):
        """
        Retrieve the residue indices for each residue and store it in a new dictionary.

        Returns
        ---------- 
        residue_ids: dict
            dictionary for each residue containing residue ids of it.
        """

        # Init empty dict to store atom selection of resids
        residue_ids = {}

        # Iterate over found leaflets
        for resname, head in self.heads.items():
            query_str = f"name {head} and resname {resname}"
            residue_ids[resname] = self.universe.select_atoms(query_str).resids
        for resname, head in self.sterol_heads.items():
            query_str = f"name {head} and resname {resname}"
            residue_ids[resname] = self.universe.select_atoms(query_str).resids
        return residue_ids

    def get_heads(self):
        """
        Make an atomgroup containing all head groups (lipids + sterols).

        Returns
        ----------
        all_heads: MDAnalysis.AtomGroup
            AtomGroup containing headgroups of all lipids and sterols
        """

        # Init empty dict to store atom selection of resids
        # TODO Cause => UserWarning: Empty string to select atoms, empty group returned.
        #  warnings.warn("Empty string to select atoms, empty group returned.",
        all_heads = self.universe.select_atoms('')

        # Iterate over found leaflets
        for resname, head in self.heads.items():
            query_str = f"name {head} and resname {resname}"
            all_heads = all_heads | self.universe.select_atoms(query_str)
        for resname, head in self.sterol_heads.items():
            query_str = f"name {head} and resname {resname}"
            all_heads = all_heads | self.universe.select_atoms(query_str)
        return all_heads

    def get_sterol_tails(self):
        """
        Make atomgroups for sterols

        Attributes
        ----------
        sterols_tail: dict
            dictionary containing the tail atomgroups for each sterol

        """

        # Init empty dicts
        sterols_tail = {}

        # Iterate over sterols -> user input
        for sterol, tail in self.sterol_tails.items():
            # Make atom group for sterol tail selection
            tail_sele_str = 'name ' + ' or name '.join(tail)
            tail_sele_str = f'resname {sterol} and ({tail_sele_str})'
            sterols_tail[sterol] = self.universe.select_atoms(tail_sele_str)
        return sterols_tail

    def get_lipid_tails(self):

        """
        Make atomgroups for tailgroups for each chain of residues

        Attributes
        ----------
        self.resid_tails_selection: dict
            dictionary containing the tail atomgroups for all residues each tail
        """
        # Temporary dictionary for query collection of tails
        resid_tails_selection = {}
        tail_select_list = {}
        # Iterate over lipid types
        for resname in self.tails.keys():
            # Make for every type an extra dictionary
            # Iterate over tails (e.g. for standard phospholipids that 2)
            for i, tail in enumerate(self.tails[resname]):
                # Prepare a MDAnalysis selection string (I.e. ['C22', 'H2R'] -> 'name C22 or name H2R')
                tail_sele_str = 'name ' + ' or name '.join(tail)
                # Select for correct lipid type
                # (I.e. I.e. ['C22', 'H2R'] -> 'name C22 or name H2R' -> 'resname POPC and ('name C22 or name H2R')')
                tail_sele_str = f'(resname {resname} and ({tail_sele_str}))'
                tail_select_list.setdefault(i, []).append(tail_sele_str)
        # Create tail selection dictionary for each chain
        for i, query_list in tail_select_list.items():
            query = " or ".join(query_list)
            resid_tails_selection[i] = self.universe.select_atoms(query)
        return resid_tails_selection

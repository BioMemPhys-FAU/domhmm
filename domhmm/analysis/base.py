"""
DomHMM --- :mod:`domhmm.analysis.LeafletAnalysisBase`
===========================================================

This module contains the :class:`LeafletAnalysisBase` class.

"""

# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
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

    Attributes
    ----------
    universe: :class:`~MDAnalysis.core.universe.Universe`
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
    frac: float
        fraction of box length in x and y outside the unit cell considered for Voronoi calculation
    p_value: float
        p_value for z_score calculation
    verbose: bool
        verbose option to print intermediate steps
    leaflet_kwargs: Optional[dict]
        dictionary containing additional arguments for the MDAnalysis LeafletFinder
    heads: Optional[dict]
        dictionary containing resname and atom selection for lipid head groups
    tails: Optional[dict]
        dictionary containing resname and atom selection for lipid tail groups

    """

    def __init__(
            self,
            universe_or_atomgroup: Union["Universe", "AtomGroup"],
            membrane_select: str = "all",
            leaflet_kwargs: Dict[str, Any] = {},
            leaflet_select: Union[None, "AtomGroup", str, list] = None,
            tails: Dict[str, Any] = {},
            heads: Dict[str, Any] = {},
            sterol_heads: Dict[str, Any] = {},
            sterol_tails: Dict[str, Any] = {},
            local: bool = False,
            frac: float = 0.5,
            p_value: float = 0.05,
            leaflet_frame_rate: Union[None, int] = None,
            sterol_frame_rate: int = 1,
            asymmetric_membrane: bool = False,
            verbose: bool = False,
            result_plots: bool = False,
            **kwargs
    ):
        # the below line must be kept to initialize the AnalysisBase class!
        super().__init__(universe_or_atomgroup.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can should used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results
        self.resid_tails_selection = {}
        self.universe = universe_or_atomgroup.universe
        self.membrane = universe_or_atomgroup.select_atoms(membrane_select)
        self.membrane_unique_resids = np.unique(self.membrane.resids)
        self.heads = heads
        self.tails = tails
        self.sterol_heads = sterol_heads
        self.sterol_tails = sterol_tails
        self.leaflet_frame_rate = leaflet_frame_rate
        self.sterol_frame_rate = sterol_frame_rate
        self.frac = frac
        self.p_value = p_value
        self.asymmetric_membrane = asymmetric_membrane
        self.verbose = verbose
        self.result_plots = result_plots

        assert heads.keys() == tails.keys(), "Heads and tails don't contain same residue names"

        self.leaflet_kwargs = leaflet_kwargs

        # -----------------------------------------------------------Local membrane properties------------------------ #
        # If local is True then properties are calculated if possible

        if local:
            self.refZ = False
        else:
            self.refZ = True

        # -----------------------------------------------------------------LEAFLETS----------------------------------- #

        if leaflet_select is None:
            # No information about leaflet assignment is provided. Raise an error and exit.
            raise ValueError("No leaflet assigned! Please provide a list containing either two MDAnalysis.AtomGroup objects, two valid MDAnalysis selection strings, or 'auto' to trigger automatic leaflet assignment.")

        elif isinstance(leaflet_select, list):
            # If the argument leaflet_select is a list, the user specified their own leaflet selection.
            # The code checks if there are exactly two groups and raises an assertion if not.
            # It then stores the provided AtomGroup in a dictionary, or creates the AtomGroup with a provided selection string and stores it in the dictionary.

            assert len(leaflet_select) == 2, (f"A bilayer is required. {len(leaflet_select)} entries found in leaflet_select list.")

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

                    #TODO: Check for atom number in upper and lower leaflet and raise a Warning
                else:
                    raise ValueError("Please provide an MDAnalysis.AtomGroup or a valid MDAnalysis selection string!")

                #Iterate over sterol compounds and check if it is part of the phospholipid leaflet assignment
                for rsn, atoms in self.sterol_heads.items():
                    if rsn in self.leaflet_selection_no_sterol[str(i)].residues.resnames:
                        raise ValueError(f"Sterol {rsn} should not be part of the initial leaflet identification! Sterols will be assigned automatically.")
                    else: pass

        elif leaflet_select.lower() == "auto":
            # 'auto' should trigger an automated leaflet assignment pipeline (e.g., LeafletFinder provided by MDAnalysis)
            self.leaflet_selection_no_sterol = self.get_leaflets()

        else:
            # An unknown argument is provided for leaflet_select
            raise ValueError("No leaflet assigned! Please provide a list containing either two MDAnalysis.AtomGroup objects, two valid MDAnalysis selection strings, or 'auto' to trigger automatic leaflet assignment.")

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
        Automatically assign non-sterol compounds to the upper and lower leaflet of a bilayer using the MDAnalysis.LeafletFinder
        """

        # Call LeafletFinder to get upper and lower leaflets
        leafletfinder = LeafletFinder(self.universe, **self.leaflet_kwargs)

        # Check for two leaflets
        self.n_leaflets = len(leafletfinder.groups())
        assert self.n_leaflets == 2, f"Bilayer is required. {self.n_leaflets} are found."

        # Init empty dict to store AtomGroups -> That would be not necessary but I want the same variables for the
        # user specified case or the automatic case
        leaflet_selection = {}

        # Iterate over found groups
        for idx, leafgroup in enumerate(leafletfinder.groups_iter()):
            leaflet_selection[str(idx)] = leafgroup

        return leaflet_selection

    def get_leaflets_sterol(self):

        """
        Assign sterol compounds to to the upper and lower leaflet of a bilayer using a distance cut-off.
        """

        # Copy dict for leaflet selection without sterols, only the AtomGroups in the copied dict should be updated
        leaflet_selection = {}

        #Iterate over each type of sterol in the membrane
        for rsn, head in self.sterol_heads.items():
            # TODO Find more user-friendly way for sterol atom selection
            sterol = self.universe.select_atoms(f"resname {rsn} and name {head}")
            upper_sterol = distances.distance_array(reference=sterol, configuration=self.leaflet_selection_no_sterol['0'],
                                                    box=self.universe.trajectory.ts.dimensions)
            lower_sterol = distances.distance_array(reference=sterol, configuration=self.leaflet_selection_no_sterol['1'],
                                                    box=self.universe.trajectory.ts.dimensions)

            # ...determining the minimum distance to each leaflet for each cholesterol,...
            upper_sterol = np.min(upper_sterol, axis=1)
            lower_sterol = np.min(lower_sterol, axis=1)

            # ...the assignment is finished by checking for which leaflet the minimum distance is smallest.
            upper_sterol = sterol[upper_sterol < lower_sterol]
            lower_sterol = sterol.difference(upper_sterol)

            # Merge the atom selections for the phospholipids and cholesterol. "+" just adds the second selection on top of the former one.
            leaflet_selection['0'] = self.leaflet_selection_no_sterol['0'] + upper_sterol
            leaflet_selection['1'] = self.leaflet_selection_no_sterol['1'] + lower_sterol

        return leaflet_selection

    def get_resids(self):
        """
        Retrieve the residue indices for each residue and store it in a new dictionary.

        Attributes
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
            # TODO Find more user-friendly way for sterol atom selection
            query_str = f"name {head} and resname {resname}"
            residue_ids[resname] = self.universe.select_atoms(query_str).resids
        return residue_ids

    def get_heads(self):
        """
        Make an atomgroup containing all head groups (lipids + sterols).

        Attributes
        ----------
        all_heads: MDAnalysis.AtomGroup
            AtomGroup containing headgroups of all lipids and sterols
        """

        # Init empty dict to store atom selection of resids
        all_heads = self.universe.select_atoms('')

        # Iterate over found leaflets
        for resname, head in self.heads.items():
            query_str = f"name {head} and resname {resname}"
            all_heads = all_heads | self.universe.select_atoms(query_str)
        for resname, head in self.sterol_heads.items():
            # TODO Find more user-friendly way for sterol atom selection
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
                # Select for correct lipid type (I.e. I.e. ['C22', 'H2R'] -> 'name C22 or name H2R' ->
                # 'resname POPC and ('name C22 or name H2R')')
                tail_sele_str = f'(resname {resname} and ({tail_sele_str}))'
                tail_select_list.setdefault(i, []).append(tail_sele_str)
        # Create tail selection dictionary for each chain
        for i, query_list in tail_select_list.items():
            query = " or ".join(query_list)
            resid_tails_selection[i] = self.universe.select_atoms(query)
        return resid_tails_selection

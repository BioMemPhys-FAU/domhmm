"""
Elbe --- :mod:`elbe.analysis.Elbe`
===========================================================

This module contains the :class:`Elbe` class.

"""

# ----MDANALYSIS---- #
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.leaflet import LeafletFinder

# ----PYTHON---- #
from typing import Union, TYPE_CHECKING, Dict, Any
import numpy as np

if TYPE_CHECKING:
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
            leaflet_select: Union[None, str] = None,
            tails: Dict[str, Any] = {},
            heads: Dict[str, Any] = {},
            sterols: Dict[str, Any] = {},
            local: bool = False,
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
        self.sterols = sterols

        assert heads.keys() == tails.keys(), "Heads and tails don't contain same residue names"
        
        self.leaflet_kwargs = leaflet_kwargs

        # -----------------------------------------------------------Local membrane properties------------------------ #
        # If local is True then properties are calculated if possible

        if local:
            self.refZ = False
        else:
            self.refZ = True

        # -----------------------------------------------------------------LEAFLETS----------------------------------- #

        if leaflet_select is not None:

            # If the argument leaflet_select is not None, the user specified its own leaflet selection. The code just
            # checks if there are exactly two groups and raises an assertion if not. It then stores the selection
            # in a dictionary

            assert len(leaflet_select) == 2, (f"Bilayer is required. {len(leaflet_select)} are found in leaflet_select "
                                              f"list.")

            # Init empty dict to store AtomGroups
            self.leaflet_selection = {}

            # Iterate over both entries
            for i in range(2):
                self.leaflet_selection[str(i)] = self.universe.select_atoms(leaflet_select[i])

        else:
            self.get_leaflets()
            

        # Save unique residue names
        _, idx = np.unique(self.membrane.resnames, return_index=True)
        self.unique_resnames = self.membrane.resnames[np.sort(idx)]

        # Get residues ids in upper and lower leaflet -> self.leaflet_resids
        self.get_resids()

        # Get dictionary with selection of head- and tailgroups in upper and lower leaflets
        self.get_leaflet_tails()

        self.get_leaflet_sterols()
    
    def get_leaflets(self):
        # Call LeafletFinder to get upper and lower leaflets
        leafletfinder = LeafletFinder(self.universe, **self.leaflet_kwargs)

        # Check for two leaflets
        self.n_leaflets = len(leafletfinder.groups())
        assert self.n_leaflets == 2, f"Bilayer is required. {self.n_leaflets} are found."

        # Init empty dict to store AtomGroups -> That would be not necessary but I want the same variables for the
        # user specified case or the automatic case
        self.leaflet_selection = {}

        # Iterate over found groups
        for idx, leafgroup in enumerate(leafletfinder.groups_iter()):
            self.leaflet_selection[str(idx)] = leafgroup

        for rsn, atoms in self.sterols.items():
            # TODO Find more user-friendly way for sterol atom selection
            sterol = self.universe.select_atoms(f"name {atoms[0]}")
            upper_sterol = distances.distance_array(reference=sterol, configuration=self.leaflet_selection['0'],
                                                    box=self.universe.trajectory.ts.dimensions)
            lower_sterol = distances.distance_array(reference=sterol, configuration=self.leaflet_selection['1'],
                                                    box=self.universe.trajectory.ts.dimensions)

            # ...determining the minimum distance to each leaflet for each cholesterol,...
            upper_sterol = np.min(upper_sterol, axis=1)
            lower_sterol = np.min(lower_sterol, axis=1)

            # ...the assignment is finished by checking for which leaflet the minimum distance is smallest.
            upper_sterol = sterol[upper_sterol < lower_sterol]
            lower_sterol = sterol.difference(upper_sterol)

            # Merge the atom selections for the phospholipids and cholesterol
            self.leaflet_selection['0'] = self.leaflet_selection['0'] + upper_sterol
            self.leaflet_selection['1'] = self.leaflet_selection['1'] + lower_sterol

    def get_resids(self):
        """
        Retrieve the residue indices for each residue and store it in a new dictionary.

        Attributes
        ---------- 
        residue_ids: dict
            dictionary for each residue containing residue ids of it.
        """

        # Init empty dict to store atom selection of resids
        self.residue_ids = {}

        # Iterate over found leaflets
        for resname, head in self.heads.items():
            query_str = f"name {head} and resname {resname}"
            self.residue_ids[resname] = self.universe.select_atoms(query_str).resids
        for resname, atoms in self.sterols.items():
            # TODO Find more user-friendly way for sterol atom selection
            query_str = f"name {atoms[0]} and resname {resname}"
            self.residue_ids[resname] = self.universe.select_atoms(query_str).resids
                
    def get_leaflet_sterols(self):
        """
        Make atomgroups for sterols

        Attributes
        ----------
        sterols_tail: dict
            dictionary containing the tail atomgroups for each sterol

        """

        # Init empty dicts
        self.sterols_tail = {}

        # Iterate over sterols -> user input
        for sterol, tail in self.sterols.items():
            # Make atom group for sterol tail selection
            tail_sele_str = 'name ' + ' or name '.join(tail)
            tail_sele_str = f'resname {sterol} and ({tail_sele_str})'
            self.sterols_tail[sterol] = self.universe.select_atoms(tail_sele_str)

    def get_leaflet_tails(self):

        """
        Make atomgroups for tailgroups for each chain of residues

        Attributes
        ----------
        self.resid_tails_selection: dict
            dictionary containing the tail atomgroups for all residues each tail
        """
        # Temporary dictionary for query collection of tails
        tail_select_list = {}
        # Iterate over lipid types
        for resname in self.tails.keys():
            # Make for every type an extra dictionary
            # Iterate over tails (e.g. for standard phospholipids that 2)
            for i, tail in enumerate(self.tails[resname]):
                # Check for correct input
                assert len(tail) % 2 == 0, 'Error: Selection list for tails must be even'
                # Prepare a MDAnalysis selection string (I.e. ['C22', 'H2R'] -> 'name C22 or name H2R')
                tail_sele_str = 'name ' + ' or name '.join(tail)
                # Select for correct lipid type (I.e. I.e. ['C22', 'H2R'] -> 'name C22 or name H2R' ->
                # 'resname POPC and ('name C22 or name H2R')')
                tail_sele_str = f'(resname {resname} and ({tail_sele_str}))'
                tail_select_list.setdefault(i, []).append(tail_sele_str)
        # Create tail selection dictionary for each chain
        for i, query_list in tail_select_list.items():
            query = " or ".join(query_list)
            self.resid_tails_selection[i] = self.universe.select_atoms(query)

"""
Elbe --- :mod:`elbe.analysis.Elbe`
===========================================================

This module contains the :class:`Elbe` class.

"""

# ----MDANALYSIS---- #
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.leaflet import LeafletFinder

# ----PYTHON---- #
from typing import Union, TYPE_CHECKING, Dict, Any
import numpy as np

from tqdm import tqdm

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
            heads: Dict[str, Any] = {},
            tails: Dict[str, Any] = {},
            sterols: list = [],
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

        self.universe = universe_or_atomgroup.universe
        self.membrane = universe_or_atomgroup.select_atoms(membrane_select)
        self.membrane_unique_resids = np.unique(self.membrane.resids)

        self.heads = heads
        self.tails = tails
        self.sterols = sterols

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

            # Call LeafletFinder to get upper and lower leaflets
            self.leafletfinder = LeafletFinder(self.universe, **leaflet_kwargs)

            # Check for two leaflets
            self.n_leaflets = len(self.leafletfinder.groups())
            assert self.n_leaflets == 2, f"Bilayer is required. {self.n_leaflets} are found."

            # Init empty dict to store AtomGroups -> That would be not necessary but I want the same variables for the
            # user specified case or the automatic case
            self.leaflet_selection = {}

            # Iterate over found groups
            for idx, leafgroup in enumerate(self.leafletfinder.groups_iter()):
                self.leaflet_selection[str(idx)] = leafgroup

        # Membrane selection -> is later used for center of mass calculation
        self.memsele = self.universe.select_atoms(membrane_select)

        # Get residues ids in upper and lower leaflet -> self.leaflet_resids
        self.get_leaflet_resids()

        # Get dictionary with selection of head- and tailgroups in upper and lower leaflets
        self.get_leaflet_heads_tails()

        self.get_leaflet_sterols()

    def get_leaflet_resids(self):

        """
        Retrieve the residue indices from the leaflet finder groups and store it in a new dictionary.

        Attributes
        ---------- 
        leaflet_resids: dict
            dictionary for each leaflet (should be two) containing a selection of residues. -> Is used by
            get_leaflet_heads() and get_leaflet_tails()

        """

        # Init empty dict to store atom selection of resids
        self.leaflet_resids = {}

        # Iterate over found leaflets
        for idx, leafgroup in zip(self.leaflet_selection.keys(), self.leaflet_selection.values()):

            # Init empty selection
            self.leaflet_resids[idx] = self.universe.select_atoms("")

            # Get unique resids in the leaflet
            uni_leaf_resids = np.unique(leafgroup.resids)

            # Iterate over residues in found group and add it to the atomgroup
            for resid in uni_leaf_resids:
                self.leaflet_resids[idx] += self.universe.select_atoms(f"resid {resid}")

    def get_leaflet_sterols(self):
        """
        Make atomgroups for sterols

        Attributes
        ---------- 
        sterols_head: dict
            dictionary containing the head atomgroups for each sterol
        sterols_tail: dict
            dictionary containing the tail atomgroups for each sterol

        """

        # Init empty dicts
        self.sterols_head = {}
        self.sterols_tail = {}

        # Iterate over sterols -> user input
        for sterol in self.sterols:
            # Make atom group for sterol head selection
            head_sele_str = 'name ' + ' or name '.join(self.heads[sterol])
            head_sele_str = f'resname {sterol} and ({head_sele_str})'
            self.sterols_head[sterol] = self.universe.select_atoms(head_sele_str)

            # Make atom group for sterol tail selection
            tail_sele_str = 'name ' + ' or name '.join(self.tails[sterol][0])
            tail_sele_str = f'resname {sterol} and ({tail_sele_str})'
            self.sterols_tail[sterol] = self.universe.select_atoms(tail_sele_str)

    def get_leaflet_heads_tails(self):

        """
        Make atomgroups for headgroups in upper and lower leaflet
        Make atomgroups for tailgroups in upper and lower leaflet

        Attributes
        ---------- 
        self.resid_heads_selection_0: dict
            dictionary containing the head atomgroups for each residue in the upper leaflet 
        self.resid_heads_selection_1: dict
            dictionary containing the head atomgroups for each residue in the lower leaflet 
        self.resid_tails_selection_0: dict
            dictionary containing the tail atomgroups for each residue in the upper leaflet 
        self.resid_tails_selection_1: dict
            dictionary containing the tail atomgroups for each residue in the lower leaflet 

        """

        # Init empty dict
        self.resid_heads_selection_0 = {}
        self.resid_heads_selection_1 = {}

        # Init empty dict
        self.resid_tails_selection_0 = {}
        self.resid_tails_selection_1 = {}

        # Select the groups for heads and tails once and not for every single resid
        head_selection_per_type = {}
        tails_selection_per_type = {}

        # Iterate over lipid types
        for resname in np.unique(self.memsele.resnames):

            # ------------------------------------------------------HEAD SELECTION------------------------------------ #

            # Prepare a MDAnalysis selection string (I.e. ['P', 'O11', 'O12', 'O13', 'O14'] ->
            # 'name P or name O11 or name O12 or name O13 or name O14')
            head_sele_str = 'name ' + ' or name '.join(self.heads[resname])

            # Select for correct lipid type (I.e. 'name P or name O11 or name O12 or name O13 or name O14' ->
            # 'resname POPC and (name P or name O11 or name O12 or name O13 or name O14)')
            head_sele_str = f'resname {resname} and ({head_sele_str})'

            # Use MDAnalysis select_atoms to make selection group for heads
            head_selection = self.universe.select_atoms(head_sele_str)

            # Check for correctness
            assert head_selection.n_atoms > 0, (f"!!!-----ERROR-----!!!\nSelection for head group {head_sele_str} is "
                                                f"empty.\n!!!-----ERROR-----!!!")

            # Store atom selection in dictionary
            head_selection_per_type[resname] = head_selection

            # ------------------------------------------------------TAIL SELECTION------------------------------------ #

            # Make for every type an extra dictionary -> Tails suck because there so many of them per lipid (at least 2)
            tails_selection_per_type[resname] = {}

            """
            Final structure should look like that

            - tails_selection_per_type (dict)
                - LipidA (dict)
                    - Chain0 (list)
                        - Atom selection Pair 0
                        - Atom selection Pair 1
                        -...
                    - Chain1 (list)
                        - Atom selection Pair 0
                        - Atom selection Pair 1
                        -...
                - LipidB (dict)
                    - ...

            """

            # Iterate over tails (e.g. for standard phospholipids that 2)
            for i, tail in enumerate(self.tails[resname]):

                # Check for correct input
                assert len(
                    tail) % 2 == 0, ('!!!-----ERROR-----!!!\nSelection list for tails must be even\n'
                                     '!!!-----ERROR-----!!!')

                n_pairs = len(tail) // 2  # Number of pairs in one tail

                # Init empty list for every CHAIN
                tails_selection_per_type[resname][str(i)] = []

                # Iterate over all pairs and make extra selections
                s, e = 0, 2
                for j in range(n_pairs):
                    # Prepare a MDAnalysis selection string (I.e. ['C22', 'H2R'] -> 'name C22 or name H2R')
                    tail_sele_str = 'name ' + ' or name '.join(tail[s:e])
                    s, e = e, e + 2  # Increase counter for next iteration -> 0,2 -> 2,4 -> 4,6 ...

                    # Select for correct lipid type (I.e. I.e. ['C22', 'H2R'] -> 'name C22 or name H2R' ->
                    # 'resname POPC and ('name C22 or name H2R')')
                    tail_sele_str = f'resname {resname} and ({tail_sele_str})'

                    # Use MDAnalysis select_atoms to make selection group for tails
                    tail_selection = self.universe.select_atoms(tail_sele_str)

                    # Check for correctness
                    assert tail_selection.n_atoms > 0, (f"!!!-----ERROR-----!!!\n"
                                                        f"Selection for tail group {tail_sele_str} is empty.\n"
                                                        f"!!!-----ERROR-----!!!")
                    assert tail_selection.n_atoms % 2 == 0, (f"!!!-----ERROR-----!!!\n"
                                                             f"Selection for tail group {tail_sele_str} is not "
                                                             f"dividable by 2.\n!!!-----ERROR-----!!!")

                    tails_selection_per_type[resname][str(i)].append(tail_selection)

        # Iterate over leaflets
        for idx, leafgroup in zip(self.leaflet_selection.keys(), self.leaflet_selection.values()):

            leaf_resids = np.unique(self.leaflet_resids[idx].resids)

            # Iterate over resids in leaflet
            for resid in tqdm(leaf_resids):

                # Select specific resid
                resid_selection = self.universe.select_atoms(f'resid {resid}')
                # Get the lipid type of the specific resid
                resid_resname = resid_selection.resnames[0]

                # -----------------------------------------------------HEAD SELECTION--------------------------------- #

                # Assign selected head group to upper or lower leaflet
                if idx == "0":
                    self.resid_heads_selection_0[str(resid)] = resid_selection.intersection(
                        head_selection_per_type[resid_resname])
                else:
                    self.resid_heads_selection_1[str(resid)] = resid_selection.intersection(
                        head_selection_per_type[resid_resname])

                # -----------------------------------------------------TAIL SELECTION--------------------------------- #
                # Make an extra dictionary for each tail
                if idx == "0":
                    self.resid_tails_selection_0[str(resid)] = {}
                else:
                    self.resid_tails_selection_1[str(resid)] = {}

                # Iterate over tails (e.g. for standard phospholipids that 2)
                for i, tail in enumerate(self.tails[resid_resname]):

                    n_pairs = len(tail) // 2  # Number of pairs in one tail

                    # Init empty list for each pair
                    if idx == "0":
                        self.resid_tails_selection_0[str(resid)][str(i)] = []
                    else:
                        self.resid_tails_selection_1[str(resid)][str(i)] = []

                    # Iterate over all pairs and make extra selections
                    s, e = 0, 2
                    for j in range(n_pairs):

                        # Use MDAnalysis select_atoms to make selection group for tails
                        tail_selection = self.universe.select_atoms(tail_sele_str)

                        if idx == "0":
                            self.resid_tails_selection_0[str(resid)][str(i)].append(
                                resid_selection.intersection(tails_selection_per_type[resid_resname][str(i)][j]))
                        else:
                            self.resid_tails_selection_1[str(resid)][str(i)].append(
                                resid_selection.intersection(tails_selection_per_type[resid_resname][str(i)][j]))

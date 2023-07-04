"""
Elbe --- :mod:`elbe.analysis.Elbe`
===========================================================

This module contains the :class:`Elbe` class.

"""

#----MDANALYSIS----#
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.leaflet import LeafletFinder

#----PYTHON----#
from typing import Union, TYPE_CHECKING, Dict, Any
import numpy as np


if TYPE_CHECKING:
    from MDAnalysis.core.universe import Universe, AtomGroup

class LeafletAnalysisBase(AnalysisBase):
    """
    LeafletAnalysisBase class.

    This class marks the start point for each analysis method. 

    It connects to the MDAnalysis core library, setups the MDAnalysis universe, and provides coordinates for each membrane leaflet. 

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
        gm_kwargs: Dict[str, Any] = {},
        heads: Dict[str, Any] = {},
        tails: Dict[str, Any] = {},
        sterols: list = [],
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

        self.gm_kwargs = gm_kwargs

        #------------------------------------------------------------------LEAFLETS------------------------------------------------------------------#
 
        #Call LeafletFinder to get upper and lower leaflets
        self.leafletfinder = LeafletFinder(self.universe, **leaflet_kwargs)

        #Check for two leaflets
        self.n_leaflets = len(self.leafletfinder.groups())
        assert self.n_leaflets == 2, f"Bilayer is required. {self.n_leaflets} are found."

        #Membrane selection -> is later used for center of mass calculation
        self.memsele = self.universe.select_atoms(membrane_select)

        #Get residues ids in upper and lower leaflet -> self.leaflet_resids
        self.get_leaflet_resids()
        
        #Get dictionary with selection of headgroups in upper and lower leaflet -> self.leaflet_heads
        self.get_leaflet_heads()
        
        #Get dictionary with selection of tailgroups in upper and lower leaflet -> self.leaflet_tails
        #self.get_leaflet_tails()

        self.get_leaflet_sterols()

        self.get_leaflet_tails()


    def get_leaflet_resids(self):

        """
        Retrieve the residue indices from the leaflet finder groups and store it in a new dictionary.

        Attributes
        ---------- 
        leaflet_resids: dict
            dictionary for each leaflet (should be two) containing a selection of residues. -> Is used by get_leaflet_heads() and get_leaflet_tails()

        """

        #Init empty dict to store atom selection of resids
        self.leaflet_resids = {}

        #Iterate over found leaflets
        for idx, leafgroup in enumerate(self.leafletfinder.groups_iter()):

            #Init empty selection 
            self.leaflet_resids[f"{idx}"] = self.universe.select_atoms("")

            #Get unique resids in the leaflet
            uni_leaf_resids = np.unique(leafgroup.resids)

            #Iterate over residues in found group and add it to the atomgroup
            for resid in uni_leaf_resids: self.leaflet_resids[f"{idx}"] += self.universe.select_atoms(f"resid {resid}")

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

        #Init empty dicts
        self.sterols_head = {}
        self.sterols_tail = {}

        #Iterate over sterols -> user input
        for sterol in self.sterols:

            #Make atom group for sterol head selection
            head_sele_str = 'name ' + (' or name ').join(self.heads[sterol])
            head_sele_str = f'resname {sterol} and ({head_sele_str})'
            self.sterols_head[sterol] = self.universe.select_atoms(head_sele_str)

            #Make atom group for sterol tail selection
            tail_sele_str = 'name ' + (' or name ').join(self.tails[sterol][0])
            tail_sele_str = f'resname {sterol} and ({tail_sele_str})'
            self.sterols_tail[sterol] = self.universe.select_atoms(tail_sele_str)

    def get_leaflet_heads(self):

        """
        Make atomgroups for headgroups in upper and lower leaflet

        Attributes
        ---------- 
        leaflet_heads: dict
            dictionary containing the head atomgroups for each residue in the upper and lower leaflet 

        """

        #Init empty dict
        self.leaflet_heads = {}

        #Iterate over leaflets
        for idx, leafgroup in enumerate(self.leafletfinder.groups_iter()):

            leaf_names = np.unique(leafgroup.resnames)

            #Init empty dict for leaflet group
            self.leaflet_heads[f"{idx}"] = {}

            #Iterate over dictionary with atoms for head groups -> This comes from user
            for key, val in zip(self.heads.keys(), self.heads.values()):

                #If the lipid type is not in the leaflet continue
                if key not in leaf_names: continue

                #Prepare a MDAnalysis selection string
                head_sele_str = 'name ' + (' or name ').join(val)
                head_sele_str = f'resname {key} and ({head_sele_str})'

                #Call MDAnalysis to make an atomgroup
                head_selection = self.universe.select_atoms(head_sele_str)

                assert head_selection.n_atoms > 0, "!!!-----ERROR-----!!!\nSelection for head group {head_sele_str} is empty\n!!!-----ERROR-----!!!"

                #Add selected group to leaflet dictionary according to lipid type. Use .intersection() to select residues from current leaflet!
                self.leaflet_heads[f"{idx}"][key] = head_selection.intersection(self.leaflet_resids[f"{idx}"])

    def get_leaflet_tails(self):

        """
        Make atomgroups for tailgroups in upper and lower leaflet

        Attributes
        ---------- 
        leaflet_tails: dict
            dictionary containing the tail atomgroups for each residue in the upper and lower leaflet 

        """

        #Init empty dict
        self.resid_selection_0 = {}
        self.resid_selection_1 = {}

        #Iterate over leaflets
        for idx, leafgroup in enumerate(self.leafletfinder.groups_iter()):

            leaf_names = np.unique(leafgroup.resnames)
            leaf_resids = np.unique(leafgroup.resids)

            for resid in self.membrane_unique_resids:

                if resid not in leaf_resids: continue

                resid_selection = self.universe.select_atoms(f'resid {resid}')
                resid_resname = np.unique(resid_selection.resnames)[0]

                if idx == 0: self.resid_selection_0[str(resid)] = {}
                else: self.resid_selection_1[str(resid)] = {}

                for i, tail in enumerate(self.tails[resid_resname]):

                    n_pairs = len(tail) // 2 #Number of pairs in one tail

                    assert len(tail) % 2 == 0, '!!!-----ERROR-----!!!\nSelection list for tails must be even\n!!!-----ERROR-----!!!' 
                
                    if idx == 0: self.resid_selection_0[str(resid)][str(i)] = []
                    else: self.resid_selection_1[str(resid)][str(i)] = []

                    s,e=0,2
                    for j in range( n_pairs ):

                        #Prepare a MDAnalysis selection string (I.e. ['C22', 'H2R'] -> 'name C22 or name H2R')
                        tail_sele_str = 'name ' + (' or name ').join(tail[s:e])
                        s,e=e, e+2 #Increase counter for next iteration
                        
                        #Select for correct lipid type (I.e. I.e. ['C22', 'H2R'] -> 'name C22 or name H2R' -> 'resname POPC and resid 78 and ('name C22 or name H2R')')
                        tail_sele_str = f'resname {resid_resname} and resid {resid} and ({tail_sele_str})'
        
                        #Use MDAnalysis select_atoms to make selection group for tails
                        tail_selection = self.universe.select_atoms(tail_sele_str)
                        
                        #Check for correctness
                        assert tail_selection.n_atoms > 0,  f"!!!-----ERROR-----!!!\nSelection for tail group {tail_sele_str} is empty.\n!!!-----ERROR-----!!!"
                        assert tail_selection.n_atoms % 2 == 0, f"!!!-----ERROR-----!!!\nSelection for tail group {tail_sele_str} is not dividable by 2.\n!!!-----ERROR-----!!!"

                        if idx == 0: self.resid_selection_0[str(resid)][str(i)].append(tail_selection)
                        else: self.resid_selection_1[str(resid)][str(i)].append(tail_selection)





"""
LocalFluctuation --- :mod:`elbe.analysis.LocalFluctuation`
===========================================================

This module contains the :class:`LocalFluctuation` class.

"""

from .base import LeafletAnalysisBase
from typing import Union, TYPE_CHECKING, Dict, Any

import numpy as np
from MDAnalysis.analysis import distances
from sklearn import mixture
from scipy import stats
import sys

class DirectorOrder(LeafletAnalysisBase):
    """
    The DirectorOrder class calculates the P2 order parameter for each selected lipid according to the forumla:

        P2 = 0.5 * (3 * cos(a)^2 - 1), (1)

    where a is the angle between the lipid director and the membrane normal.

    """

    def _prepare(self):
        """
        Prepare results array before the calculation of the height spectrum begins

        Attributes
        ----------
        """

        #Altough sterols maybe do not play a larger role in the future for the domain identification it seems to be a good idea to keep this functionality
        self.resid_selection_sterols = {}
        self.resid_selection_sterols_heads = {}

        #Next, a dictionary for EACH selected resid will be created. That's pretty much, but it is important to have the order parameters for each lipid over the whole
        #trajectory for the domain identification
        
        #Iterate over all residues in the selected membrane
        for resid in self.membrane_unique_resids:

            #Select specific resid
            resid_selection = self.universe.select_atoms(f"resid {resid}")
            #Get its lipid type
            resname = np.unique(resid_selection.resnames)[0]

            #Check leaflet assignment -> based on RESID
            #LEAFLET 0?
            if resid in self.leafletfinder.group(0).resids and resid not in self.leafletfinder.group(1).resids:

                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})                     #-> Setup an empty dictionary
                getattr(self.results, f'id{resid}')['Leaflet'] = 0          #-> Store information about leaflet assignment
                getattr(self.results, f'id{resid}')['Resname'] = resname    #-> Store lipid type

                #Iterate over leaflet tails
                n_tails = len(self.tails[resname])
                for i in range(n_tails):

                    #n_pairs -> Number of C-H pairs
                    #Should be an even number -> Check for this
                    assert len(self.tails[resname][i]) % 2 == 0, f"Tail selection for {rsn} in chain {i} must be divisable by 2!"
                    n_pairs = len(self.tails[resname][i]) // 2

                    #Init storage for P2 values for each lipid
                    getattr(self.results, f'id{resid}')[f'P2_{i}'] = np.zeros( (self.n_frames, n_pairs ), dtype = np.float32)

                #Store 3-D position of head group for each lipid
                getattr(self.results, f'id{resid}')[f'Heads'] = np.zeros( (self.n_frames, 3), dtype = np.float32) 

            #LEAFLET 1?
            elif resid in self.leafletfinder.group(1).resids and resid not in self.leafletfinder.group(0).resids:
                
                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})                     #-> Setup an empty dictionary
                getattr(self.results, f'id{resid}')['Leaflet'] = 1          #-> Store information about leaflet assignment
                getattr(self.results, f'id{resid}')['Resname'] = resname    #-> Store lipid type

                #Iterate over leaflet tails
                n_tails = len(self.tails[resname])
                for i in range(n_tails):

                    #n_pairs -> Number of C-H pairs
                    #Should be an even number -> Check for this
                    assert len(self.tails[resname][i]) % 2 == 0, f"Tail selection for {rsn} in chain {i} must be divisable by 2!"
                    n_pairs = len(self.tails[resname][i]) // 2

                    #Init storage for P2 values for each lipid
                    getattr(self.results, f'id{resid}')[f'P2_{i}'] = np.zeros( (self.n_frames, n_pairs ), dtype = np.float32)

                #Store 3-D position of head group for each lipid
                getattr(self.results, f'id{resid}')[f'Heads'] = np.zeros( (self.n_frames, 3), dtype = np.float32) 

            #STEROL?
            elif resid not in self.leafletfinder.group(0).resids and resid not in self.leafletfinder.group(1).resids and resname in self.sterols:

                #Sterols are not assigned to a specific leaflet -> They can flip. Maybe it is unlikely that it happens in some membrane (especially atomistic ones)
                #but it can happen and the code keeps track of them.

                #Make a MDAnalysis atom selection for each resid. For the other lipids this was already done in the LeafletAnalysisBase class
                self.resid_selection_sterols[str(resid)] = resid_selection.intersection(self.sterols_tail[resname])
                self.resid_selection_sterols_heads[str(resid)] = resid_selection.intersection(self.sterols_head[resname])
                
                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})
                #Init storage array for leaflet assignment
                getattr(self.results, f'id{resid}')['Leaflet'] = np.zeros( (self.n_frames), dtype = np.float32)
                #Resname of the sterol compound
                getattr(self.results, f'id{resid}')['Resname'] = resname
                #For sterol only one P2 value is calculated but for each frame
                getattr(self.results, f'id{resid}')['P2_0'] = np.zeros( (self.n_frames), dtype = np.float32)
                #Init storage array for head group position for each sterol
                getattr(self.results, f'id{resid}')[f'Heads'] = np.zeros( (self.n_frames, 3), dtype = np.float32) 

            #NOTHING?
            else: raise ValueError(f'{resname} with resid {resid} not found in leaflets or sterol list!')


    def calc_p2(self, pair):

        """
        Calculates the deuterium order parameter according to equation (1) for each pair.

        Parameters
        ----------
        pair: MDAnalysis atom selection
            selection group containing the two atoms for the director calculation
        """

        r = pair.positions[0] - pair.positions[1]
        r /= np.sqrt(np.sum(r**2))

        #Dot product between membrane normal (z axis) and orientation vector
        a = np.arccos(r[2]) #Angle in radians
        P2 = 0.5 * (3 * np.cos(a)**2 - 1)

        return P2

    def get_p2_per_lipid(self, resid_selection_leaflet, leaflet, resid_heads_selection_leaflet):

        """
        Applies P2 calculation for each C-H pair in an individual lipid for each leaflet.

        Parameters
        ----------
        resid_selection_leaflet : dictionary
            Contains resids for a specific leaflet
        leaflet : int
            Leaflet of interest
        resid_heads_selection_leaflet : dictionary
            Contains MDAnalysis atom selection for head group of individual lipids per leaflet


        """

        #Iterate over resids in leaflet
        for key in resid_selection_leaflet.keys():

            #Check if leaflet is correct -> Sanity check
            assert getattr(self.results, f'id{key}')['Leaflet'] == leaflet, '!!!-----ERROR-----!!!\nWrong leaflet\n!!!-----ERROR-----!!!'

            #Store head position -> Center of Mass of head group selection
            getattr(self.results, f'id{key}')[f'Heads'][self.index] = resid_heads_selection_leaflet[key].center_of_mass()

            #Get resname
            rsn = getattr(self.results, f'id{key}')['Resname']
            
            #Iterate over number of acyl chains in lipid named "rsn"
            for n_chain in range( len(self.tails[rsn]) ):

                #self.tails[rsn][n_chain] contains atoms names in tail, if the input is correctly given it should look like this:
                #I.E. -> ["C1", "H1R", "C1", "H1S", ...]

                #Iterate over these pairs -> I.E. ("C1","H1R"), ("C1", "H1S"), ... -> In this order the P2 values should be also stored
                for i in range(len(self.tails[rsn][n_chain]) // 2):

                    getattr(self.results, f'id{key}')[f'P2_{n_chain}'][self.index, i] = self.calc_p2(pair = resid_selection_leaflet[str(key)][ str(n_chain) ][ i ])

    def _single_frame(self):
        """
        Calculate data from a single frame of the trajectory.
        """

        #Get number of frame from trajectory
        self.frame = self.universe.trajectory.ts.frame
        #Calculate correct index if skipping step not equals 1 or start point not equals 0
        self.index = self.frame // self.step - self.start

        self.get_p2_per_lipid(resid_selection_leaflet = self.resid_selection_0, leaflet = 0, resid_heads_selection_leaflet = self.resid_heads_selection_0)
        self.get_p2_per_lipid(resid_selection_leaflet = self.resid_selection_1, leaflet = 1, resid_heads_selection_leaflet = self.resid_heads_selection_1)

        #Sterols
        for key, val in zip(self.resid_selection_sterols.keys(), self.resid_selection_sterols.values()):
            #Check leaflet assignment -> Iterate first over both leaflets
            min_dists = []
            for idx, leafgroup in enumerate(self.leafletfinder.groups_iter()):

                dist_arr = distances.distance_array(self.resid_selection_sterols_heads[str(key)].center_of_mass(),
                                                    leafgroup.positions,
                                                    box=self.universe.trajectory.ts.dimensions)

                min_dists.append(np.min(dist_arr))

            #Check closest distance to leaflet
            getattr(self.results, f'id{key}')['Leaflet'][self.index] = np.argmin(min_dists)
            #Store head position
            getattr(self.results, f'id{key}')[f'Heads'][self.index] = self.resid_selection_sterols_heads[str(resid)].center_of_mass()
            
            getattr(self.results, f'id{key}')['P2_0'][self.index] = self.calc_p2(pair = self.resid_selection_sterols[str(key)])


    def _conclude(self):

        """
        Calculate the final results of the analysis

        Extract the obtained data and put them into a clear and accessible data structure
        """

        #-----------------------------------------------------------------------
        #Make a dictionary for the P2 values of each lipid type for each leaflet
        #-----------------------------------------------------------------------

        """
        The result should be a dictionary with the following structure:

            - p2_per_type
                - Leaf0
                    - TypeA_0 -> ( NumberOfLipids, NumberOfFrames, NumberOfPairs)
                    - TypeA_1 -> ( NumberOfLipids, NumberOfFrames, NumberOfPairs) 
                    - TypeB_0 -> ( NumberOfLipids, NumberOfFrames, NumberOfPairs) 
                    - ...
                - Leaf1
                    - TypeA_0 -> ( NumberOfLipids, NumberOfFrames, NumberOfPairs) 
                    - ...
        """

        #Initialize storage dictionary

        self.results.p2_per_type = {}

        #Iterate over leaflets
        for i in range(2):

            #Make dictionary for each leaflet
            self.results.p2_per_type[f"Leaf{i}"] = {}
        
            #Iterate over resnames in each leaflet
            for rsn in np.unique(self.leafletfinder.group(i).resnames):

                #Iterate over number of acyl chains in lipid named "rsn"
                for n_chain in range( len(self.tails[rsn]) ): 

                    #Make a list for each acyl chain in resn
                    self.results.p2_per_type[f"Leaf{i}"][f"{rsn}_{n_chain}"] = []

        #-------------------------------------------------------------

        #Fill dictionary with obtained data

        #Iterate over all residues in the selected membrane
        for resid in self.membrane_unique_resids:

            #Grab leaflet and resname
            leaflet = getattr(self.results, f'id{resid}')['Leaflet']
            rsn = getattr(self.results, f'id{resid}')['Resname']

            #Check if lipid is a sterol compound or not
            if rsn not in self.sterols:
            
                #Iterate over chains -> For a normal phospholipid that should be 2
                for n_chain in range( len(self.tails[rsn]) ): 

                    #Get individual lipid p2 values for corresponding chain
                    indv_p2 = getattr(self.results, f'id{resid}')[f'P2_{n_chain}']

                    #Add it to the lipid type list
                    self.results.p2_per_type[f"Leaf{leaflet}"][f"{rsn}_{n_chain}"].append(indv_p2)

            elif rsn in self.sterols:

                pass

            #NOTHING?
            else: raise ValueError(f'{resname} with resid {resid} not found in leaflets or sterol list!')

        #-------------------------------------------------------------
        
        #Transform lists to arrays

        #Iterate over leaflets
        for i in range(2):

            #Iterate over lipid in leaflet
            for rsn in np.unique(self.leafletfinder.group(i).resnames): 
                
                #Check for sterol compound
                if rsn not in self.sterols:

                    #Iterate over chain
                    for n_chain in range( len(self.tails[rsn]) ):

                        #Transform list to array
                        self.results.p2_per_type[f"Leaf{i}"][f"{rsn}_{n_chain}"] = np.array(self.results.p2_per_type[f"Leaf{i}"][f"{rsn}_{n_chain}"])

        #-------------------------------------------------------------
        #-------------------------------------------------------------

        #---------------------------------------------------------------------------
        #Make a dictionary with averaged P2 values per C-H2 (or C-H) group PER chain 
        #---------------------------------------------------------------------------

        self.results.mean_p2_per_type = {}

        #Iterate over leaflets
        for leaf_key, leaf in zip(self.results.p2_per_type.keys(), self.results.p2_per_type.values()):

            self.results.mean_p2_per_type[leaf_key] = {}

            #Iterate over lipid types
            for key, val in zip(self.tails.keys(), self.tails.values()):

                #Iterate over chains for each lipid type
                for i, chain in enumerate(val):

                    #Check if lipid type is in leaflet
                    if f"{key}_{i}" in leaf.keys():

                        #Get all pairs in chain
                        pairs_in_chain = np.array_split(chain, len(chain) // 2)
                        n_pairs = len(pairs_in_chain)

                        order_per_chain = []

                        #Iterate over pairs
                        for j in range( n_pairs - 2 + 1):

                            #Check if a pair has the same aliphatic C-Atom

                            #If so -> Calculate the average (i.e. C1-H1S and C1-H1R)
                            if pairs_in_chain[j][0] == pairs_in_chain[j+1][0]: order_per_chain.append( leaf[f"{key}_{i}"][:, :, j:j+2].mean(-1).T )

                            #If there is a C-Atom UNEQUAL to the former AND the following C-Atom -> Assume double bond -> No average over pairs

                            #Edge case:
                            # j = 0 -> j-1 = -1 
                            # Should not matter since latest atom in aliphatic name is named differently than first one -> Should also work for double bonds at first place
                            elif pairs_in_chain[j][0] != pairs_in_chain[j+1][0] and pairs_in_chain[j][0] != pairs_in_chain[j-1][0]: order_per_chain.append( leaf[f"{key}_{i}"][:, :, j].T )

                            #If just the following C-Atom is unequal pass on
                            elif pairs_in_chain[j][0] != pairs_in_chain[j+1][0]: pass

                            else:
                                raise ValueError(f"Something odd in merging order parameters for {key} in chain {i} per CH2 happened!")

                        mean_p2_per_type[leaf_key][f"{key}_{i}"] = np.array(order_per_chain).T

                else: pass







#        self.results.gaussian_mixture = {}
#        for key in self.heads.keys():
#
#            #---------------------------------------Prep data---------------------------------------#
#            self.results.gaussian_mixture[key] = {}
#            mean_ = np.hstack((self.results.p2_per_type['0'][key].flatten(), self.results.p2_per_type['1'][key].flatten()))
#
#            #---------------------------------------Gaussian Mixture---------------------------------------#
#            self.results.gaussian_mixture[key]['Mean'] = mean_
#
#            GM = mixture.GaussianMixture(n_components=2, **self.gm_kwargs).fit( mean_.reshape((-1, 1)) )
#
#            self.results.gaussian_mixture[key]['GaussianMixtureModel'] = GM
#
#            #---------------------------------------Gaussian Mixture Results---------------------------------------#
#            param_o = np.argmax(GM.means_)
#            param_d = np.argmin(GM.means_)
#            mu_o, sig_o = GM.means_[param_o], np.sqrt(GM.covariances_[param_o][0])
#            mu_d, sig_d = GM.means_[param_d], np.sqrt(GM.covariances_[param_d][0])
#
#            weights_o = GM.weights_[param_o]
#            weights_d = GM.weights_[param_d]
#            
#            self.results.gaussian_mixture[key]['O_Mean'] = mu_o
#            self.results.gaussian_mixture[key]['O_STD'] = sig_o
#            self.results.gaussian_mixture[key]['O_W'] = weights_o
#
#            self.results.gaussian_mixture[key]['D_Mean'] = mu_d
#            self.results.gaussian_mixture[key]['D_STD'] = sig_d
#            self.results.gaussian_mixture[key]['D_W'] = weights_d
#
#            lipid_o = weights_o * stats.norm.pdf(x = np.linspace(-.5, 1, 1501), loc = mu_o, scale = sig_o)
#            lipid_d = weights_d * stats.norm.pdf(x = np.linspace(-.5, 1, 1501), loc = mu_d, scale = sig_d)
#
#            self.results.gaussian_mixture[key]['Lipid_O'] = np.vstack((np.linspace(-.5, 1, 1501), lipid_o ))
#            self.results.gaussian_mixture[key]['Lipid_D'] = np.vstack((np.linspace(-.5, 1, 1501), lipid_d ))
#
#            #---------------------------------------Intermediate Distribution---------------------------------------#
#            mu_I = (sig_d * mu_o + sig_o * mu_d) / (sig_d + sig_o)
#            sig_I= np.min( [ np.abs(mu_o - mu_I), np.abs(mu_d - mu_I) ] ) / 3
#            lipid_i = stats.norm.pdf(x = np.linspace(-.5, 1, 1501), loc = mu_I, scale = sig_I)
#
#            self.results.gaussian_mixture[key]['Lipid_I'] = np.vstack((np.linspace(-.5, 1, 1501), lipid_i ))
#            self.results.gaussian_mixture[key]['I_Mean'] = mu_I
#            self.results.gaussian_mixture[key]['I_STD'] = sig_I
#            self.results.gaussian_mixture[key]['I_W'] = (weights_o + weights_d)/2
#

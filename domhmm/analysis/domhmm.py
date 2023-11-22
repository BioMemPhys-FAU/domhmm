"""
LocalFluctuation --- :mod:`elbe.analysis.LocalFluctuation`
===========================================================

This module contains the :class:`LocalFluctuation` class.

"""

from .base import LeafletAnalysisBase
from .hmm import HMM

from typing import Union, TYPE_CHECKING, Dict, Any

import numpy as np
from MDAnalysis.analysis import distances
from sklearn import mixture
from hmmlearn.hmm import GaussianHMM
from scipy import stats
import sys
import memsurfer

from tqdm import tqdm

class PropertyCalculation(LeafletAnalysisBase):
    """
    The DirectorOrder class calculates a order parameter for each selected lipid according to the formula:

        P2 = 0.5 * (3 * cos(a)^2 - 1), (1)

    where a is the angle between a pre-defined director in the lipid (e.g. C-H or CC) and a reference axis.

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
            if resid in self.leaflet_selection["0"].resids and resid not in self.leaflet_selection["1"].resids:

                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})                     #-> Setup an empty dictionary
                getattr(self.results, f'id{resid}')['Leaflet'] = 0          #-> Store information about leaflet assignment
                getattr(self.results, f'id{resid}')['Resname'] = resname    #-> Store lipid type

                #Iterate over leaflet tails
                n_tails = len(self.tails[resname])
                for i in range(n_tails):

                    n_pairs = len(self.tails[resname][i]) // 2

                    #Init storage for P2 values for each lipid
                    getattr(self.results, f'id{resid}')[f'P2_{i}'] = np.zeros( (self.n_frames, n_pairs ), dtype = np.float32)

                #Store 3-D position of head group for each lipid
                getattr(self.results, f'id{resid}')[f'Heads'] = np.zeros( (self.n_frames, 3), dtype = np.float32) 

                #Store the area per lipid for each lipid
                getattr(self.results, f'id{resid}')[f'APL'] = np.zeros( self.n_frames, dtype = np.float32) 

            #LEAFLET 1?
            elif resid in self.leaflet_selection["1"].resids and resid not in self.leaflet_selection["0"].resids:
                
                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})                     #-> Setup an empty dictionary
                getattr(self.results, f'id{resid}')['Leaflet'] = 1          #-> Store information about leaflet assignment
                getattr(self.results, f'id{resid}')['Resname'] = resname    #-> Store lipid type

                #Iterate over leaflet tails
                n_tails = len(self.tails[resname])
                for i in range(n_tails):

                    n_pairs = len(self.tails[resname][i]) // 2

                    #Init storage for P2 values for each lipid
                    getattr(self.results, f'id{resid}')[f'P2_{i}'] = np.zeros( (self.n_frames, n_pairs ), dtype = np.float32)

                #Store 3-D position of head group for each lipid
                getattr(self.results, f'id{resid}')[f'Heads'] = np.zeros( (self.n_frames, 3), dtype = np.float32) 

                #Store the area per lipid for each lipid
                getattr(self.results, f'id{resid}')[f'APL'] = np.zeros( self.n_frames, dtype = np.float32) 

            #STEROL?
            elif resid not in self.leaflet_selection["0"].resids and resid not in self.leaflet_selection["1"].resids and resname in self.sterols :

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


    def calc_p2(self, pair, reference_axis):

        """
        Calculates the deuterium order parameter according to equation (1) for each pair.

        Parameters
        ----------
        pair: MDAnalysis atom selection
            selection group containing the two atoms for the director calculation

        reference_axis: numpy.array
            numpy array containing the reference vector for the angle calculation.
        """

        r = pair.positions[0] - pair.positions[1]
        r /= np.sqrt(np.sum(r**2))


        #Dot product between membrane normal (z axis) and orientation vector
        dot_prod = np.dot(r, reference_axis)
        a = np.arccos(dot_prod) #Angle in radians
        P2 = 0.5 * (3 * np.cos(a)**2 - 1)

        #Flip sign of order parameters
        P2 = -1 * P2

        return P2

    def get_p2_per_lipid(self, resid_tails_selection_leaflet, leaflet, resid_heads_selection_leaflet, local_normals, refZ):

        """
        Applies P2 calculation for each C-H pair in an individual lipid for each leaflet.

        Parameters
        ----------
        resid_tails_selection_leaflet : dictionary
            Contains MDAnalysis atom selection for tail group of individual lipids per leaflet
        leaflet : int
            Leaflet of interest
        resid_heads_selection_leaflet : dictionary
            Contains MDAnalysis atom selection for head group of individual lipids per leaflet
        local_normals : dictionary
            Containing local normals per lipid -> keys are the resids
        refZ : bool
            Using the z-axis as reference axis or the local normal defined per lipid?


        """

        #Iterate over resids in leaflet
        for key in resid_heads_selection_leaflet.keys():

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
                for j in range(len(self.tails[rsn][n_chain]) // 2):

                    if refZ:
                        getattr(self.results, f'id{key}')[f'P2_{n_chain}'][self.index, j] = self.calc_p2(pair = resid_tails_selection_leaflet[str(key)][ str(n_chain) ][ j ], reference_axis = np.array([0, 0, 1]))
                    else:
                        getattr(self.results, f'id{key}')[f'P2_{n_chain}'][self.index, j] = self.calc_p2(pair = resid_tails_selection_leaflet[str(key)][ str(n_chain) ][ j ], reference_axis = local_normals[f"{key}"])

    def get_local_area_normal(self, leaflet, boxdim, periodic = True, exactness_level = 10):

        """
        Calculate area per lipid and local membrane normal with MemSurfer library.

        Parameters
        ----------

        leaflet: int
            Top or bottom leaflet
        boxdim: np.array
            Box dimensions in x, y, z
        periodic: bool
            Usage of periodic boundary conditions during simulation
        exactness_level: int
            Approximating surface using Poisson reconstruction
        """

        #Prepare box dimensions -> Seems to be used also for box width calculation (like bbox[1,:] - bbox[0, :]). First row where therefore the lower limit and second the upper limit
        bbox = np.zeros( (2, 3) )
        bbox[1, :] = boxdim

        mem = memsurfer.Membrane( points = self.leafletfinder.groups( leaflet ).positions,
                                  labels = self.leafletfinder.groups( leaflet ).resids.astype("U"), 
                                  bbox = bbox,
                                  periodic = periodic,
                                  boundary_layer = 0.2 #Default value
                                 )

        #Put points back into box
        mem.fit_points_to_box_xy()

        #Approximate surface -> Uses as standard 18 k-neighbours for normal calculation
        mem.compute_approx_surface(exactness_level = exactness_level)

        #Compute membrane surfaces based on the approximated surface calculated above:
        # - memb_planar := Planar projections of points on the smoothed surface
        # - memb_smooth := Points of the smoothed surface
        # - memb_exact  := Exact coordinates of lipids from trajectory
        mem.compute_membrane_surface()

        local_normals = mem.memb_smooth.compute_normals()

        local_area_per_lipid = mem.memb_smooth.compute_pointareas()

        local_normals_dict = dict( zip(mem.labels, local_normals) )

        for resid, apl in zip(mem.labels, local_area_per_lipid): getattr(self.results, f'id{resid}')[f'APL'][self.index] = apl

        return local_normals_dict

    def _single_frame(self):
        """
        Calculate data from a single frame of the trajectory.
        """


        #Get number of frame from trajectory
        self.frame = self.universe.trajectory.ts.frame
        #Calculate correct index if skipping step not equals 1 or start point not equals 0
        self.index = self.frame // self.step - self.start

        #------------------------------------------------------Local Normals/Area per Lipid------------------------------------------------------#
        boxdim = self.universe.trajectory.ts.dimensions[0:3]
        local_normals_dict_0 = self.get_local_area_normal(leaflet = 0, boxdim = boxdim, periodic = True, exactness_level = 10)
        local_normals_dict_1 = self.get_local_area_normal(leaflet = 1, boxdim = boxdim, periodic = True, exactness_level = 10)

        #------------------------------------------------------Order parameter------------------------------------------------------#
        self.get_p2_per_lipid(resid_tails_selection_leaflet = self.resid_tails_selection_0, leaflet = 0, resid_heads_selection_leaflet = self.resid_heads_selection_0, local_normals = local_normals_dict_0, refZ = self.refZ)
        self.get_p2_per_lipid(resid_tails_selection_leaflet = self.resid_tails_selection_1, leaflet = 1, resid_heads_selection_leaflet = self.resid_heads_selection_1, local_normals = local_normals_dict_1, refZ = self.refZ)

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
        #Make a dictionary for the calculated values of each lipid type for each leaflet
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

            - apl_per_type
                - Leaf0
                    - TypeA -> (NumberOfLipids, NumberOfFrames)
                    - TypeB -> (NumberOfLipids, NumberOfFrames)
                    - ...
                - Leaf1
                    - TypeA -> (NumberOfLipids, NumberOfFrames)
                    - ...

        """

        #Initialize storage dictionary

        self.results.p2_per_type = {}
        self.results.apl_per_type = {}

        #Iterate over leaflets
        for i in range(2):

            #Make dictionary for each leaflet
            self.results.p2_per_type[f"Leaf{i}"] = {}
            self.results.apl_per_type[f"Leaf{i}"] = {}
        
            #Iterate over resnames in each leaflet
            for rsn in np.unique(self.leafletfinder.group(i).resnames):

                #Iterate over number of acyl chains in lipid named "rsn"
                for n_chain in range( len(self.tails[rsn]) ): 

                    #Make a list for each acyl chain in resn
                    self.results.p2_per_type[f"Leaf{i}"][f"{rsn}_{n_chain}"] = []
                
                self.results.apl_per_type[f"Leaf{i}"][f"{rsn}"] = []

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
            
                #Get area per lipid for specific residue
                apl = getattr(self.results, f'id{resid}')['APL']
                self.results.apl_per_type[f"Leaf{leaflet}"][f"{rsn}"].append( apl )

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
                    
                    #Just transform for area per lipid
                    self.results.apl_per_type[f"Leaf{i}"][f"{rsn}"] = np.array(self.results.apl_per_type[f"Leaf{i}"][f"{rsn}"])

        #-------------------------------------------------------------
        #-------------------------------------------------------------

        #---------------------------------------------------------------------------
        #Make a dictionary with averaged P2 values per C-H2 (or C-H) group PER chain 
        #---------------------------------------------------------------------------

        """
        The result should be a dictionary with the following structure:

            - mean_p2_per_type
                - Leaf0
                    - TypeA_0 -> ( NumberOfLipids, NumberOfFrames, NumberOfAcylCAtoms)
                    - TypeA_1 -> ( NumberOfLipids, NumberOfFrames, NumberOfAcylCAtoms) 
                    - TypeB_0 -> ( NumberOfLipids, NumberOfFrames, NumberOfAcylCAtoms) 
                    - ...
                - Leaf1
                    - TypeA_0 -> ( NumberOfLipids, NumberOfFrames, NumberOfAcylCAtoms) 
                    - ...

        That is not necessary for the area per lipid since this is just a scalar
        """

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

                        #Adding a dummy array ensures that double bonds at the end of an acyl chain are taken into account
                        pairs_in_chain +=  [np.array(["dummy", "dummy"])]

                        n_pairs = len(pairs_in_chain)

                        order_per_chain = []

                        #Iterate over pairs
                        for j in range( n_pairs - 2 + 1):

                            #Check if a pair has the same aliphatic C-Atom

                            #If so -> Calculate the average (i.e. C1-H1S and C1-H1R)
                            #I transpose the resulting arrays several times to get a more logical shape of the resulting array
                            if pairs_in_chain[j][0] == pairs_in_chain[j+1][0]: order_per_chain.append( leaf[f"{key}_{i}"][:, :, j:j+2].mean(-1).T )

                            #If there is a C-Atom UNEQUAL to the former AND the following C-Atom -> Assume double bond -> No average over pairs

                            #Edge case:
                            # j = 0 -> j-1 = -1 
                            # Should not matter since latest atom in aliphatic name is named differently than first one -> Should also work for double bonds at the first place of the 
                            elif pairs_in_chain[j][0] != pairs_in_chain[j+1][0] and pairs_in_chain[j][0] != pairs_in_chain[j-1][0]: order_per_chain.append( leaf[f"{key}_{i}"][:, :, j].T )

                            #If just the following C-Atom is unequal pass on
                            elif pairs_in_chain[j][0] != pairs_in_chain[j+1][0]: pass

                            else:
                                raise ValueError(f"Something odd in merging order parameters for {key} in chain {i} per CH2 happened!")

                        self.results.mean_p2_per_type[leaf_key][f"{key}_{i}"] = np.array(order_per_chain).T

                else: pass

    #-------------------------------------------------------------FIT GAUSSIAN MIXTURE MODEL-------------------------------------------------------------#
    def GMM(self, gmm_kwargs = {}):

        self.results["GMM"] = {}

        """
        Structure as follows:

            - GMM
                - Leaf0
                    - LipidA
                        - GMM Results Tail 1
                        - GMM Results Tail 2
                        - ..
                        - GMM Results APL
                    - ...
                - Leaf1
                    - LipidA
                        - ...


        """

        #Iterate over leaflets
        for idx, leafgroup in zip(self.leaflet_selection.keys(), self.leaflet_selection.values()):

            #Init empty dictionary for each leaflet
            self.results["GMM"][f"Leaf{idx}"] = {}

        self.get_gmm_order_parameters(0, gmm_kwargs = gmm_kwargs)
        self.get_gmm_order_parameters(1, gmm_kwargs = gmm_kwargs)

        self.get_gmm_area_per_lipid(0, gmm_kwargs = gmm_kwargs)
        self.get_gmm_area_per_lipid(1, gmm_kwargs = gmm_kwargs)

    def get_gmm_order_parameters(self, leaflet, gmm_kwargs):

        #Get lipid types in leaflet
        leaflet_resnames = np.unique( self.leaflet_resids[ str(leaflet) ].resnames ) 

        #Iterate over lipids in leaflet
        for rsn in leaflet_resnames:

            #Iterate over tails (e.g. for standard phospholipids that 2)
            for i, tail in enumerate(self.tails[rsn]):

                self.results["GMM"][f"Leaf{leaflet}"][f"{rsn}_{i}"] = self.fit_gmm(property_ = self.results.mean_p2_per_type[f"Leaf{leaflet}"][f"{rsn}_{i}"].mean(2), gmm_kwargs = gmm_kwargs)

    def get_gmm_area_per_lipid(self, leaflet, gmm_kwargs):

        #Get lipid types in leaflet
        leaflet_resnames = np.unique( self.leaflet_resids[ str(leaflet) ].resnames ) 

        #Iterate over lipids in leaflet
        for rsn in leaflet_resnames:

            self.results["GMM"][f"Leaf{leaflet}"][f"{rsn}_APL"] = self.fit_gmm(property_ = self.results.apl_per_type[f"Leaf{leaflet}"][f"{rsn}"], gmm_kwargs = gmm_kwargs)

    def fit_gmm(self, property_, gmm_kwargs):

        """
        Fit a Gaussian Mixture Model for each lipid type to the results of the property calculation.
        This is done here for each leaflet seperatley!


        Parameters
        ----------
        property_ : numpy.array
            Input data for the gaussian mixture model ( Shape: (NLipids, NFrames) )


        """

        assert self.n_frames == property_.shape[1], "Wrong input shape for the fitting of the GMM!"

        #---------------------------------------Prep data---------------------------------------#

        #Take arithmetic mean over chain order parameters
        property_flatten = property_.flatten() #Shape change (NLipids, NFrames) -> (NLipids * NFrames, )

        #---------------------------------------Gaussian Mixture---------------------------------------#

        #Run the GaussianMixture Model for two components
        GM = mixture.GaussianMixture(n_components=2, **gmm_kwargs).fit( property_flatten.reshape((-1, 1)) )

        #---------------------------------------Gaussian Mixture Results---------------------------------------#

        #The Gaussian distribution with the highest mean corresponds to the ordered state
        param_o = np.argmax(GM.means_)
        #The Gaussian distribution with the lowest mean corresponds to the disoredered state
        param_d = np.argmin(GM.means_)

        #Get mean and variance of the fitted Gaussian distributions
        mu_o, var_o = GM.means_[param_o], GM.covariances_[param_o][0]
        mu_d, var_d = GM.means_[param_d], GM.covariances_[param_d][0]

        sig_o = np.sqrt( var_o )
        sig_d = np.sqrt( var_d )

        #---------------------------------------Intermediate Distribution---------------------------------------#
        mu_I = (sig_d * mu_o + sig_o * mu_d) / (sig_d + sig_o)
        sig_I= np.min( [ np.abs(mu_o - mu_I), np.abs(mu_d - mu_I) ] ) / 3
        var_I = sig_I**2

        #----------------------------------------Conclude----------------------------------------#
        #Put the fitted results in an easy to access format
        fit_results = np.empty( (3, 2), dtype =np.float32)

        fit_results[0, 0], fit_results[0, 1] = mu_d, var_d
        fit_results[1, 0], fit_results[1, 1] = mu_I, var_I
        fit_results[2, 0], fit_results[2, 1] = mu_o, var_o

        return fit_results


    #-------------------------------------------------------------FIT HIDDEN MARKOW MODEL-------------------------------------------------------------#

    def HMM(self, n_repeats, hmm_kwargs = {}):

        if len(self.results["GMM"]) == 0:
            print("!!!---WARNING---!!!")
            print("No Gaussian Mixture Model data found! Pleasr run GMM first!")
            return

        else: pass

        self.results["HMM"] = {}

        """
        Structure as follows:

            - HMM
                - Leaf0
                    - LipidA
                        - Trained HMM for Tail 1 of Lipid A
                        - Trained HMM for Tail 2 of Lipid A
                        - Trained HMM for APL of Lipid A
                    - ...
                - Leaf1
                    - LipidA
                        - ...


        """

        #Iterate over leaflets
        for idx, leafgroup in zip(self.leaflet_selection.keys(), self.leaflet_selection.values()):

            #Init empty dictionary for each leaflet
            self.results["HMM"][f"Leaf{idx}"] = {}

        self.get_hmm_order_parameters(leaflet = 0, n_repeats = 10, hmm_kwargs = hmm_kwargs)
        self.get_hmm_order_parameters(leaflet = 1, n_repeats = 10, hmm_kwargs = hmm_kwargs)

        self.get_hmm_area_per_lipid(leaflet = 0, n_repeats = 10, hmm_kwargs = hmm_kwargs)
        self.get_hmm_area_per_lipid(leaflet = 1, n_repeats = 10, hmm_kwargs = hmm_kwargs)

    def get_hmm_order_parameters(self, leaflet, n_repeats, hmm_kwargs):

        #Get lipid types in leaflet
        leaflet_resnames = np.unique( self.leaflet_resids[ str(leaflet) ].resnames ) 

        #Iterate over lipids in leaflet
        for rsn in leaflet_resnames:

            #Iterate over tails (e.g. for standard phospholipids that 2)
            for i, tail in enumerate(self.tails[rsn]):

                self.results["HMM"][f"Leaf{leaflet}"][f"{rsn}_{i}"] = self.fit_hmm(property_ = self.results.mean_p2_per_type[f"Leaf{leaflet}"][f"{rsn}_{i}"].mean(2),
                                                                                   init_params = self.results.GMM[f"Leaf{leaflet}"][f"{rsn}_{i}"],
                                                                                   n_repeats = n_repeats,
                                                                                   hmm_kwargs = hmm_kwargs)

    def get_hmm_area_per_lipid(self, leaflet, n_repeats, hmm_kwargs):

        #Get lipid types in leaflet
        leaflet_resnames = np.unique( self.leaflet_resids[ str(leaflet) ].resnames ) 

        #Iterate over lipids in leaflet
        for rsn in leaflet_resnames:

            self.results["HMM"][f"Leaf{leaflet}"][f"{rsn}_APL"] = self.fit_hmm(property_ = self.results.apl_per_type[f"Leaf{leaflet}"][f"{rsn}"],
                                                                               init_params = self.results.GMM[f"Leaf{leaflet}"][f"{rsn}_APL"],
                                                                               n_repeats = n_repeats,
                                                                               hmm_kwargs = hmm_kwargs)

    def fit_hmm(self, property_, init_params, n_repeats, hmm_kwargs):


        assert self.n_frames == property_.shape[1], "Wrong input shape for the fitting of the HMM!"

        n_lipids = property_.shape[0]

        means_ = init_params[:, 0].reshape(-1, 1)
        vars_ = init_params[:, 1].reshape(-1, 1)

        bagging = []
        bic_ = []

        for i in range(n_repeats):
            
            GHMM = GaussianHMM(n_components = 3, means_prior = means_ , covars_prior = vars_, **hmm_kwargs)

            GHMM.fit( property_.flatten().reshape(-1, 1), lengths = np.repeat( self.n_frames, n_lipids ) )

            bic_.append( GHMM.bic( property_.flatten().reshape(-1, 1), lengths = np.repeat( self.n_frames, n_lipids ) ) )

            bagging.append( GHMM )

            del GHMM

        best_model = bagging[ np.argmin(bic_) ]

        return best_model










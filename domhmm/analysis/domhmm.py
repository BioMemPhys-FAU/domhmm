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

            #Check leaflet assignment
            #LEAFLET 0?
            if resid in self.leafletfinder.group(0).resids and resid not in self.leafletfinder.group(1).resids:

                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})                     #-> Setup an empty dictionary
                getattr(self.results, f'id{resid}')['Leaflet'] = 0          #-> Store information about leaflet assignment
                getattr(self.results, f'id{resid}')['Resname'] = resname    #-> Store lipid type

                #Iterate over leaflet tails
                n_tails = len(self.tails[resname])
                for i in range(n_tails):

                    #For each tail store an array
                    n_pairs = len(self.tails[resname][i]) // 2
                    getattr(self.results, f'id{resid}')[f'P2_{i}'] = np.zeros( (self.n_frames, n_pairs ), dtype = np.float32)

            #LEAFLET 1?
            elif resid in self.leafletfinder.group(1).resids and resid not in self.leafletfinder.group(0).resids:
                
                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})                     #-> Setup an empty dictionary
                getattr(self.results, f'id{resid}')['Leaflet'] = 1          #-> Store information about leaflet assignment
                getattr(self.results, f'id{resid}')['Resname'] = resname    #-> Store lipid type

                #Iterate over leaflet tails
                n_tails = len(self.tails[resname])
                for i in range(n_tails):

                    #For each tail store an array
                    n_pairs = len(self.tails[resname][i]) // 2
                    getattr(self.results, f'id{resid}')[f'P2_{i}'] = np.zeros( (self.n_frames, n_pairs ), dtype = np.float32)

            #STEROL?
            elif resid not in self.leafletfinder.group(0).resids and resid not in self.leafletfinder.group(1).resids and resname in self.sterols:

                #Sterols are not assigned to a specific leaflet -> They can flip. Maybe it is unlikely that it happens in each membrane (especially atomistic ones)
                #but it can happen and the code keeps track of them.

                #Make a MDAnalysis atom selection for each resid. For the other lipids this was already done in the LeafletAnalysisBase class
                self.resid_selection_sterols[str(resid)] = resid_selection.intersection(self.sterols_tail[resname])
                self.resid_selection_sterols_heads[str(resid)] = resid_selection.intersection(self.sterols_head[resname])
                
                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})
                getattr(self.results, f'id{resid}')['Leaflet'] = np.zeros( (self.n_frames), dtype = np.float32)
                getattr(self.results, f'id{resid}')['Resname'] = resname
                getattr(self.results, f'id{resid}')['P2_0'] = np.zeros( (self.n_frames), dtype = np.float32)

            #NOTHING?
            else: raise ValueError(f'{resname} with resid {resid} not found in leaflets and sterol list!')


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
        a = np.arccos(r[2])
        P2 = 0.5 * (3 * np.cos(a)**2 - 1)

        return P2

    def _single_frame(self):
        """
        Calculate data from a single frame of the trajectory.
        """

        self.frame = self.universe.trajectory.ts.frame 
        index = self.frame // self.step - self.start

        #Iterate over resids in leaflet 0
        for key, val in zip(self.resid_selection_0.keys(), self.resid_selection_0.values()): 
            #Check if leaflet is correct
            assert getattr(self.results, f'id{key}')['Leaflet'] == 0, '!!!-----ERROR-----!!!\nWrong leaflet\n!!!-----ERROR-----!!!'

            rsn = getattr(self.results, f'id{key}')['Resname']
            
            #Iterate over tails
            for i in range(len(self.tails[rsn])):
                #Iterate over pairs
                for j in range(len(self.tails[rsn][i]) // 2):

                    getattr(self.results, f'id{key}')[f'P2_{i}'][index, j] = self.calc_p2(pair = self.resid_selection_0[str(key)][str(i)][j])

        #Iterate over resids in leaflet 1
        for key, val in zip(self.resid_selection_1.keys(), self.resid_selection_1.values()):
            #Check if leaflet is correct
            assert getattr(self.results, f'id{key}')['Leaflet'] == 1, '!!!-----ERROR-----!!!\nWrong leaflet\n!!!-----ERROR-----!!!'
            
            rsn = getattr(self.results, f'id{key}')['Resname']

            #Iterate over tails
            for i in range(len(self.tails[rsn])):
                #Iterate over pairs
                for j in range(len(self.tails[rsn][i]) // 2):
                    getattr(self.results, f'id{key}')[f'P2_{i}'][index, j] = self.calc_p2(pair = self.resid_selection_1[str(key)][str(i)][j])

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
            getattr(self.results, f'id{key}')['Leaflet'][index] = np.argmin(min_dists)
            getattr(self.results, f'id{key}')['P2_0'][index] = self.calc_p2(pair = self.resid_selection_sterols[str(key)])


    def _conclude(self):
        """Calculate the final results of the analysis"""

        self.results.p2_per_lipid = {}

        for key in self.tails.keys(): 
            if key not in self.sterols:

                #Create dictionary entry for each lipid type and add two lists for the two membrane leaflets
                self.results.p2_per_lipid[key] = [[], []]

                #For each leaflet list a list for each tail is written
                for i in range(len(self.tails[key])): 
                    self.results.p2_per_lipid[key][0].append([])
                    self.results.p2_per_lipid[key][1].append([])

        #Iterate over resids
        for resid in self.membrane_unique_resids:

            #Check resname
            resn = getattr(self.results, f'id{resid}')['Resname']

            #Only non-sterols are considered
            if resn not in self.sterols:
                #Get leaflet assignment
                leaf = getattr(self.results, f'id{resid}')['Leaflet']
                #Iterate over tails
                for i in range(len(self.tails[resn])):
                    #Append the correct order parameters
                    self.results.p2_per_lipid[resn][leaf][i].append( getattr(self.results, f'id{resid}')[f'P2_{i}'] )

        for key in self.results.p2_per_lipid.keys():
            for i in range(2):
                for j in range( len(self.results.p2_per_lipid[key][i])):
                    self.results.p2_per_lipid[key][i][j] = np.array(self.results.p2_per_lipid[key][i][j])





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

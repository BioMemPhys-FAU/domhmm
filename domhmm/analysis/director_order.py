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

        #Make a selection for resids in the upper and lower leaflet
        self.resid_selection_0 = {}
        self.resid_selection_1 = {}

        #Make selections for sterols and their headgroup
        self.resid_selection_sterols = {}
        self.resid_selection_sterols_heads = {}
        
        #Iterate over resids in the membrane
        for resid in self.membrane_unique_resids:

            #Select specific resid
            resid_selection = self.universe.select_atoms(f"resid {resid}")

            #Get resname of resid
            resname = np.unique(resid_selection.resnames)[0]

            #---------------------------------------------------------------------------------------------------------------------------------------------------------
            #Look up if resid is in the UPPER leaflet or ...
            if resid in self.leafletfinder.group(0).resids and resid not in self.leafletfinder.group(1).resids:

                #Go through list of selected directors and choose the atoms for this resid
                self.resid_selection_0[str(resid)] = [ resid_selection.intersection(self.leaflet_tails['0'][resname][i]) for i in range( len(self.tails[resname])//2 )]
                
                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})

                #Assign...
                getattr(self.results, f'id{resid}')['Leaflet'] = 0       #...leaflet
                getattr(self.results, f'id{resid}')['Resname'] = resname #...resname
                getattr(self.results, f'id{resid}')['P2'] = np.zeros( (self.n_frames), dtype = np.float32) #...storage array for P2 values

            #---------------------------------------------------------------------------------------------------------------------------------------------------------
            #... in the LOWER leaflet or ...
            elif resid in self.leafletfinder.group(1).resids and resid not in self.leafletfinder.group(0).resids:

                #Go through list of selected directors and choose the atoms for this resid
                self.resid_selection_1[str(resid)] = [ resid_selection.intersection(self.leaflet_tails['1'][resname][i] for i in range( len(self.tails[resname])//2 )]
                
                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})

                getattr(self.results, f'id{resid}')['Leaflet'] = 1
                getattr(self.results, f'id{resid}')['Resname'] = resname
                getattr(self.results, f'id{resid}')['P2'] = np.zeros( (self.n_frames), dtype = np.float32)


            #---------------------------------------------------------------------------------------------------------------------------------------------------------
            #... its a sterol or ...
            elif resid not in self.leafletfinder.group(0).resids and resid not in self.leafletfinder.group(1).resids and resname in self.sterols:

                self.resid_selection_sterols[str(resid)] = resid_selection.intersection(self.sterols_tail[resname])
                self.resid_selection_sterols_heads[str(resid)] = resid_selection.intersection(self.sterols_head[resname])

                #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
                setattr(self.results, f'id{resid}', {})

                getattr(self.results, f'id{resid}')['Leaflet'] = np.zeros( (self.n_frames), dtype = np.float32)
                getattr(self.results, f'id{resid}')['Resname'] = resname
                getattr(self.results, f'id{resid}')['P2'] = np.zeros( (self.n_frames), dtype = np.float32)

            #... in none of them.
            else: raise ValueError(f'{resname} not found in any leaflet! Maybe a sterol?')


    def calc_p2(self, select_list):

        """
        Function calculates the P2 order parameter according to equation (1) for a list of atom pairs

        select_list := List of atom pairs, each entry contains a MDAnalysis atom selection with two atoms

        """

        #Make average over tails
        P2 = 0
        for pair in select_list:

            assert pair.positions.shape == (2, 3), f"Error! Your pair selection in calc_p2 does not contain 2 atoms or/and 3 dimensions, but {pair.positions.shape}!"

            #Distance vector between atoms
            r = pair.positions[0] - pair.positions[1]
            #Norm the distance vector
            r /= np.sqrt(np.sum(r**2))

            #Dot product between membrane normal (z axis) and orientation vector
            a = np.arccos(r[2]) #Calculate the angle between z axis and orientation vector in radians 
            P2 += 0.5 * (3* np.cos(a)**2 - 1) #Calculate order parameter

        #Calculate arithmetic mean
        P2 /= len(select_list)

        return P2

    def _single_frame(self):
        """
        Calculate data from a single frame of the trajectory.
        """

        #Get number of frame from trajectory
        self.frame = self.universe.trajectory.ts.frame 

        #Obtain 0-based index
        index = self.frame // self.step - self.start

        #Iterate over resids in leaflet 0
        for key, val in zip(self.resid_selection_0.keys(), self.resid_selection_0.values()): 
            #Check if leaflet is correct
            assert getattr(self.results, f'id{key}')['Leaflet'] == 0, '!!!-----ERROR-----!!!\nWrong leaflet\n!!!-----ERROR-----!!!'
            #Calculate P2 parameter as average over lipid tails
            getattr(self.results, f'id{key}')['P2'][index] = self.calc_p2(select_list = self.resid_selection_0[str(key)])

        #Iterate over resids in leaflet 1
        for key, val in zip(self.resid_selection_1.keys(), self.resid_selection_1.values()):
            #Check if leaflet is correct
            assert getattr(self.results, f'id{key}')['Leaflet'] == 1, '!!!-----ERROR-----!!!\nWrong leaflet\n!!!-----ERROR-----!!!'
            #Calculate P2 parameter as average over lipid tails
            getattr(self.results, f'id{key}')['P2'][index] = self.calc_p2(select_list = self.resid_selection_1[str(key)])

        #Iterate over sterols
        for key, val in zip(self.resid_selection_sterols.keys(), self.resid_selection_sterols.values()):

            #Assign sterols to leaflet by their minimum distance to the phosphates

            min_dists = []
            #Iterate first over both leaflets
            for idx, leafgroup in enumerate(self.leafletfinder.groups_iter()):

                dist_arr = distances.distance_array(self.resid_selection_sterols_heads[str(key)].center_of_mass(),
                                                    leafgroup.positions,
                                                    box=self.universe.trajectory.ts.dimensions)

                #Get minimum distance to each leaflet
                min_dists.append(np.min(dist_arr))

            #Which minimum distance to each leaflet is the smallest one
            getattr(self.results, f'id{key}')['Leaflet'][index] = np.argmin(min_dists)
            getattr(self.results, f'id{key}')['P2'][index] = self.calc_p2(select_list = [self.resid_selection_sterols[str(key)]])


    def _conclude(self):
        """
        Calculate the final results of the analysis

        Extract the obtained data and put them in clear data structures

        """

        self.results.p2_per_type = {}

        #Iterate over leaflets
        for i in range(2):

            #Make for each leaflet a own dictionary entry
            self.results.p2_per_type[f"Leaf{i}"] = {}

            #Iterate over lipids in this leaflet and initialize an empty storage array
            for key in self.leaflet_tails[str(i)].keys(): self.results.p2_per_type[str(i)][key] = np.empty((0, self.n_frames))

            #Iterate over sterols and initialize an empty storage array
            for key in self.sterols: self.results.p2_per_type[str(i)][key] = np.empty((0, self.n_frames))

        #Sort resids in lipid types
        for resid in self.membrane_unique_resids:
            
            idx = getattr(self.results, f'id{resid}')['Leaflet']
            rsn = getattr(self.results, f'id{resid}')['Resname']
            p2 = getattr(self.results, f'id{resid}')['P2']

            #For sterols take the most common leaflet
            if type(idx) != int: idx = int(np.round(idx.mean()))

            self.results.p2_per_type[str(idx)][rsn] = np.vstack(( self.results.p2_per_type[str(idx)][rsn], p2))

        self.results.gaussian_mixture = {}
        for key in self.heads.keys():

            #---------------------------------------Prep data---------------------------------------#
            self.results.gaussian_mixture[key] = {}
            mean_ = np.hstack((self.results.p2_per_type['0'][key].flatten(), self.results.p2_per_type['1'][key].flatten()))

            #---------------------------------------Gaussian Mixture---------------------------------------#
            self.results.gaussian_mixture[key]['Mean'] = mean_

            GM = mixture.GaussianMixture(n_components=2, **self.gm_kwargs).fit( mean_.reshape((-1, 1)) )

            self.results.gaussian_mixture[key]['GaussianMixtureModel'] = GM

            #---------------------------------------Gaussian Mixture Results---------------------------------------#
            param_o = np.argmax(GM.means_)
            param_d = np.argmin(GM.means_)
            mu_o, sig_o = GM.means_[param_o], np.sqrt(GM.covariances_[param_o][0])
            mu_d, sig_d = GM.means_[param_d], np.sqrt(GM.covariances_[param_d][0])

            weights_o = GM.weights_[param_o]
            weights_d = GM.weights_[param_d]
            
            self.results.gaussian_mixture[key]['O_Mean'] = mu_o
            self.results.gaussian_mixture[key]['O_STD'] = sig_o
            self.results.gaussian_mixture[key]['O_W'] = weights_o

            self.results.gaussian_mixture[key]['D_Mean'] = mu_d
            self.results.gaussian_mixture[key]['D_STD'] = sig_d
            self.results.gaussian_mixture[key]['D_W'] = weights_d

            lipid_o = weights_o * stats.norm.pdf(x = np.linspace(-.5, 1, 1501), loc = mu_o, scale = sig_o)
            lipid_d = weights_d * stats.norm.pdf(x = np.linspace(-.5, 1, 1501), loc = mu_d, scale = sig_d)

            self.results.gaussian_mixture[key]['Lipid_O'] = np.vstack((np.linspace(-.5, 1, 1501), lipid_o ))
            self.results.gaussian_mixture[key]['Lipid_D'] = np.vstack((np.linspace(-.5, 1, 1501), lipid_d ))

            #---------------------------------------Intermediate Distribution---------------------------------------#
            mu_I = (sig_d * mu_o + sig_o * mu_d) / (sig_d + sig_o)
            sig_I= np.min( [ np.abs(mu_o - mu_I), np.abs(mu_d - mu_I) ] ) / 3
            lipid_i = stats.norm.pdf(x = np.linspace(-.5, 1, 1501), loc = mu_I, scale = sig_I)

            self.results.gaussian_mixture[key]['Lipid_I'] = np.vstack((np.linspace(-.5, 1, 1501), lipid_i ))
            self.results.gaussian_mixture[key]['I_Mean'] = mu_I
            self.results.gaussian_mixture[key]['I_STD'] = sig_I
            self.results.gaussian_mixture[key]['I_W'] = (weights_o + weights_d)/2

            #---------------------------------------Emission statss---------------------------------------#

            deltaP2 = (mu_o - mu_d) / 7

            self.results.gaussian_mixture[key]['Bins'] = mu_d + np.arange(0,8)*deltaP2

            stacked_p2 = np.hstack((self.results.p2_per_type['0'][key].flatten(), self.results.p2_per_type['1'][key].flatten()))
             
            fitdata = np.digitize(x = stacked_p2, bins = self.results.gaussian_mixture[key]['Bins'], right = False)

            self.results.gaussian_mixture[key]['FitData'] = fitdata


        #Sort resids in lipid types
        for resid in self.membrane_unique_resids:
            
            rsn = getattr(self.results, f'id{resid}')['Resname']
            p2 = getattr(self.results, f'id{resid}')['P2']

            getattr(self.results, f'id{resid}')['EmissionStates'] = np.digitize(x = p2, bins = self.results.gaussian_mixture[rsn]['Bins'], right = False)



















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
import memsurfer

class HMM(LeafletAnalysisBase):
    """
    HMM Class

    Train and fit a Hidden Markov Model to the obtained data

    """

    def 





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


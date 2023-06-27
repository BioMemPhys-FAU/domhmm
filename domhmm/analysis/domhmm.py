"""
LocalFluctuation --- :mod:`elbe.analysis.LocalFluctuation`
===========================================================

This module contains the :class:`LocalFluctuation` class.

"""

from .base import LeafletAnalysisBase
import elbe.lib
from typing import Union, TYPE_CHECKING, Dict, Any

import numpy as np

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

        #Init results for order parameters -> For each resid we should have an array containing the order parameters for each frame
        for resid in membrane_unique_resids: getattr(self.results, resid) = np.zeros( (self.n_frames), dtype = np.float32)



    def _single_frame(self):
        """
        Calculate data from a single frame of the trajectory.
        """




    def _conclude(self):
        """Calculate the final results of the analysis"""
        # This is an optional method that runs after
        # _single_frame loops over the trajectory.
        # It is useful for calculating the final results
        # of the analysis.
        # For example, below we determine the

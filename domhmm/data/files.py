"""
Location of data files
======================

Use as ::

    from elbe.data.files import *

"""

__all__ = [
    "MDANALYSIS_LOGO",  # example file of MDAnalysis logo
    "DSPC_CHOL_AA", #17nm x 17nm equilibrated membrane patch containing DSPC and cholesterol
    "DOPC_CG_TPR", #GROMCAS v2021 tpr file for a coarse grained system (MARTINI forcefield) containing a pure DOPC bilayer
    "DOPC_CG_XTC", #GROMACS compressed trajectory file containing a coarse grained system (MARTINI forcefield) containing a pure DOPC bilayer
]

from pkg_resources import resource_filename

MDANALYSIS_LOGO = resource_filename(__name__, "mda.txt")
DSPC_CHOL_AA = resource_filename(__name__, "dspc_chol.gro")
DOPC_CG_TPR = resource_filename(__name__, "DOPC_CG_TPR.tpr")
DOPC_CG_XTC = resource_filename(__name__, "DOPC_CG_XTC.xtc")

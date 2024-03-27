"""
LocalFluctuation --- :mod:`elbe.analysis.LocalFluctuation`
===========================================================

This module contains the :class:`LocalFluctuation` class.

"""

from .base import LeafletAnalysisBase

# from typing import Union, TYPE_CHECKING, Dict, Any

import numpy as np
# from MDAnalysis.analysis import distances
from sklearn import mixture
from hmmlearn.hmm import GaussianHMM
# from scipy import stats
from scipy.spatial import Voronoi, ConvexHull  # voronoi_plot_2d
# import sys
# import memsurfer

# import os
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

        # Altough sterols maybe do not play a larger role in the future for the domain identification it seems to be
        # a good idea to keep this functionality
        self.resid_selection_sterols = {}
        self.resid_selection_sterols_heads = {}

        # Next, a dictionary for EACH selected resid will be created. That's pretty much, but it is important to have
        # the order parameters for each lipid over the whole trajectory for the domain identification

        # Iterate over all residues in the selected membrane
        for resid in self.membrane_unique_resids:

            # Select specific resid
            resid_selection = self.universe.select_atoms(f"resid {resid}")
            # Get its lipid type
            resname = np.unique(resid_selection.resnames)[0]

            # Check leaflet assignment -> based on RESID
            # LEAFLET 0?
            if resid in self.leaflet_selection["0"].resids and resid not in self.leaflet_selection["1"].resids:

                # Init results for order parameters -> For each resid we should have an array containing the order
                # parameters for each frame
                setattr(self.results, f'id{resid}', {})  # -> Setup an empty dictionary
                getattr(self.results, f'id{resid}')['Leaflet'] = 0  # -> Store information about leaflet assignment
                getattr(self.results, f'id{resid}')['Resname'] = resname  # -> Store lipid type

                # Iterate over leaflet tails
                n_tails = len(self.tails[resname])
                for i in range(n_tails):
                    n_pairs = len(self.tails[resname][i]) // 2

                    # Init storage for P2 values for each lipid
                    getattr(self.results, f'id{resid}')[f'P2_{i}'] = np.zeros((self.n_frames, n_pairs),
                                                                              dtype=np.float32)

                # Store 3-D position of head group for each lipid
                getattr(self.results, f'id{resid}')[f'Heads'] = np.zeros((self.n_frames, 3), dtype=np.float32)

                # Store the area per lipid for each lipid
                getattr(self.results, f'id{resid}')[f'APL'] = np.zeros(self.n_frames, dtype=np.float32)

                # LEAFLET 1?
            elif resid in self.leaflet_selection["1"].resids and resid not in self.leaflet_selection["0"].resids:

                # Init results for order parameters -> For each resid we should have an array containing the order
                # parameters for each frame
                setattr(self.results, f'id{resid}', {})  # -> Setup an empty dictionary
                getattr(self.results, f'id{resid}')['Leaflet'] = 1  # -> Store information about leaflet assignment
                getattr(self.results, f'id{resid}')['Resname'] = resname  # -> Store lipid type

                # Iterate over leaflet tails
                n_tails = len(self.tails[resname])
                for i in range(n_tails):
                    n_pairs = len(self.tails[resname][i]) // 2

                    # Init storage for P2 values for each lipid
                    getattr(self.results, f'id{resid}')[f'P2_{i}'] = np.zeros((self.n_frames, n_pairs),
                                                                              dtype=np.float32)

                # Store 3-D position of head group for each lipid
                getattr(self.results, f'id{resid}')[f'Heads'] = np.zeros((self.n_frames, 3), dtype=np.float32)

                # Store the area per lipid for each lipid
                getattr(self.results, f'id{resid}')[f'APL'] = np.zeros(self.n_frames, dtype=np.float32)

                # STEROL?
            elif (resid not in self.leaflet_selection["0"].resids and resid not in self.leaflet_selection["1"].resids
                  and resname in self.sterols):

                # Sterols are not assigned to a specific leaflet -> They can flip. Maybe it is unlikely that it
                # happens in some membrane (especially atomistic ones) but it can happen and the code keeps track of
                # them.

                # Make a MDAnalysis atom selection for each resid. For the other lipids this was already done in the
                # LeafletAnalysisBase class
                self.resid_selection_sterols[str(resid)] = resid_selection.intersection(self.sterols_tail[resname])
                self.resid_selection_sterols_heads[str(resid)] = resid_selection.intersection(
                    self.sterols_head[resname])

                # Init results for order parameters -> For each resid we should have an array containing the order
                # parameters for each frame
                setattr(self.results, f'id{resid}', {})
                # Init storage array for leaflet assignment
                getattr(self.results, f'id{resid}')['Leaflet'] = np.zeros((self.n_frames), dtype=np.float32)
                # Resname of the sterol compound
                getattr(self.results, f'id{resid}')['Resname'] = resname
                # For sterol only one P2 value is calculated but for each frame
                getattr(self.results, f'id{resid}')['P2_0'] = np.zeros((self.n_frames), dtype=np.float32)
                # Also sterols can get an area per lipid assigned
                getattr(self.results, f'id{resid}')['APL'] = np.zeros((self.n_frames), dtype=np.float32)
                # Init storage array for head group position for each sterol
                getattr(self.results, f'id{resid}')[f'Heads'] = np.zeros((self.n_frames, 3), dtype=np.float32)

                # NOTHING?
            else:
                raise ValueError(f'{resname} with resid {resid} not found in leaflets or sterol list!')

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
        r /= np.sqrt(np.sum(r ** 2))

        # Dot product between membrane normal (z axis) and orientation vector
        dot_prod = np.dot(r, reference_axis)
        a = np.arccos(dot_prod)  # Angle in radians
        P2 = 0.5 * (3 * np.cos(a) ** 2 - 1)

        # Flip sign of order parameters
        # P2 = -1 * P2

        return P2

    # def get_p2_per_lipid(self, resid_tails_selection_leaflet, leaflet, resid_heads_selection_leaflet, local_normals,
    #                      refZ):
    #
    #     """
    #     Applies P2 calculation for each C-H pair in an individual lipid for each leaflet.
    #
    #     Parameters
    #     ----------
    #     resid_tails_selection_leaflet : dictionary
    #         Contains MDAnalysis atom selection for tail group of individual lipids per leaflet
    #     leaflet : int
    #         Leaflet of interest
    #     resid_heads_selection_leaflet : dictionary
    #         Contains MDAnalysis atom selection for head group of individual lipids per leaflet
    #     local_normals : dictionary
    #         Containing local normals per lipid -> keys are the resids
    #     refZ : bool
    #         Using the z-axis as reference axis or the local normal defined per lipid?
    #
    #
    #     """
    #
    #     # Iterate over resids in leaflet
    #     for key in resid_heads_selection_leaflet.keys():
    #
    #         # Check if leaflet is correct -> Sanity check
    #         assert getattr(self.results, f'id{key}')[
    #                    'Leaflet'] == leaflet, '!!!-----ERROR-----!!!\nWrong leaflet\n!!!-----ERROR-----!!!'
    #
    #         # Store head position -> Center of Mass of head group selection
    #         getattr(self.results, f'id{key}')[f'Heads'][self.index] = resid_heads_selection_leaflet[
    #             key].center_of_mass()
    #
    #         # Get resname
    #         rsn = getattr(self.results, f'id{key}')['Resname']
    #
    #         # Iterate over number of acyl chains in lipid named "rsn"
    #         for n_chain in range(len(self.tails[rsn])):
    #
    #             # self.tails[rsn][n_chain] contains atoms names in tail, if the input is correctly given it should look like this:
    #             # I.E. -> ["C1", "H1R", "C1", "H1S", ...]
    #
    #             # Iterate over these pairs -> I.E. ("C1","H1R"), ("C1", "H1S"), ... -> In this order the P2 values should be also stored
    #             for j in range(len(self.tails[rsn][n_chain]) // 2):
    #
    #                 if refZ:
    #                     getattr(self.results, f'id{key}')[f'P2_{n_chain}'][self.index, j] = self.calc_p2(
    #                         pair=resid_tails_selection_leaflet[str(key)][str(n_chain)][j],
    #                         reference_axis=np.array([0, 0, 1]))
    #                 else:
    #                     getattr(self.results, f'id{key}')[f'P2_{n_chain}'][self.index, j] = self.calc_p2(
    #                         pair=resid_tails_selection_leaflet[str(key)][str(n_chain)][j],
    #                         reference_axis=local_normals[f"{key}"])

    # def get_local_area_normal(self, leaflet, boxdim, periodic = True, exactness_level = 10):
    #
    #     """
    #     Calculate area per lipid and local membrane normal with MemSurfer library.
    #
    #     Parameters
    #     ----------
    #
    #     leaflet: int
    #         Top or bottom leaflet
    #     boxdim: np.array
    #         Box dimensions in x, y, z
    #     periodic: bool
    #         Usage of periodic boundary conditions during simulation
    #     exactness_level: int
    #         Approximating surface using Poisson reconstruction
    #     """
    #
    #     #Prepare box dimensions -> Seems to be used also for box width calculation (like bbox[1,:] - bbox[0, :]). First row where therefore the lower limit and second the upper limit
    #     bbox = np.zeros( (2, 3) )
    #     bbox[1, :] = boxdim
    #
    #     old_stdout = sys.stdout # backup current stdout
    #     sys.stdout = open(os.devnull, "w")
    #     mem = memsurfer.Membrane( points = self.surface_lipids_per_frame[ str(leaflet) ].positions,
    #                               labels = self.surface_lipids_per_frame[ str(leaflet) ].resids.astype("U"),
    #                               bbox = bbox,
    #                               periodic = periodic,
    #                               boundary_layer = 0.2 #Default value
    #                              )
    #
    #     #Put points back into box
    #     mem.fit_points_to_box_xy()
    #
    #
    #     #Approximate surface -> Uses as standard 18 k-neighbours for normal calculation
    #     mem.compute_approx_surface(exactness_level = exactness_level)
    #
    #     #Compute membrane surfaces based on the approximated surface calculated above:
    #     # - memb_planar := Planar projections of points on the smoothed surface
    #     # - memb_smooth := Points of the smoothed surface
    #     # - memb_exact  := Exact coordinates of lipids from trajectory
    #     mem.compute_membrane_surface()
    #
    #     local_normals = mem.memb_smooth.compute_normals()
    #
    #     local_area_per_lipid = mem.memb_smooth.compute_pointareas()
    #     sys.stdout = old_stdout # reset old stdout
    #
    #     local_normals_dict = dict( zip(mem.labels, local_normals) )
    #
    #     for resid, apl in zip(mem.labels, local_area_per_lipid): getattr(self.results, f'id{resid}')[f'APL'][self.index] = apl
    #
    #     return local_normals_dict

    def area_per_lipid_vor(self, coor_xy, bx, by):

        """
        Calculation of the area per lipid employing Voronoi tessellation on coordinates mapped to the xy plane.
        The function takes also the periodic boundary conditions of the box into account.

        Parameters
        ----------
        coor_xy : numpy.ndarray
            Mapped positions of the lipid headgroups
        bx : float
            Length of box vector in x direction
        by : float
            Length of box vector in y direction

        Returns
        -------
        apl : numpy.ndarray
            Area per lipid for each residue.
        vor : Voronoi Tesselation
            Scipy's Voronoi Diagram object
        """

        # Number of points in the plane
        ncoor = coor_xy.shape[0]
        # TODO - This is used for corrdinates = Is it headgroups ?
        #   self.surface_lipids_per_frame[str(leaflet)].positions,

        # Create periodic images of the coordinates
        # to take periodic boundary conditions into account
        pbc = np.zeros((9 * ncoor, 2), dtype=np.float32)

        # Iterate over all possible periodic images
        k = 0
        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                # Multiply the coordinates in a direction
                pbc[k * ncoor: (k + 1) * ncoor, 0] = coor_xy[:, 0] % bx + i * bx
                pbc[k * ncoor: (k + 1) * ncoor, 1] = coor_xy[:, 1] % by + j * by

                k += 1

        # Call scipy's Voronoi implementation
        # There is the (rare!) possibility that two points have the exact same xy positions,
        # to prevent issues at further calculation steps, the qhull_option "QJ" was employed to introduce small random
        # displacement of the points to resolve these issue.
        # TODO Decide Voronoi dimension. Keep in 2D or change to 3D
        vor = Voronoi(pbc, qhull_options="QJ")

        # Iterate over all members of the unit cell and calculate their occupied area
        apl = np.array([ConvexHull(vor.vertices[vor.regions[vor.point_region[i]]]).volume for i in range(ncoor)])

        return apl, vor

    def _single_frame(self):
        """
        Calculate data from a single frame of the trajectory.
        """

        # Make selection of non-flip/flop lipids and flip/flop lipids if there are sterols present

        self.surface_lipids_per_frame = {}

        # Iterate over leafelts
        for leaflet in range(2):

            self.surface_lipids_per_frame[str(leaflet)] = self.leaflet_selection[str(leaflet)]

            for sterol in self.sterols:
                self.surface_lipids_per_frame[str(leaflet)] += self.sterols_head[sterol].select_atoms(
                    " around 12 global group leaflet", leaflet=self.leaflet_selection[str(leaflet)])

        if self.surface_lipids_per_frame["0"].select_atoms("group leaf1",
                                                           leaf1=self.surface_lipids_per_frame["1"]):
            raise ValueError("Atoms in both leaflets!")

        # print(self.leaflet_selection[ str(leaflet) ].n_atoms)
        # print(self.surface_lipids_per_frame[ str(leaflet) ].n_atoms)

        # Get number of frame from trajectory
        self.frame = self.universe.trajectory.ts.frame
        # Calculate correct index if skipping step not equals 1 or start point not equals 0
        self.index = self.frame // self.step - self.start

        # ------------------------------ Local Normals/Area per Lipid ------------------------------------------------ #
        boxdim = self.universe.trajectory.ts.dimensions[0:3]
        # local_normals_dict_0 = self.get_local_area_normal(leaflet=0, boxdim=boxdim, periodic=True, exactness_level=10)
        # local_normals_dict_1 = self.get_local_area_normal(leaflet=1, boxdim=boxdim, periodic=True, exactness_level=10)
        # TODO Add Voronoi area calculation

        # ------------------------------ Order parameter ------------------------------------------------------------- #
        # self.get_p2_per_lipid(resid_tails_selection_leaflet=self.resid_tails_selection_0, leaflet=0,
        #                       resid_heads_selection_leaflet=self.resid_heads_selection_0,
        #                       local_normals=local_normals_dict_0, refZ=self.refZ)
        # self.get_p2_per_lipid(resid_tails_selection_leaflet=self.resid_tails_selection_1, leaflet=1,
        #                       resid_heads_selection_leaflet=self.resid_heads_selection_1,
        #                       local_normals=local_normals_dict_1, refZ=self.refZ)
        # TODO Add order parameter calculation

        # TODO Decide how to deal with weight matrix based on Voronoi diagram

        # Sterols
        for key, val in zip(self.resid_selection_sterols.keys(), self.resid_selection_sterols.values()):

            if key in self.surface_lipids_per_frame["0"].resids and key not in self.surface_lipids_per_frame[
                "1"].resids:
                # Check closest distance to leaflet
                getattr(self.results, f'id{key}')['Leaflet'][self.index] = 0

            elif key in self.surface_lipids_per_frame["1"].resids and key not in self.surface_lipids_per_frame[
                "0"].resids:
                # Check closest distance to leaflet
                getattr(self.results, f'id{key}')['Leaflet'][self.index] = 1

            elif key in self.surface_lipids_per_frame["0"].resids and key in self.surface_lipids_per_frame["1"].resids:
                print("!!!---WARNING---!!!")
                print(f"Cholesterol with ID {key} is in both leaflets!")

            elif key not in self.surface_lipids_per_frame["1"].resids and key not in self.surface_lipids_per_frame[
                "0"].resids:
                # Cholesterol is not assigned to a specific leaflet -> resides in the mid of the membrane
                getattr(self.results, f'id{key}')['Leaflet'][self.index] = 2

            else:
                ValueError("Cholesterol isn't anywhere! That shouldn't happen! Time for Scully and Moulder!")

            # Store head position
            getattr(self.results, f'id{key}')[f'Heads'][self.index] = self.resid_selection_sterols_heads[
                str(key)].center_of_mass()

            getattr(self.results, f'id{key}')[f'P2_0'][self.index] = self.calc_p2(
                pair=self.resid_selection_sterols[str(key)], reference_axis=np.array([0, 0, 1]))

    def _conclude(self):

        """
        Calculate the final results of the analysis

        Extract the obtained data and put them into a clear and accessible data structure
        """

        # -----------------------------------------------------------------------
        # Make a dictionary for the calculated values of each lipid type for each leaflet
        # -----------------------------------------------------------------------

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

        # Initialize storage dictionary

        self.results.p2_per_type = {}
        self.results.apl_per_type = {}

        # Iterate over leaflets -> 0 top, 1 bottom
        for i in range(2):

            # Make dictionary for each leaflet
            self.results.p2_per_type[f"Leaf{i}"] = {}
            self.results.apl_per_type[f"Leaf{i}"] = {}

            # Iterate over resnames in each leaflet
            for rsn in np.unique(self.leaflet_selection[str(i)].resnames):

                # Iterate over number of acyl chains in lipid named "rsn"
                for n_chain in range(len(self.tails[rsn])):
                    # Make a list for each acyl chain in resn
                    self.results.p2_per_type[f"Leaf{i}"][f"{rsn}_{n_chain}"] = []

                self.results.apl_per_type[f"Leaf{i}"][f"{rsn}"] = []

        # -------------------------------------------------------------

        # Fill dictionary with obtained data

        # Iterate over all residues in the selected membrane
        for resid in self.membrane_unique_resids:

            # Grab leaflet and resname
            leaflet = getattr(self.results, f'id{resid}')['Leaflet']
            rsn = getattr(self.results, f'id{resid}')['Resname']

            # Check if lipid is a sterol compound or not
            if rsn not in self.sterols:

                # Iterate over chains -> For a normal phospholipid that should be 2
                for n_chain in range(len(self.tails[rsn])):
                    # Get individual lipid p2 values for corresponding chain
                    indv_p2 = getattr(self.results, f'id{resid}')[f'P2_{n_chain}']

                    # Add it to the lipid type list
                    self.results.p2_per_type[f"Leaf{leaflet}"][f"{rsn}_{n_chain}"].append(indv_p2)

                # Get area per lipid for specific residue
                apl = getattr(self.results, f'id{resid}')['APL']
                self.results.apl_per_type[f"Leaf{leaflet}"][f"{rsn}"].append(apl)

            elif rsn in self.sterols:
                pass

            # NOTHING?
            else:
                raise ValueError(f'{rsn} with resid {resid} not found in leaflets or sterol list!')

        # -------------------------------------------------------------

        # Transform lists to arrays

        # Iterate over leaflets
        for i in range(2):

            # Iterate over lipid in leaflet
            for rsn in np.unique(self.leaflet_selection[str(i)].resnames):

                # Check for sterol compound
                if rsn not in self.sterols:

                    # Iterate over chain
                    for n_chain in range(len(self.tails[rsn])):
                        # Transform list to array
                        self.results.p2_per_type[f"Leaf{i}"][f"{rsn}_{n_chain}"] = np.array(
                            self.results.p2_per_type[f"Leaf{i}"][f"{rsn}_{n_chain}"])

                    # Just transform for area per lipid
                    self.results.apl_per_type[f"Leaf{i}"][f"{rsn}"] = np.array(
                        self.results.apl_per_type[f"Leaf{i}"][f"{rsn}"])

        # -------------------------------------------------------------
        # -------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # Make a dictionary with averaged P2 values per C-H2 (or C-H) group PER chain
        # ---------------------------------------------------------------------------

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

        # Iterate over leaflets
        for leaf_key, leaf in zip(self.results.p2_per_type.keys(), self.results.p2_per_type.values()):

            self.results.mean_p2_per_type[leaf_key] = {}

            # Iterate over lipid types
            for key, val in zip(self.tails.keys(), self.tails.values()):

                # Iterate over chains for each lipid type
                for i, chain in enumerate(val):

                    # Check if lipid type is in leaflet
                    if f"{key}_{i}" in leaf.keys():

                        # Get all pairs in chain
                        pairs_in_chain = np.array_split(chain, len(chain) // 2)

                        # Adding a dummy array ensures that double bonds at the end of an acyl chain are taken
                        # into account
                        pairs_in_chain += [np.array(["dummy", "dummy"])]

                        n_pairs = len(pairs_in_chain)

                        order_per_chain = []

                        # Iterate over pairs
                        for j in range(n_pairs - 2 + 1):

                            # Check if a pair has the same aliphatic C-Atom

                            # If so -> Calculate the average (i.e. C1-H1S and C1-H1R)
                            # I transpose the resulting arrays several times to get a more logical shape of the
                            # resulting array
                            if pairs_in_chain[j][0] == pairs_in_chain[j + 1][0]:
                                order_per_chain.append(leaf[f"{key}_{i}"][:, :, j:j + 2].mean(-1).T)

                            # If there is a C-Atom UNEQUAL to the former AND the following C-Atom -> Assume double bond
                            # -> No average over pairs

                            # Edge case:
                            # j = 0 -> j-1 = -1 
                            # Should not matter since latest atom in aliphatic name is named differently than first one
                            # -> Should also work for double bonds at the first place of the
                            elif pairs_in_chain[j][0] != pairs_in_chain[j + 1][0] and pairs_in_chain[j][0] != \
                                    pairs_in_chain[j - 1][0]:
                                order_per_chain.append(leaf[f"{key}_{i}"][:, :, j].T)

                            # If just the following C-Atom is unequal pass on
                            elif pairs_in_chain[j][0] != pairs_in_chain[j + 1][0]:
                                pass

                            else:
                                raise ValueError(
                                    f"Something odd in merging order parameters for {key} in chain {i} per CH2 happened!")

                        self.results.mean_p2_per_type[leaf_key][f"{key}_{i}"] = np.array(order_per_chain).T

                    else:
                        pass

        # -------------------------------------------------------------
        # -------------------------------------------------------------

        # ---------------------------------------------
        # Make a dictionary with values for cholesterol
        # ---------------------------------------------

        """
        Sterol are able to flip-flop between leaflets, they are there for special treated and have their own output data
         structure

        - Sterols
            - SterolA
                - Leaf
                    - Leaflet assignments (NSterols, NFrames)
                - P2
                    - P2 values (NSterols, NFrames)
            - SterolB
                - Leaf
                    - ...
            - ...
        """

        self.results["Sterols"] = {}

        for sterol in self.sterols: self.results["Sterols"][sterol] = {"Leaf": [], "P2": []}

        for key, val in zip(self.resid_selection_sterols.keys(), self.resid_selection_sterols.values()):
            rsn = getattr(self.results, f'id{resid}')['Resname']

            self.results["Sterols"][rsn]["Leaf"].append(getattr(self.results, f'id{key}')['Leaflet'])
            self.results["Sterols"][rsn]["P2_0"].append(getattr(self.results, f'id{key}')[f'P2_0'])

        for sterol in self.sterols:
            self.results["Sterols"][sterol]["Leaf"] = np.array(self.results["Sterols"][sterol]["Leaf"])
            self.results["Sterols"][sterol]["P2"] = np.array(self.results["Sterols"][sterol]["P2"])

    # ------------------------------ FIT GAUSSIAN MIXTURE MODEL ------------------------------------------------------ #
    def GMM(self, n_repeats, start_frame, gmm_kwargs={}):

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

        # Iterate over leaflets
        for idx, leafgroup in zip(self.leaflet_selection.keys(), self.leaflet_selection.values()):
            # Init empty dictionary for each leaflet
            self.results["GMM"][f"Leaf{idx}"] = {}

        self.get_gmm_order_parameters(leaflet=0, n_repeats=n_repeats, start_frame=start_frame, gmm_kwargs=gmm_kwargs)
        self.get_gmm_order_parameters(leaflet=1, n_repeats=n_repeats, start_frame=start_frame, gmm_kwargs=gmm_kwargs)

        self.get_gmm_area_per_lipid(leaflet=0, n_repeats=n_repeats, start_frame=start_frame, gmm_kwargs=gmm_kwargs)
        self.get_gmm_area_per_lipid(leaflet=1, n_repeats=n_repeats, start_frame=start_frame, gmm_kwargs=gmm_kwargs)

    def get_gmm_order_parameters(self, n_repeats, leaflet, start_frame, gmm_kwargs):

        # Get lipid types in leaflet
        leaflet_resnames = np.unique(self.leaflet_resids[str(leaflet)].resnames)

        # Iterate over lipids in leaflet
        for rsn in leaflet_resnames:

            if rsn in self.sterols: continue

            # Iterate over tails (e.g. for standard phospholipids that 2)
            for i, tail in enumerate(self.tails[rsn]):
                self.results["GMM"][f"Leaf{leaflet}"][f"{rsn}_{i}"] = self.fit_gmm(
                    property_=self.results.mean_p2_per_type[f"Leaf{leaflet}"][f"{rsn}_{i}"].mean(2),
                    n_repeats=n_repeats, start_frame=start_frame, gmm_kwargs=gmm_kwargs)

    def get_gmm_area_per_lipid(self, n_repeats, leaflet, start_frame, gmm_kwargs):

        # Get lipid types in leaflet
        leaflet_resnames = np.unique(self.leaflet_resids[str(leaflet)].resnames)

        # Iterate over lipids in leaflet
        for rsn in leaflet_resnames:

            if rsn in self.sterols: continue

            self.results["GMM"][f"Leaf{leaflet}"][f"{rsn}_APL"] = self.fit_gmm(
                property_=self.results.apl_per_type[f"Leaf{leaflet}"][f"{rsn}"], n_repeats=n_repeats,
                start_frame=start_frame, gmm_kwargs=gmm_kwargs, apl=True)

    def fit_gmm(self, property_, gmm_kwargs, n_repeats, start_frame, apl=False):

        """
        Fit a Gaussian Mixture Model for each lipid type to the results of the property calculation.
        This is done here for each leaflet separately!


        Parameters
        ----------
        property_ : numpy.array
            Input data for the gaussian mixture model ( Shape: (NLipids, NFrames) )


        """

        assert self.n_frames == property_.shape[1], "Wrong input shape for the fitting of the GMM!"

        # ---------------------------------------Prep data---------------------------------------#

        # Take arithmetic mean over chain order parameters
        property_flatten = property_[:,start_frame:].flatten() # Shape change (NLipids, NFrames) -> (NLipids * NFrames,)

        # ---------------------------------------Gaussian Mixture---------------------------------------#

        # Run the GaussianMixture Model for two components
        best_score = -np.inf

        for n in tqdm(range(n_repeats)):

            GM_n = mixture.GaussianMixture(n_components=2, **gmm_kwargs).fit(property_flatten.reshape((-1, 1)))

            score_n = GM_n.score(property_.flatten().reshape((-1, 1)))

            if score_n > best_score:
                GM = GM_n
                best_score = score_n

            del GM_n

        # ---------------------------------------Gaussian Mixture Results---------------------------------------#

        if apl == False:
            # The Gaussian distribution with the highest mean corresponds to the ordered state
            param_o = np.argmax(GM.means_)
            # The Gaussian distribution with the lowest mean corresponds to the disordered state
            param_d = np.argmin(GM.means_)
        else:
            # The Gaussian distribution with the lowest mean corresponds to the ordered state
            param_o = np.argmin(GM.means_)
            # The Gaussian distribution with the highest mean corresponds to the disordered state
            param_d = np.argmax(GM.means_)

        # Get mean and variance of the fitted Gaussian distributions
        mu_o, var_o, weights_o = GM.means_[param_o], GM.covariances_[param_o][0], GM.weights_[param_o]
        mu_d, var_d, weights_d = GM.means_[param_d], GM.covariances_[param_d][0], GM.weights_[param_d]

        sig_o = np.sqrt(var_o)
        sig_d = np.sqrt(var_d)

        # ---------------------------------------Intermediate Distribution---------------------------------------#
        mu_I = (sig_d * mu_o + sig_o * mu_d) / (sig_d + sig_o)
        sig_I = np.min([np.abs(mu_o - mu_I), np.abs(mu_d - mu_I)]) / 3
        var_I = sig_I ** 2
        weights_I = (weights_d + weights_o) / 2

        # ----------------------------------------Conclude----------------------------------------#
        # Put the fitted results in an easy to access format
        fit_results = np.empty((3, 3), dtype=np.float32)

        fit_results[0, 0], fit_results[0, 1], fit_results[0, 2] = mu_d, var_d, weights_d
        fit_results[1, 0], fit_results[1, 1], fit_results[1, 2] = mu_I, var_I, weights_I
        fit_results[2, 0], fit_results[2, 1], fit_results[2, 2] = mu_o, var_o, weights_o

        return fit_results

    # ------------------------------ HIDDEN MARKOW MODEL ------------------------------------------------------------- #

    def HMM(self, n_repeats, start_frame, hmm_kwargs={}):

        if "GMM" not in self.results.keys() or len(self.results["GMM"]) == 0:
            print("!!!---WARNING---!!!")
            print("No Gaussian Mixture Model data found! Please run GMM first!")
            return

        else:
            pass

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

        # Iterate over leaflets
        for idx, leafgroup in zip(self.leaflet_selection.keys(), self.leaflet_selection.values()):
            # Init empty dictionary for each leaflet
            self.results["HMM"][f"Leaf{idx}"] = {}

        self.get_hmm_order_parameters(leaflet=0, n_repeats=n_repeats, start_frame=start_frame, hmm_kwargs=hmm_kwargs)
        self.get_hmm_order_parameters(leaflet=1, n_repeats=n_repeats, start_frame=start_frame, hmm_kwargs=hmm_kwargs)

        self.get_hmm_area_per_lipid(leaflet=0, n_repeats=n_repeats, start_frame=start_frame, hmm_kwargs=hmm_kwargs)
        self.get_hmm_area_per_lipid(leaflet=1, n_repeats=n_repeats, start_frame=start_frame, hmm_kwargs=hmm_kwargs)

    def get_hmm_order_parameters(self, leaflet, n_repeats, start_frame, hmm_kwargs):

        # Get lipid types in leaflet
        leaflet_resnames = np.unique(self.leaflet_resids[str(leaflet)].resnames)

        # Iterate over lipids in leaflet
        for rsn in leaflet_resnames:

            if rsn in self.sterols: continue

            # Iterate over tails (e.g. for standard phospholipids that 2)
            for i, tail in enumerate(self.tails[rsn]):
                self.results["HMM"][f"Leaf{leaflet}"][f"{rsn}_{i}"] = self.fit_hmm(
                    property_=self.results.mean_p2_per_type[f"Leaf{leaflet}"][f"{rsn}_{i}"].mean(2),
                    init_params=self.results.GMM[f"Leaf{leaflet}"][f"{rsn}_{i}"],
                    n_repeats=n_repeats,
                    start_frame=start_frame,
                    hmm_kwargs=hmm_kwargs)

    def get_hmm_area_per_lipid(self, leaflet, n_repeats, start_frame, hmm_kwargs):

        # Get lipid types in leaflet
        leaflet_resnames = np.unique(self.leaflet_resids[str(leaflet)].resnames)

        # Iterate over lipids in leaflet
        for rsn in leaflet_resnames:

            if rsn in self.sterols: continue

            self.results["HMM"][f"Leaf{leaflet}"][f"{rsn}_APL"] = self.fit_hmm(
                property_=self.results.apl_per_type[f"Leaf{leaflet}"][f"{rsn}"],
                init_params=self.results.GMM[f"Leaf{leaflet}"][f"{rsn}_APL"],
                n_repeats=n_repeats,
                start_frame=start_frame,
                hmm_kwargs=hmm_kwargs)

    def fit_hmm(self, property_, init_params, n_repeats, start_frame, hmm_kwargs):

        assert self.n_frames == property_.shape[1], "Wrong input shape for the fitting of the HMM!"

        n_lipids = property_.shape[0]

        means_ = init_params[:, 0].reshape(3, -1)
        vars_ = init_params[:, 1].reshape(3, -1)
        weights_ = init_params[:, 2].reshape(3, -1)

        best_score = -np.inf

        for i in tqdm(range(n_repeats)):

            GHMM_n = GaussianHMM(n_components=3, means_prior=means_, covars_prior=vars_, **hmm_kwargs)

            GHMM_n.fit(property_[:, start_frame:].flatten().reshape(-1, 1),
                       lengths=np.repeat(self.n_frames - start_frame, n_lipids))

            score_n = GHMM_n.score(property_[:, start_frame:].flatten().reshape(-1, 1),
                                   lengths=np.repeat(self.n_frames - start_frame, n_lipids))

            if score_n > best_score:
                best_score = score_n
                GHMM = GHMM_n

            del GHMM_n

        return GHMM

    def predict_states(self):

        for resid in self.memsele.resids:

            rsn = getattr(self.results, f'id{resid}')["Resname"]
            leaflet = getattr(self.results, f'id{resid}')["Leaflet"]

            if rsn not in self.sterols:

                # ---------------------------------------------------------P2 Prediction---------------------------------------------------------#
                # Iterate over tails (e.g. for standard phospholipids that 2)
                for i, tail in enumerate(self.tails[rsn]):
                    X = getattr(self.results, f'id{resid}')[f"P2_{i}"].mean(1)

                    sorted_means = np.argsort(self.results["HMM"][f"Leaf{leaflet}"][f"{rsn}_{i}"].means_[:, 0])

                    predX = self.results["HMM"][f"Leaf{leaflet}"][f"{rsn}_{i}"].predict(X.reshape(-1, 1))

                    re_predX = np.array([sorted_means[predX_i] for predX_i in predX])

                    getattr(self.results, f'id{resid}')[f"Pred_P2_{i}"] = re_predX

                # ---------------------------------------------------------APL Prediction---------------------------------------------------------#
                X = getattr(self.results, f'id{resid}')[f"APL"]

                sorted_means = np.argsort(-1 * self.results["HMM"][f"Leaf{leaflet}"][f"{rsn}_APL"].means_[:, 0])

                predX = self.results["HMM"][f"Leaf{leaflet}"][f"{rsn}_APL"].predict(X.reshape(-1, 1))

                re_predX = np.array([sorted_means[predX_i] for predX_i in predX])

                getattr(self.results, f'id{resid}')["Pred_APL"] = re_predX

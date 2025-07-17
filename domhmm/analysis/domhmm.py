"""
PropertyCalculation --- :mod:`domhmm.analysis.PropertyCalculation`
===========================================================

This module contains the :class:`PropertyCalculation` class.

"""

import logging as log
import sys

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.sparse import csr_array
from scipy.spatial import Voronoi, ConvexHull
from scipy import stats
from sklearn import mixture
from tqdm import tqdm

from .base import LeafletAnalysisBase


class PropertyCalculation(LeafletAnalysisBase):
    """
    The DirectorOrder class calculates an order parameter for each selected lipid according to the formula:

        P2 = 0.5 * (3 * cos(a)^2 - 1), (1)

    where an is the angle between a pre-defined director in the lipid (e.g. C-H or CC) and a reference axis.

    """

    def _prepare(self):
        """
        Prepare results array before the calculation of the height spectrum begins

        Attributes
        ----------
        """
        if self.verbose:
            log.basicConfig(level=log.INFO, stream=sys.stdout)
        log.info("Preparation Step")
        # Initialize result storage dictionaries
        self.results.train_data_per_type = {}
        self.results.GMM = {}
        self.results.HMM = {}
        self.results.HMM_Pred = {}
        self.results.Getis_Ord = {}
        self.max_tail_len = -1
        for _, each in self.tails.items():
            if len(each) > self.max_tail_len:
                self.max_tail_len = len(each)
        # Total number of parameters are area per lipid + order parameter for each tail = max_tail_len + 1
        self.results.train = np.zeros(
            (len(self.membrane_unique_resids), self.n_frames, 1 + self.max_tail_len + len(self.sterol_heads)),
            dtype=np.float32)
        # Initalize weight matrix storage for each leaflet.
        setattr(self.results, "upper_weight_all", [])
        setattr(self.results, "lower_weight_all", [])

        # Initialized leaflet assignment array for each frame
        self.leaflet_assignment = np.zeros((len(self.membrane_unique_resids), self.n_frames), dtype=np.int32)

    @staticmethod
    def calc_order_parameter(chain):

        """
        Calculate average Scc order parameters per acyl chain according to the equation:

        S_cc = (3 * cos( theta )^2 - 1) / 2,

        where theta describes the angle between the z-axis of the system and the vector between two subsequent tail
        beads.

        Parameters
        ----------
        chain : Selection
            MDAnalysis Selection object

        Returns
        -------
        s_cc : numpy.ndarray
            Mean S_cc parameter for the selected chain of each residue
        """

        # Separate the coordinates according to their residue index
        ridx = np.where( np.abs(np.diff(chain.resids)) > 0)[0] + 1

        pos = np.split(chain.positions, ridx)

        # Calculate the normalized orientation vector between two subsequent tail beads
        vec = [np.diff(pos_i, axis=0) for pos_i in pos]

        vec_norm = [np.sqrt((vec_i ** 2).sum(axis=-1)) for vec_i in vec]

        vec = [vec_i / vec_norm_i.reshape(-1, 1) for vec_i, vec_norm_i in zip(vec, vec_norm)]
        # TODO z axis multiplication inside. It needs to be changed in future work
        # Choose the z-axis as membrane normal and take care of the machine precision
        dot_prod = [np.clip(vec_i[:, 2], -1., 1.) for vec_i in vec]

        # Calculate the order parameter
        s_cc = np.array([np.mean(0.5 * (3 * dot ** 2 - 1)) for dot in dot_prod])

        return s_cc

    def order_parameter(self):
        """
        Calculation of scc order parameter for each chain of each residue.
        """
        # Iterate over each tail with chain id
        for chain, tail in self.resid_tails_selection.items():
            # SCC calculation
            s_cc = self.calc_order_parameter(tail)
            idx = self.get_residue_idx(self.resids_index_map, np.unique(tail.resids))
            self.results.train[idx, self.index, 1 + chain] = s_cc
        for i, (resname, tail) in enumerate(self.sterol_tails_selection.items()):
            s_cc = self.calc_order_parameter(tail)
            idx = self.get_residue_idx(self.resids_index_map, np.unique(tail.resids))
            self.results.train[idx, self.index, 1 + self.max_tail_len + i] = s_cc

    @staticmethod
    def area_per_lipid_vor(coor_xy, boxdim, frac):

        """
        Calculation of the area per lipid employing Voronoi tessellation on coordinates mapped to the xy plane.
        The function takes also the periodic boundary conditions of the box into account.

        Parameters
        ----------
        coor_xy : numpy.ndarray
            Coordinates of upper/lower leaflet
        boxdim : numpy.ndarray
            Length of box vectors in all directions
        frac : float
            Fraction of box length in x and y outside the unit cell considered for Voronoi calculation 

        Returns
        -------
        vor : Voronoi Tesselation
            Scipy's Voronoi Diagram object
        apl : Area per lipid
            Area per lipid based on Scipy's Voronoi Diagram
        pbc_idx : Indices
            Unit cell indices for periodic image coordinates
        """

        # Number of points in the plane
        ncoor = coor_xy.shape[0]
        bx = boxdim[0]
        by = boxdim[1]
        # Create periodic images of the coordinates
        # to take periodic boundary conditions into account

        # Store coordinates of periodic images
        pbc = np.zeros((9 * ncoor, 2), dtype=np.float32)
        # Store unit cell indices for coordinates of periodic images
        pbc_idx = np.arange(9 * ncoor, dtype=np.int64) % ncoor

        # Iterate over all possible periodic images
        k = 0
        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                # Multiply the coordinates in a direction
                pbc[k * ncoor: (k + 1) * ncoor, 0] = coor_xy[:, 0] % bx + i * bx
                pbc[k * ncoor: (k + 1) * ncoor, 1] = coor_xy[:, 1] % by + j * by

                k += 1

        # Create a boolean mask for positions within the unit cell and a smaller fraction of positions outside the
        # unit cell
        # Check along x-axis
        f0 = pbc[:, 0] >= -frac * bx
        f1 = pbc[:, 0] <= bx + frac * bx

        # Check along y-axis
        f2 = pbc[:, 1] >= -frac * by
        f3 = pbc[:, 1] <= by + frac * by

        # Merge the four masks together
        mask = f0 * f1 * f2 * f3

        # Filter positions and unit cell indices of all periodic images
        pbc = pbc[mask]
        pbc_idx = pbc_idx[mask]

        # Call scipy's Voronoi implementation
        # There is the (rare!) possibility that two points have the exact same xy positions,
        # to prevent issues at further calculation steps, the qhull_option "QJ" was employed to introduce small random
        # displacement of the points to resolve these issue.

        # IMPORTANT: The use of "QJ" makes the resulting Voronoi diagram depending on frac. Values for ridge lengths
        # can vary!
        vor = Voronoi(pbc, qhull_options="QJ")

        # Iterate over all members of the unit cell and calculate their occupied area
        apl = np.array([ConvexHull(vor.vertices[vor.regions[vor.point_region[i]]]).volume for i in range(ncoor)])

        return vor, apl, pbc_idx

    @staticmethod
    def weight_matrix(vor, pbc_idx, coor_xy):

        """
        Calculate the weight factors between neighbored lipid pairs based on a Voronoi tessellation.

        Parameters
        ----------
        vor : Voronoi Tesselation
            Scipy's Voronoi Diagram object
        pbc_idx : Indices
            Unit cell indices of periodic image coordinates
        coor_xy : numpy.ndarray
            Coordinates of upper/lower leaflet

        Returns
        -------
        weight_matrix : numpy.ndarray
            Weight factors wij between neighbored lipid pairs.
            Is 0 if lipids are not directly neighbored.
        """

        # Number of points in the plane
        ncoor = coor_xy.shape[0]

        # Calculate the distance for all pairs of points between which a ridge exists
        dij = vor.points[vor.ridge_points[:, 0]] - vor.points[vor.ridge_points[:, 1]]
        dij = np.sqrt(np.sum(dij ** 2, axis=1))

        # There is the (rare!) possibility that two points have the exact same xy positions,
        # to prevent issues at further calculation steps, their distance is set to a very small
        # distance of 4.7 Angstrom (2 times the VdW radius of a regular bead in MARTINI3)
        dij[dij < 1E-5] = 4.7

        # Calculate the distance for all pairs of vertices connected via a ridge
        vert_idx = np.array(vor.ridge_vertices)
        bij = vor.vertices[vert_idx[:, 0]] - vor.vertices[vert_idx[:, 1]]
        bij = np.sqrt(np.sum(bij ** 2, axis=1))

        # INFO: vor.ridge_points and vor.ridge_vertices should be sorted -> Check vor.ridge_dict

        # Calculate weight factor
        wij = bij / dij

        # Set up an empty array to store the weight factors for each lipid
        weight_matrix = np.zeros((ncoor, ncoor))

        # Select all indices of ridges that contain members of the unit cell
        mask_unit_cell = np.logical_or(vor.ridge_points[:, 0] < ncoor, vor.ridge_points[:, 1] < ncoor)

        # Apply the modulus operator since some of the indices in "unit_cell_point" will point to coordinates outside
        # the unit cell. Applying the modulus operator "%" will allow an indexing of the "weight_matrix". However, some
        # of the indices in "unit_cell_point" will be doubled that shouldn't be an issue since the same weight factor is
        # then just put several times in the same entry of the array (no summing or something similar!)
        # Previous: unit_cell_point = vor.ridge_points[mask_unit_cell] % ncoor

        # Transform the indices in vor.ridge_points back to unit cell indices -> Filter than for all indices of
        # ridges that contain members of the unit cell
        unit_cell_point = pbc_idx[vor.ridge_points][mask_unit_cell]

        weight_matrix[unit_cell_point[:, 0], unit_cell_point[:, 1]] = wij[mask_unit_cell]
        weight_matrix[unit_cell_point[:, 1], unit_cell_point[:, 0]] = wij[mask_unit_cell]
        return weight_matrix

    def _single_frame(self):
        """
        Calculate data from a single frame of the trajectory.
        """

        # Get number of frame from trajectory
        self.frame = self.universe.trajectory.ts.frame
        # Calculate correct index if skipping step not equals 1 or start point not equals 0
        self.index = (self.frame - self.start) // self.step

        # Update leaflet assignment (if leaflet_frame_rate is None, leaflets will never get updated during analysis)
        if self.lipid_leaflet_rate is not None and not self.index % self.lipid_leaflet_rate:
            # Call leaflet assignment functions for non-sterol and sterol compounds
            self.leaflet_selection_no_sterol = self.get_leaflets()
            self.leaflet_selection = self.get_leaflets_sterol()

            # Write assignments to array
            assignment_index = int(self.index / self.lipid_leaflet_rate)
            start_index = assignment_index * self.lipid_leaflet_rate
            end_index = (assignment_index + 1) * self.lipid_leaflet_rate
            if end_index > self.leaflet_assignment.shape[1]:
                end_index = self.leaflet_assignment.shape[1]

            self.uidx = self.get_residue_idx(self.resids_index_map, self.leaflet_selection["0"].resids)
            self.lidx = self.get_residue_idx(self.resids_index_map, self.leaflet_selection["1"].resids)

            self.leaflet_assignment[self.lidx, start_index:end_index] = 1
            self.leaflet_assignment[self.uidx, start_index:end_index] = 0
            self.leaflet_assignment[self.lidx, start_index:end_index] = 1

        # Update sterol assignment. Don't do the update if it was already done in the if-statement before
        if not self.index % self.sterol_leaflet_rate and (
                self.lipid_leaflet_rate is None or self.index % self.lipid_leaflet_rate):
            # Call leaflet assignment function for sterol compounds
            self.leaflet_selection = self.get_leaflets_sterol()

            # Write assignments to array
            assignment_index = int(self.index / self.sterol_leaflet_rate)
            start_index = assignment_index * self.sterol_leaflet_rate
            end_index = (assignment_index + 1) * self.sterol_leaflet_rate
            if end_index > self.leaflet_assignment.shape[1]:
                end_index = self.leaflet_assignment.shape[1]

            self.uidx = self.get_residue_idx(self.resids_index_map, self.leaflet_selection["0"].resids)
            self.lidx = self.get_residue_idx(self.resids_index_map, self.leaflet_selection["1"].resids)

            self.leaflet_assignment[self.uidx, start_index:end_index] = 0
            self.leaflet_assignment[self.lidx, start_index:end_index] = 1

        if self.leaflet_selection["0"].select_atoms("group leaf1", leaf1=self.leaflet_selection["1"]):
            raise ValueError("Atoms in both leaflets !")

        # ------------------------------ Local Normals/Area per Lipid ------------------------------------------------ #
        boxdim = self.universe.trajectory.ts.dimensions[0:3]
        upper_coor_xy = self.leaflet_selection[str(0)].positions
        lower_coor_xy = self.leaflet_selection[str(1)].positions
        # Check Transmembrane domain existence
        if self.tmd_protein is not None:
            tmd_upper_coor_xy = self.tmd_protein["0"]
            # Check if dimension of coordinates is same in both array
            if tmd_upper_coor_xy.shape[1] == upper_coor_xy.shape[1]:
                upper_coor_xy = np.append(upper_coor_xy, tmd_upper_coor_xy, axis=0)
            tmd_lower_coor_xy = self.tmd_protein["1"]
            # Check if dimension of coordinates is same in both array
            if tmd_lower_coor_xy.shape[1] == lower_coor_xy.shape[1]:
                lower_coor_xy = np.append(lower_coor_xy, tmd_lower_coor_xy, axis=0)
        upper_vor, upper_apl, upper_pbc_idx = self.area_per_lipid_vor(coor_xy=upper_coor_xy, boxdim=boxdim,
                                                                      frac=self.frac)
        lower_vor, lower_apl, lower_pbc_idx = self.area_per_lipid_vor(coor_xy=lower_coor_xy, boxdim=boxdim,
                                                                      frac=self.frac)
        if self.tmd_protein is not None:
            # Remove TMD Protein numbers from training data
            upper_apl = np.array(upper_apl[:-len(tmd_upper_coor_xy)])
            lower_apl = np.array(lower_apl[:-len(tmd_lower_coor_xy)])
        self.results.train[self.uidx, self.index, 0] = upper_apl
        self.results.train[self.lidx, self.index, 0] = lower_apl
        # TODO Local normal calculation

        # ------------------------------ Order parameter ------------------------------------------------------------- #
        self.order_parameter()
        # ------------------------------ Weight Matrix --------------------------------------------------------------- #
        upper_weight_matrix = self.weight_matrix(upper_vor, pbc_idx=upper_pbc_idx, coor_xy=upper_coor_xy)
        lower_weight_matrix = self.weight_matrix(lower_vor, pbc_idx=lower_pbc_idx, coor_xy=lower_coor_xy)
        if self.tmd_protein is not None:
            # Remove TMD Protein numbers from weight matrix
            upper_weight_matrix = upper_weight_matrix[:-len(tmd_upper_coor_xy), :-len(tmd_upper_coor_xy)]
            lower_weight_matrix = lower_weight_matrix[:-len(tmd_lower_coor_xy), :-len(tmd_lower_coor_xy)]

        # Keep weight matrices in scipy.sparse.csr_array format since both is sparse matrices
        self.results["upper_weight_all"].append(csr_array(upper_weight_matrix))
        self.results["lower_weight_all"].append(csr_array(lower_weight_matrix))

    def _conclude(self):
        """
        Calculate the final results of the analysis

        Extract the obtained data and put them into a clear and accessible data structure
        """
        log.info("Conclusion step is starting.")
        self.prepare_train_data()
        # -------------------------------------------------------------

        # Check for user-provided HMMs
        if not any(self.trained_hmms):
            # No user-provided HMMs, perform usual workflow
            log.info("Gaussian Mixture Model training is starting.")
            self.GMM(gmm_kwargs=self.gmm_kwargs)

            log.info("Hidden Markov Model training is starting.")
            self.HMM(hmm_kwargs=self.hmm_kwargs)
        else:
            # User-provided HMMs found, use them!
            log.info("No Gaussian Mixture Model is trained.")
            log.info("No Hidden Markov Model is trained. Instead use the user-provided HMMs.")

            # Use the provided dictionary directly, the checks for validity were already done before
            self.results['HMM'] = self.trained_hmms

            # Same workflow as in self.HMM()
            if self.result_plots:
                # Plot result of hmm
                self.plot_hmm_result()

            # Make predictions based on HMM model
            self.predict_states()

            # Validate states and result prediction
            self.state_validate()
            if self.result_plots:
                # Plot prediction result
                self.predict_plot()

        log.info("Getis-Ord Statistic calculation is starting.")

        self.getis_ord()

        # The user argument do_clustering decides if the hierarchical clustering is (not) performed
        if not self.do_clustering:
            pass
        else:
            log.info("Clustering is starting.")
            self.result_clustering()

            if self.result_plots:
                self.clustering_plot()

        print("It's done! DomHMM finished successfully. Have a nice day and enjoy your data! :-)")

    def prepare_train_data(self):
        """
        Prepare train data for GMM and HMM while separating self.results.train with respect to each residue
        """
        self.results.train_data_per_type = {}
        resid_dict = {}
        for resname in self.unique_resnames:
            idx = self.get_residue_idx(self.resids_index_map, self.residue_ids[resname])
            resid_dict[resname] = idx
            self.results.train_data_per_type[f"{resname}"] = [[] for _ in range(3)]
        if self.asymmetric_membrane:
            # For asymmetric membranes, models will be trained for each leaflets' each residue type
            leaflet_residx = {}
            # leaflet_train_residx contains residue indexes that are not too flip-floppy for model training
            leaflet_train_residx = {}
            # Save each residues indexes with respect to leaflet assignment
            for resname, idx in resid_dict.items():
                leaflet_assign = self.leaflet_assignment[idx]
                leaflet_train_residx[resname] = {0: [], 1: []}
                leaflet_residx[resname] = {0: [], 1: []}
                for i in range(len(leaflet_assign)):
                    if (leaflet_assign[i] == 0).all():
                        leaflet_train_residx[resname][0].append(idx[i])
                        leaflet_residx[resname][0].append(idx[i])
                    elif (leaflet_assign[i] == 1).all():
                        leaflet_train_residx[resname][1].append(idx[i])
                        leaflet_residx[resname][1].append(idx[i])
                    else:
                        # If a residue is %80 of time belongs to a leaflet, accept it as residue of that leaflet
                        lower_leaflet_percentage = len(np.nonzero(leaflet_assign[i] == 1)[0]) / len(leaflet_assign[i])
                        if lower_leaflet_percentage >= 0.8:
                            leaflet_train_residx[resname][1].append(idx[i])
                        elif lower_leaflet_percentage <= 0.2:
                            leaflet_train_residx[resname][0].append(idx[i])
                        if lower_leaflet_percentage > 0.5:
                            leaflet_residx[resname][1].append(idx[i])
                        else:
                            leaflet_residx[resname][0].append(idx[i])
            self.leaflet_residx = leaflet_residx
            # Prepare rest of train data for each residue
            for resname, tails in self.tails.items():
                rsn_ids = self.residue_ids[resname]
                self.results.train_data_per_type[resname][0] = rsn_ids
                # Select columns of area per lipid and tails' scc parameters
                residx = leaflet_train_residx[resname]
                upper_leaflet_data = self.results.train[residx[0]][:, :, 0:len(tails) + 1]
                lower_leaflet_data = self.results.train[residx[1]][:, :, 0:len(tails) + 1]
                self.results.train_data_per_type[resname][1] = [upper_leaflet_data, lower_leaflet_data]
                self.results.train_data_per_type[resname][2] = self.leaflet_assignment[idx]

            for i, (resname, tail) in enumerate(self.sterol_tails_selection.items()):
                rsn_ids = self.residue_ids[resname]
                self.results.train_data_per_type[resname][0] = rsn_ids
                idx = self.get_residue_idx(self.resids_index_map, rsn_ids)
                # Select columns of area per lipid and sterol's scc parameter
                residx = leaflet_train_residx[resname]
                upper_leaflet_data = self.results.train[residx[0]][:, :, [0, 1 + self.max_tail_len + i]]
                lower_leaflet_data = self.results.train[residx[1]][:, :, [0, 1 + self.max_tail_len + i]]
                self.results.train_data_per_type[resname][1] = [upper_leaflet_data, lower_leaflet_data]
                self.results.train_data_per_type[resname][2] = self.leaflet_assignment[idx]
        else:
            # For symmetric membranes, models will be trained for each residue type
            for resname, tails in self.tails.items():
                rsn_ids = self.residue_ids[resname]
                self.results.train_data_per_type[resname][0] = rsn_ids
                idx = resid_dict[resname]
                # Select columns of area per lipid and tails' scc parameters
                self.results.train_data_per_type[resname][1] = self.results.train[idx][:, :, 0:len(tails) + 1]
                self.results.train_data_per_type[resname][2] = self.leaflet_assignment[idx]

            for i, (resname, tail) in enumerate(self.sterol_tails_selection.items()):
                rsn_ids = self.residue_ids[resname]
                self.results.train_data_per_type[resname][0] = rsn_ids
                idx = resid_dict[resname]
                # Select columns of area per lipid and sterol's scc parameter
                self.results.train_data_per_type[resname][1] = self.results.train[idx][:, :,
                                                               [0, 1 + self.max_tail_len + i]]
                self.results.train_data_per_type[resname][2] = self.leaflet_assignment[idx]

    # ------------------------------ FIT GAUSSIAN MIXTURE MODEL ------------------------------------------------------ #
    def GMM(self, gmm_kwargs):
        """
        Fit Gaussian Mixture

        Parameters
        ----------
        gmm_kwargs : dict
            Additional parameters for mixture.GaussianMixture
        """
        # Iterate over each residue and implement gaussian mixture model
        for res, data in self.results.train_data_per_type.items():
            if self.asymmetric_membrane:
                temp_dict = {}
                for leaflet in range(2):
                    gmm_data = data[1][leaflet]
                    if len(gmm_data) == 0:
                        temp_dict[leaflet] = None
                    else:
                        gmm = mixture.GaussianMixture(n_components=2, **gmm_kwargs).fit(
                            gmm_data.reshape(-1, gmm_data.shape[2]))
                        temp_dict[leaflet] = gmm
                    log.info(f"Leaflet {leaflet}, {res} Gaussian Mixture Model is trained.")
                self.results["GMM"][res] = temp_dict
            else:
                gmm = mixture.GaussianMixture(n_components=2, **gmm_kwargs).fit(data[1].reshape(-1, data[1].shape[2]))
                self.results["GMM"][res] = gmm
                log.info(f"{res} Gaussian Mixture Model is trained.")

        # Check for convergence
        for resname, each in self.results["GMM"].items():
            if self.asymmetric_membrane:
                if each[0] is not None and not each[0].converged_:
                    log.warning(f"{resname} lower leaflet Gaussian Mixture Model is not converged.")
                if each[1] is not None and not each[1].converged_:
                    log.warning(f"{resname} upper leaflet Gaussian Mixture Model is not converged.")
            else:
                if not each.converged_:
                    log.warning(f"{resname} Gaussian Mixture Model is not converged.")

    def mixture_plot(self, resname, gmm_model, hmm_model, leaflet=None):

        """
        Plot results of the Gaussian Mixture Model for each residue type.
        The function should help to spot flaws in the results of the Gaussian Mixture Model.

        The Gaussian Mixture Model in DomHMM is trained on a three-dimensional space (1 APL + 2 Acyl chains), hence also
        its ouput (mean vectors and covariance matrices) are three0-dimensional. However, plots in 3 dimensions are rather hard 
        to understand, therefore we plot here the projections of the raw/fitted probability densities on the xz, yz, and xy plane.

        Parameters
        ----------
        resname : str
            Resname of the actual residue type
        gmm_model : GaussianMixture Model
            Scikit-learn object
        hmm_model : Hidden Markow Model
            hmmlearn object
        leaflet : int or None
            If membrane is asymmetric ensure that plots are made for upper and lower leaflet
        """

        # Define parameters that define a grid for the calculations of the fitted distributions
        # The chosen values should cover a wide range of different use cases as long as enough sample data is provided!
        MIN_APL, MAX_APL, STEP_APL = 0, 200, .5
        MIN_SCC, MAX_SCC, STEP_SCC = -.55, 1.05, .05

        x01, y01 = np.mgrid[MIN_APL:MAX_APL:STEP_APL, MIN_SCC:MAX_SCC:STEP_SCC]
        x2, y2 = np.mgrid[MIN_SCC:MAX_SCC:STEP_SCC, MIN_SCC:MAX_SCC:STEP_SCC]

        xy01 = np.dstack((x01, y01))
        xy2 = np.dstack((x2, y2))

        # Define colors for the markers that mark the fitted means of the Gaussians
        cx = ["crimson", "cyan"]
        cx_hmm = ["orange", "hotpink"]

        # Get trainings data for this type lipid and check if its asymmetric
        train_data_per_type = self.results["train_data_per_type"][resname][1]

        if leaflet != None:
            train_data_per_type = train_data_per_type[leaflet]
        else:
            pass

        # Joint distributions for current lipid type

        cm = 1 / 2.54

        # Do the plot for non-sterolic lipid types
        if resname not in self.sterol_heads.keys():

            # Init suplots
            fig, ax = plt.subplots(1, 3, figsize=(35 * cm, 10 * cm))

            # Area per Lipid - Scc Chain A
            ax[0].hist2d(x=train_data_per_type.reshape(-1, 3)[:, 0],
                         y=train_data_per_type.reshape(-1, 3)[:, 1],
                         bins=[np.linspace(MIN_APL, MAX_APL, int(np.round((MAX_APL - MIN_APL) / 1 + 1))),
                               np.linspace(MIN_SCC, MAX_SCC, int(np.round((MAX_SCC - MIN_SCC) / 0.025 + 1)))],
                         density=True, cmap="viridis")

            # Area per Lipid - Scc Chain B
            ax[1].hist2d(x=train_data_per_type.reshape(-1, 3)[:, 0],
                         y=train_data_per_type.reshape(-1, 3)[:, 2],
                         bins=[np.linspace(MIN_APL, MAX_APL, int(np.round((MAX_APL - MIN_APL) / 1 + 1))),
                               np.linspace(MIN_SCC, MAX_SCC, int(np.round((MAX_SCC - MIN_SCC) / 0.025 + 1)))],
                         density=True, cmap="viridis")

            # Scc Chain A - Scc Chain B
            ax[2].hist2d(x=train_data_per_type.reshape(-1, 3)[:, 1],
                         y=train_data_per_type.reshape(-1, 3)[:, 2],
                         bins=[np.linspace(MIN_SCC, MAX_SCC, int(np.round((MAX_SCC - MIN_SCC) / 0.025 + 1))),
                               np.linspace(MIN_SCC, MAX_SCC, int(np.round((MAX_SCC - MIN_SCC) / 0.025 + 1)))],
                         density=True, cmap="viridis")

            # Calculate and display fitted distributions
            # Iterate over both Gaussian distributions
            for i in range(2):
                # -------------------------------------------------------GMM-------------------------------------------------------

                # Area per Lipid (0) - Scc Chain A (1)
                rv0 = stats.multivariate_normal(gmm_model.means_[i][0:2], gmm_model.covariances_[i][:2, :2])
                z0 = rv0.pdf(xy01)

                # Area per Lipid (0) - Scc Chain B (2)
                rv1 = stats.multivariate_normal(gmm_model.means_[i][::2], gmm_model.covariances_[i][::2, ::2])
                z1 = rv1.pdf(xy01)

                # Scc Chain A (1) - Scc Chain B (2)
                rv2 = stats.multivariate_normal(gmm_model.means_[i][1:], gmm_model.covariances_[i][1:, 1:])
                z2 = rv2.pdf(xy2)

                # Plot contours -> Be aware that the first two plots share the same ranges
                ax[0].contour(x01, y01, z0, colors=[cx[i]], alpha=0.2, linewidths=2.25, zorder=25)
                ax[1].contour(x01, y01, z1, colors=[cx[i]], alpha=0.2, linewidths=2.25, zorder=25)
                ax[2].contour(x2, y2, z2, colors=[cx[i]], alpha=0.2, linewidths=2.25, zorder=25)

                del rv0
                del rv1
                del rv2
                del z0
                del z1
                del z2

                # Mark with a cross the means of the fitted Gaussians
                ax[0].scatter(gmm_model.means_[i][0], gmm_model.means_[i][1], marker="x", s=100, zorder=100,
                              color=cx[i])
                ax[1].scatter(gmm_model.means_[i][0], gmm_model.means_[i][2], marker="x", s=100, zorder=100,
                              color=cx[i], label=f"GMM Mean {i}")
                ax[2].scatter(gmm_model.means_[i][1], gmm_model.means_[i][2], marker="x", s=100, zorder=100,
                              color=cx[i])

                # -------------------------------------------------------HMM-------------------------------------------------------

                # Area per Lipid (0) - Scc Chain A (1)
                rv0 = stats.multivariate_normal(hmm_model.means_[i][0:2], hmm_model.covars_[i][:2, :2])
                z0 = rv0.pdf(xy01)

                # Area per Lipid (0) - Scc Chain B (2)
                rv1 = stats.multivariate_normal(hmm_model.means_[i][::2], hmm_model.covars_[i][::2, ::2])
                z1 = rv1.pdf(xy01)

                # Scc Chain A (1) - Scc Chain B (2)
                rv2 = stats.multivariate_normal(hmm_model.means_[i][1:], hmm_model.covars_[i][1:, 1:])
                z2 = rv2.pdf(xy2)

                # Plot contours -> Be aware that the first two plots share the same ranges
                ax[0].contour(x01, y01, z0, colors=[cx_hmm[i]], alpha=0.2, zorder=50, linewidths=1.25)
                ax[1].contour(x01, y01, z1, colors=[cx_hmm[i]], alpha=0.2, zorder=50, linewidths=1.25)
                ax[2].contour(x2, y2, z2, colors=[cx_hmm[i]], alpha=0.2, zorder=50, linewidths=1.25)

                del rv0
                del rv1
                del rv2
                del z0
                del z1
                del z2

                # Mark with a circle the means of the fitted Gaussians from the HMM
                ax[0].scatter(hmm_model.means_[i][0], hmm_model.means_[i][1], marker="o", facecolor="none", s=100,
                              zorder=100, edgecolor=cx_hmm[i])
                ax[1].scatter(hmm_model.means_[i][0], hmm_model.means_[i][2], marker="o", facecolor="none", s=100,
                              zorder=100, edgecolor=cx_hmm[i], label=f"HMM Mean {i}")
                ax[2].scatter(hmm_model.means_[i][1], hmm_model.means_[i][2], marker="o", facecolor="none", s=100,
                              zorder=100, edgecolor=cx_hmm[i])

            # Set ticks on the y axis
            for i in range(3): ax[i].set_yticks([-0.5, 0, 0.5, 1], [r"$-0.5$", r"$0$", r"$0.5$", r"$1$"])

            # Set ticks on the x axis
            ax[0].set_xticks(np.linspace(MIN_APL, MAX_APL, int(np.round((MAX_APL - MIN_APL) / 25 + 1))))
            ax[1].set_xticks(np.linspace(MIN_APL, MAX_APL, int(np.round((MAX_APL - MIN_APL) / 25 + 1))))
            ax[2].set_xticks([-0.5, 0, 0.5, 1], [r"$-0.5$", r"$0$", r"$0.5$", r"$1$"])

            # Label y axis
            yl0 = ax[0].set_ylabel(r"$p(\bar{S}_{CC}^{\text{sn-2}})$", labelpad=-7, fontsize=18)
            yl1 = ax[1].set_ylabel(r"$p(\bar{S}_{CC}^{\text{sn-1}})$", labelpad=-7, fontsize=18)
            yl2 = ax[2].set_ylabel(r"$p(\bar{S}_{CC}^{\text{sn-1}})$", labelpad=-7, fontsize=18)

            # Label x axis
            xl0 = ax[0].set_xlabel(r"$p(a)$", labelpad=0, fontsize=18)
            xl1 = ax[1].set_xlabel(r"$p(a)$", labelpad=0, fontsize=18)
            xl2 = ax[2].set_xlabel(r"$p(\bar{S}_{CC}^{\text{sn-2}})$", labelpad=0, fontsize=18)

            # Make plot more readable
            for i in range(3):
                ax[i].tick_params(rotation=45)
                ax[i].grid(False)
                ax[i].tick_params(axis="both", labelsize=11)

            plt.subplots_adjust(hspace=0.1, wspace=0.25)

            lg = ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=False, framealpha=0.5,
                              facecolor="grey", ncols=4)

            if leaflet != None:
                plt.savefig(f"GMM_{resname}_{leaflet}.pdf", dpi=300, transparent=True,
                            bbox_extra_artists=(xl0, xl1, xl2, yl0, yl1, yl2, lg), bbox_inches="tight")
            else:
                plt.savefig(f"GMM_{resname}.pdf", dpi=300, transparent=True,
                            bbox_extra_artists=(xl0, xl1, xl2, yl0, yl1, yl2, lg), bbox_inches="tight")

        # Do the plot for sterolic lipid types
        else:

            # Init only one subplot
            fig, ax = plt.subplots(1, 1, figsize=(15 * cm, 15 * cm))

            # Area per Lipid - Scc Chain A
            ax.hist2d(x=train_data_per_type.reshape(-1, 2)[:, 0],
                      y=train_data_per_type.reshape(-1, 2)[:, 1],
                      bins=[np.linspace(MIN_APL, MAX_APL, int(np.round((MAX_APL - MIN_APL) / 1 + 1))),
                            np.linspace(MIN_SCC, MAX_SCC, int(np.round((MAX_SCC - MIN_SCC) / 0.025 + 1)))],
                      density=True, cmap="viridis")

            # Calculate and display fitted distributions
            # Iterate over both Gaussian distributions
            for i in range(2):
                # -------------------------------------------------------GMM-------------------------------------------------------

                # Area per Lipid (0) - Scc Chain C (1)
                rv0 = stats.multivariate_normal(gmm_model.means_[i], gmm_model.covariances_[i])
                z0 = rv0.pdf(xy01)

                # Plot contours
                ax.contour(x01, y01, z0, colors=[cx[i]], alpha=0.2, linewidths=2.25, zorder=25)

                del rv0
                del z0

                # Mark with a cross the means of the fitted Gaussians
                ax.scatter(gmm_model.means_[i][0], gmm_model.means_[i][1], marker="x", s=100, zorder=100, color=cx[i],
                           label=f"GMM Mean {i}")

                # -------------------------------------------------------HMM-------------------------------------------------------

                # Area per Lipid (0) - Scc Chain C (1)
                rv0 = stats.multivariate_normal(hmm_model.means_[i], hmm_model.covars_[i])
                z0 = rv0.pdf(xy01)

                # Plot contours
                ax.contour(x01, y01, z0, colors=[cx_hmm[i]], alpha=0.2, zorder=50, linewidths=1.25)

                del rv0
                del z0

                # Mark with a circle the means of the fitted Gaussians from the HMM
                ax.scatter(hmm_model.means_[i][0], hmm_model.means_[i][1], marker="o", facecolor="none", s=100,
                           zorder=100, edgecolor=cx_hmm[i], label=f"HMM Mean {i}")

            # Set ticks on the x and y axis
            ax.set_yticks([-0.5, 0, 0.5, 1], [r"$-0.5$", r"$0$", r"$0.5$", r"$1$"])
            ax.set_xticks(np.linspace(MIN_APL, MAX_APL, int(np.round((MAX_APL - MIN_APL) / 25 + 1))))

            # Label the x and y axis
            yl = ax.set_ylabel(r"$p(P_2)$", labelpad=-7, fontsize=18)
            xl = ax.set_xlabel(r"$p(a)$", labelpad=0, fontsize=18)

            # Make plot more readable
            ax.tick_params(rotation=45)
            ax.grid(False)
            ax.tick_params(axis="both", labelsize=11)

            lg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=False, framealpha=0.5,
                           facecolor="grey", ncols=4)

            if leaflet != None:
                plt.savefig(f"GMM_{resname}_{leaflet}.pdf", dpi=300, transparent=True, bbox_extra_artists=(xl, yl, lg),
                            bbox_inches="tight")
            else:
                plt.savefig(f"GMM_{resname}.pdf", dpi=300, transparent=True, bbox_extra_artists=(xl, yl, lg),
                            bbox_inches="tight")

        plt.close()

    # ------------------------------ HIDDEN MARKOV MODEL ------------------------------------------------------------- #

    def HMM(self, hmm_kwargs):
        """
        Create Gaussian based Hidden Markov Models for each residue type

        Parameters
        ----------
        hmm_kwargs : dict
            Additional parameters for hmmlearn.hmm.GaussianHMM
        """
        # Iterate over each residue and implement gaussian-hidden markov model
        for resname, data in self.results.train_data_per_type.items():
            if self.asymmetric_membrane:
                temp_dict = {}
                for leaflet in range(2):
                    gmm = self.results["GMM"][resname][leaflet]
                    if gmm is not None:
                        hmm_data = data[1][leaflet]
                        hmm = self.fit_hmm(data=hmm_data, gmm=gmm, hmm_kwargs=hmm_kwargs,
                                           n_repeats=self.n_init_hmm)
                        temp_dict[leaflet] = hmm
                    else:
                        temp_dict[leaflet] = None
                    log.info(f"Leaflet {leaflet}, {resname} Gaussian Hidden Markov Model is trained.")
                self.results["HMM"][resname] = temp_dict
            else:
                hmm = self.fit_hmm(data=data[1], gmm=self.results["GMM"][resname], hmm_kwargs=hmm_kwargs,
                                   n_repeats=self.n_init_hmm)
                self.results["HMM"][resname] = hmm
                log.info(f"{resname} Gaussian Hidden Markov Model is trained.")
        if self.result_plots:
            # Plot result of hmm
            self.plot_hmm_result()
        # Make predictions based on HMM model
        self.predict_states()
        # Validate states and result prediction
        self.state_validate()
        if self.result_plots:
            # Plot prediction result
            self.predict_plot()

            # Plot results of GMM and HMM

            # Iterate over fitted Gaussian Mixture Models
            for resname in self.results["GMM"].keys():

                gmm_trained = self.results["GMM"][resname]
                hmm_trained = self.results["HMM"][resname]

                if self.asymmetric_membrane:

                    if gmm_trained[0] is not None and hmm_trained[0] is not None and gmm_trained[0].converged_ and \
                            hmm_trained[0].monitor_.converged: self.mixture_plot(resname=resname,
                                                                                 gmm_model=gmm_trained[0],
                                                                                 hmm_model=hmm_trained[0], leaflet=0)

                    if gmm_trained[1] is not None and hmm_trained[1] is not None and gmm_trained[1].converged_ and \
                            hmm_trained[1].monitor_.converged: self.mixture_plot(resname=resname,
                                                                                 gmm_model=gmm_trained[1],
                                                                                 hmm_model=hmm_trained[1], leaflet=1)

                else:
                    if gmm_trained.converged_ and hmm_trained.monitor_.converged: self.mixture_plot(resname=resname,
                                                                                                    gmm_model=gmm_trained,
                                                                                                    hmm_model=hmm_trained)

    def fit_hmm(self, data, gmm, hmm_kwargs, n_repeats=10):

        """
        Fit several HMM models to the data and return the best one.

        Parameters
        ----------
        data : numpy.ndarray
            Data of the single lipid properties for one lipid type at each time step
        gmm : GaussianMixture Model
            Scikit-learn object
        hmm_kwargs: dict
            Additional parameters for Hidden Markov Model
        n_repeats : int
            Number of independent fits

        Returns
        -------
        best_ghmm : GaussianHMM
            hmmlearn object

        """
        best_ghmm = None
        # Specify the length of the sequence for each lipid
        n_lipids = data.shape[0]
        dim = data.shape[2]
        lengths = np.repeat(self.n_frames, n_lipids)

        # The HMM fitting is started multiple times from
        # different starting conditions

        # Initialize the best score with minus infinity
        best_score = -np.inf

        # Re-start the HMM fitting 10 times
        for i in tqdm(range(n_repeats)):

            # Initialize a HMM for one lipid type
            ghmm_i = GaussianHMM(n_components=2,
                                 means_prior=gmm.means_,
                                 covars_prior=gmm.covariances_,
                                 **hmm_kwargs)

            # Check whether an optimization of the mean vectors and the covariance matrices is requested
            # Optimization of means is not required  -> Take it from the Gaussian Mixture Model
            if "m" not in hmm_kwargs['params']: ghmm_i.means_ = gmm.means_
            # Optimization of covariances is not required -> Take it from the Gaussian Mixture Model
            if "c" not in hmm_kwargs['params']: ghmm_i.covars_ = gmm.covariances_

            # Train the HMM based on the data for every lipid and frame
            ghmm_i.fit(data.reshape(-1, dim),
                       lengths=lengths)

            # Obtain the log-likelihood
            # probability of the current model
            score_i = ghmm_i.score(data.reshape(-1, dim),
                                   lengths=lengths)

            # Check if the quality of the result improved
            if score_i > best_score:
                best_score = score_i
                best_ghmm = ghmm_i

            # Delete the current model
            del ghmm_i

        return best_ghmm

    def plot_hmm_result(self):
        """
        Plots convergence of HMM model training process.
        """
        x_len = 0
        for resname, ghmm in self.results['HMM'].items():
            if self.asymmetric_membrane:
                for leaflet in range(2):
                    if ghmm[leaflet] is not None:
                        x_len = max(x_len, len(ghmm[leaflet].monitor_.history) - 1)
                        plt.semilogy(np.arange(len(ghmm[leaflet].monitor_.history) - 1),
                                     np.diff(np.array(ghmm[leaflet].monitor_.history)),
                                     ls="-", label=f"{resname}_{leaflet}", lw=2)
            else:
                x_len = max(x_len, len(ghmm.monitor_.history) - 1)
                plt.semilogy(np.arange(len(ghmm.monitor_.history) - 1), np.diff(np.array(ghmm.monitor_.history)),
                             ls="-", label=resname, lw=2)
        x_len = (x_len // 100 + 1) * 100
        plt.legend(fontsize=15)
        plt.semilogy(np.arange(x_len), np.repeat(1E-4, x_len), color="k", ls="--", lw=2)
        plt.xlim(0, x_len)
        plt.ylim(1E-5, 15E5)
        plt.ylabel(r"$\Delta(log(\hat{L}))$", fontsize=18)
        plt.xlabel("Iterations", fontsize=18)
        plt.tick_params(axis="both", labelsize=11)
        plt.text(s="Tolerance 1e-4", x=1, y=1E-4 + 0.00005, color="k", fontsize=15)
        plt.title("a", fontsize=20, fontweight="bold", loc="left")
        plt.tight_layout()
        if self.save_plots:
            plt.savefig("a.pdf")
        plt.show()

    def predict_states(self):
        """
        Each residues prediction assignment to ordered or disordered domains.

        """
        if self.asymmetric_membrane:
            # Since we select stable lipids for training, we need all training data of lipids to predict order of
            # frequently flip-flop doing ones
            for resname, tails in self.tails.items():
                temp_array = []
                for leaflet in range(2):
                    idx = self.leaflet_residx[resname][leaflet]
                    data = self.results.train[idx][:, :, 0:len(tails) + 1]
                    shape = data.shape
                    hmm = self.results['HMM'][resname][leaflet]
                    if hmm is not None:
                        lengths = np.repeat(shape[1], shape[0])
                        prediction = hmm.predict(data.reshape(-1, shape[2]), lengths=lengths).reshape(shape[0],
                                                                                                      shape[1])
                        prediction = self.hmm_diff_checker(hmm.means_, prediction)
                        temp_array.append([idx, prediction])
                    else:
                        temp_array.append([])

                # Leaflet checks in case lipid is only in one leaflet
                if len(temp_array[0]) == 0:
                    idx = np.array(temp_array[1][0]).argsort()
                    result = temp_array[1][1]
                elif len(temp_array[1]) == 0:
                    idx = np.array(temp_array[0][0]).argsort()
                    result = temp_array[0][1]
                else:
                    idx = np.concatenate((temp_array[0][0], temp_array[1][0])).argsort()
                    result = np.concatenate((temp_array[0][1], temp_array[1][1]))
                result = result[idx]
                self.results['HMM_Pred'][resname] = result

            for i, (resname, tail) in enumerate(self.sterol_tails_selection.items()):
                temp_array = []
                for leaflet in range(2):
                    idx = self.leaflet_residx[resname][leaflet]
                    data = self.results.train[idx][:, :, [0, 1 + self.max_tail_len + i]]
                    shape = data.shape
                    hmm = self.results['HMM'][resname][leaflet]
                    if hmm is not None:
                        lengths = np.repeat(shape[1], shape[0])
                        prediction = hmm.predict(data.reshape(-1, shape[2]), lengths=lengths).reshape(shape[0],
                                                                                                      shape[1])
                        prediction = self.hmm_diff_checker(hmm.means_, prediction)
                        temp_array.append([idx, prediction])
                    else:
                        temp_array.append(None)

                # Leaflet checks in case lipid is only in one leaflet
                if temp_array[0] is None:
                    idx = np.array(temp_array[1][0]).argsort()
                    result = temp_array[1][1]
                elif temp_array[1] is None:
                    idx = np.array(temp_array[0][0]).argsort()
                    result = temp_array[0][1]
                else:
                    idx = np.concatenate((temp_array[0][0], temp_array[1][0])).argsort()
                    result = np.concatenate((temp_array[0][1], temp_array[1][1]))
                result = result[idx]
                self.results['HMM_Pred'][resname] = result
        else:
            # Symmetric membrane case
            for resname, data in self.results.train_data_per_type.items():
                shape = data[1].shape
                hmm = self.results['HMM'][resname]
                # Lengths consists of number of frames and number of residues
                lengths = np.repeat(shape[1], shape[0])
                prediction = hmm.predict(data[1].reshape(-1, shape[2]), lengths=lengths).reshape(shape[0], shape[1])
                # Save prediction result of each residue
                self.results['HMM_Pred'][resname] = prediction

    def state_validate(self):
        """
        Validate state assignments of HMM model by checking means of the model of each residue.
        """
        if not self.asymmetric_membrane:
            # Asymmetric membrane validation is done in prediction step due to nature of it
            for resname, hmm in self.results["HMM"].items():
                if hmm is not None:
                    means = hmm.means_
                    prediction_results = self.results['HMM_Pred'][resname]
                    self.results['HMM_Pred'][resname] = self.hmm_diff_checker(means, prediction_results)

    @staticmethod
    def hmm_diff_checker(means, prediction_results):
        """
        Checks if prediction assignments correct with respect to means. Since HMM is unsupervised, model can assign 0 to
         disordered domains and 1 to ordered domains. In these cases needs to be changed to vice versa.

        Parameters
        ----------
        means : np.ndarray
            Table of model which contains means of area per lipid and order parameters
        prediction_results: np.ndarray
            Initial prediction results which is 1s and 0s

        Returns
        -------
        prediction_results : np.ndarray
            Same or flipped prediction results with respect to means table

        """
        diff_percents = (means[1, 0] - means[0, 0]) / means[0, 0]
        if diff_percents > 0.0:
            return np.abs(prediction_results - 1)
        else:
            return prediction_results

    @staticmethod
    def get_residue_idx(resids_index_map, resids):
        """
        Checks resids index map (lookup table) to return corresponding indexes to use in DomHMM result section.

        Parameters
        ----------
        resids_index_map: dict
            Lookup table where each residue id's index in self.membrane_unique_resids is kepts
        resids: numpy.ndarray
            Residue id list for indexing

        Returns
        -------
        idx: numpy.ndarray
            Indexes in the same order with resids

        """
        idx = np.array([resids_index_map[resid] for resid in resids])
        return idx

    def predict_plot(self):
        """
        Plots average ordered lipids of HMM prediction for each lipid type.
        """
        start_time = self.universe.trajectory[self.start].time
        end_time = self.universe.trajectory[self.stop - 1].time
        # Configured as printing 0.1 microsecond instead 1000 nanoseconds for readability.
        if end_time - start_time >= 1e8:
            time_unit = "ms"
            scale_factor = 1e9
        elif end_time - start_time >= 1e5:
            time_unit = "μs"
            scale_factor = 1e6
        else:
            time_unit = "ns"
            scale_factor = 1e3
        start_time /= scale_factor
        end_time /= scale_factor

        t = np.linspace(start_time, end_time, self.n_frames)
        for resname in self.unique_resnames:
            plt.plot(t, self.results['HMM_Pred'][resname].mean(0), label=resname)

        plt.xlabel(f"t ({time_unit})", fontsize=18)
        plt.ylabel(r"$\bar{O}_{Lipid}$", fontsize=18)
        plt.legend(fontsize=15, ncols=1, loc="lower left")
        plt.ylim(0, 1)
        plt.title("b", fontsize=20, fontweight="bold", loc="left")
        plt.tight_layout()
        if self.save_plots:
            plt.savefig("b.pdf")
        plt.show()

    # ------------------------------ GETIS-ORD STATISTIC ------------------------------------------------------------- #
    def getis_ord(self):
        """
        Calculates Getis-Ord statistic for identification model
        """
        self.getis_ord_stat(self.results["upper_weight_all"], 0)
        self.getis_ord_stat(self.results["lower_weight_all"], 1)
        log.info("Getis-Ord for leaflets are calculated.")
        if self.result_plots:
            self.getis_ord_plot()
        self.results["Getis_Ord"]["Permut_0"] = self.permut_getis_ord_stat(self.results["upper_weight_all"], 0)
        self.results["Getis_Ord"]["Permut_1"] = self.permut_getis_ord_stat(self.results["lower_weight_all"], 1)
        log.info("Permutations of Getis-Ord are calculated.")
        self.results["z_score"] = self.z_score_calc()
        log.info("Z score is calculated.")

    def getis_ord_stat(self, weight_matrix_all, leaflet):
        """
            Getis-Ord Local Spatial Autocorrelation Statistics calculation based on the predicted order states
            of each lipid and the weighting factors between neighbored lipids.

            Parameters
            ----------
            weight_matrix_all : sparse.csr_array
                Weight matrix for all lipid in a leaflet at current time step
            leaflet : int
                0 = upper leaflet, 1 = lower leafet

            Returns
            -------
            g_star_i : list of numpy.ndarrays
                G*i values for each lipid at each time step
            w_ii_all : list of numpy.ndarrays
                Self-influence of the lipids
            """
        # Get the number of frames
        nframes = self.n_frames

        # Initialize empty lists to store the G*i values and the self-influence of the lipids in a lipid
        g_star_i = []
        w_ii_all = []

        # Iterate over frames
        for step in range(nframes):
            # Get the weight matrix of the leaflet at the current time step
            weight_matrix = weight_matrix_all[step]

            # Number of lipids in the leaflet
            n = weight_matrix.shape[0]

            # In case the code was already executed beforehand
            weight_matrix[range(n), range(n)] = 0.

            # Get the order state of each lipid in the leaflet at the current time step
            order_states = self.get_leaflet_step_order(leaflet=leaflet, step=step)

            # Number of neighbors per lipid -> The number is 0 (or close to 0) for not neighboured lipids
            nneighbor = np.sum(weight_matrix > 1E-5, axis=1)
            # Parameters for the Getis-Ord statistic
            w_ii = np.sum(weight_matrix, axis=-1) / nneighbor  # Self-influence!
            weight_matrix[range(n), range(n)] = w_ii
            w_star_i = np.sum(weight_matrix, axis=-1)  # + w_ii
            s_star_1i = np.sum(weight_matrix.power(2), axis=-1)  # + w_ii**2

            # Empirical standard deviation over all order states in the leaflet
            s = np.std(order_states, ddof=1)

            # Employ matrix-vector multiplication
            so = weight_matrix @ order_states

            # Calculate the nominator for G*i
            nom = so - w_star_i * 0.5

            # Calculate the denominator for G*i
            denom = s * np.sqrt((n * s_star_1i - w_star_i ** 2) / (n - 1))

            g_star = nom / denom

            assert not np.any(nneighbor < 1), "Lipid found without a neighbor!"

            g_star_i.append(g_star)
            w_ii_all.append(w_ii)
        self.results['Getis_Ord'][leaflet] = {f"g_star_i_{leaflet}": g_star_i, f"w_ii_{leaflet}": w_ii_all}

    def getis_ord_plot(self):
        """
        Plots average order state for each lipid type per frame
        """

        # Get number of lipid residues
        resnum = len(self.unique_resnames)

        # For each residue type there is an empty list initialized
        g_star_i_temp = [[] for _ in range(resnum)]

        # Iterate over all frames
        for step in range(self.n_frames):

            # Get the indices of the lipids in leaflet 0 and 1 and their corresponding positions
            index_dict_0, pos_dict_0 = self.get_leaflet_step_order_index(leaflet=0, step=step)
            index_dict_1, pos_dict_1 = self.get_leaflet_step_order_index(leaflet=1, step=step)

            # Iterate over the lipid types
            for i, rsn in enumerate(self.unique_resnames):
                # Merge Getis-Ord values for the upper and the lower leaflet to display them as a histogram
                # The G* values should be sorted according to the indices of the resids
                if rsn in index_dict_0.keys():
                    g_start_sorted_0 = self.results['Getis_Ord'][0]['g_star_i_0'][step][index_dict_0[rsn]]
                else:
                    g_start_sorted_0 = []
                # The G* values should be sorted according to the indices of the resids
                if rsn in index_dict_1.keys():
                    g_start_sorted_1 = self.results['Getis_Ord'][1]['g_star_i_1'][step][index_dict_1[rsn]]
                else:
                    g_start_sorted_1 = []
                g_star_i_temp[i] += list(np.append(g_start_sorted_0, g_start_sorted_1))

        for i in range(resnum):
            plt.hist(g_star_i_temp[i], bins=np.linspace(-3, 3, 201), density=True, histtype="step", lw=2,
                     label=self.unique_resnames[i])

        plt.legend(fontsize=15, loc="upper left", ncols=2)

        plt.xlim(-3, 3)
        plt.ylim(0, .9)

        _ = plt.xlabel("$G^*_i$", fontsize=18)
        plt.ylabel("$p(G^*_i)$", fontsize=18)
        plt.tick_params(labelsize=11)

        plt.title("c", fontsize=20, fontweight="bold", loc="left")
        plt.tight_layout()
        if self.save_plots:
            plt.savefig("c.pdf")
        plt.show()

    def permut_getis_ord_stat(self, weight_matrix_all, leaflet):

        """
        Getis-Ord Local Spatial Autocorrelation Statistics calculation based on the predicted order states
        of each lipid and the weighting factors between neighbored lipids.

        Parameters
        ----------
        weight_matrix_all : sparse.csr_array
            Weight matrices for all lipid in a leaflet at each time step
        leaflet : int
            0 = upper leaflet, 1 = lower leafet

        Returns
        -------
        g_star_i : list of numpy.ndarrays
            G*i values for each lipid at each time step
        """

        # Get the number of frames
        nframes = self.n_frames

        # Initialize empty lists to store the G*i values and the self-influence of the lipids in a lipid
        g_star_i = []

        # Do 10 permutations per frame
        n_permut = 10

        # Iterate over frames
        for step in tqdm(range(nframes)):

            for permut in range(n_permut):
                # Get the weightmatrix of the leaflet at the current time step
                weight_matrix = weight_matrix_all[step]

                # Number of lipids in the leaflet
                n = weight_matrix.shape[0]

                # In case the code was already executed beforehand
                weight_matrix[range(n), range(n)] = 0.0

                # Get the order state of each lipid in the leaflet at the current time step
                order_states = self.get_leaflet_step_order(leaflet=leaflet, step=step)

                np.random.shuffle(order_states)

                # Number of neighbors per lipid -> The number is 0 (or close to 0) for not neighboured lipids
                nneighbor = np.sum(weight_matrix > 1E-5, axis=1)

                # Parameters for the Getis-Ord statistic
                w_ii = np.sum(weight_matrix, axis=-1) / nneighbor  # Self-influence!
                weight_matrix[range(n), range(n)] = w_ii
                w_star_i = np.sum(weight_matrix, axis=-1)  # + w_ii
                s_star_1i = np.sum(weight_matrix ** 2, axis=-1)  # + w_ii**2

                # Empirical standard deviation over all order states in the leaflet
                s = np.std(order_states, ddof=1)

                # Employ matrix-vector multiplication
                so = weight_matrix @ order_states

                # Calculate the nominator for G*i
                nom = so - w_star_i * 0.5

                # Calculate the denominator for G*i
                denom = s * np.sqrt((n * s_star_1i - w_star_i ** 2) / (n - 1))

                g_star = nom / denom

                assert not np.any(nneighbor < 1), "Lipid found without a neighbor!"

                g_star_i += list(g_star)

        return g_star_i

    def z_score_calc(self):
        """
        Z score calculation for Getis-Ord statistics
        """
        result = {}
        for i in range(2):
            z_score = {}
            getis_ord_permut = self.results["Getis_Ord"][f"Permut_{i}"]
            z_score["z_a"] = np.quantile(getis_ord_permut, self.p_value)
            z_score["z1_a"] = np.quantile(getis_ord_permut, 1 - self.p_value)
            result[i] = z_score
        return result

    # ------------------------------ HIERARCHICAL CLUSTERING --------------------------------------------------------- #
    def clustering_plot(self):
        """
        Runs hierarchical clustering and plots clustering results in different frames.
        """

        n_frames = self.n_frames
        # Plot %5, %50 and %95 points of frame list
        frame_list = [int(n_frames / 20), int(n_frames / 2), int(n_frames / 1.05)]
        fig, ax = plt.subplots(1, len(frame_list), figsize=(20, 5))

        # Iterate over three frames illustrate the clustering results
        for k, i in enumerate(frame_list):
            order_states_0 = self.get_leaflet_step_order(0, i)

            # Clustering
            # ----------------------------------------------------------------------------------------------------------------------
            core_lipids = self.assign_core_lipids(weight_matrix_f=self.results["upper_weight_all"][i],
                                                  g_star_i_f=self.results['Getis_Ord'][0]['g_star_i_0'][i],
                                                  order_states_f=order_states_0,
                                                  w_ii_f=self.results["Getis_Ord"][0]["w_ii_0"][i],
                                                  z_score=self.results["z_score"][0])

            clusters = self.hierarchical_clustering(weight_matrix_f=self.results["upper_weight_all"][i],
                                                    w_ii_f=self.results["Getis_Ord"][0]["w_ii_0"][i],
                                                    core_lipids=core_lipids)

            # Plot coordinates
            # ----------------------------------------------------------------------------------------------------------------------
            residue_indexes, positions = self.get_leaflet_step_order_index(leaflet=0, step=i)

            for resname, index in residue_indexes.items():
                ax[k].scatter(positions[resname][:, 0],
                              positions[resname][:, 1], marker="s", alpha=1, s=5, label=resname)

            # Choose color scheme for clustering coloring
            colors = plt.cm.nipy_spectral(np.linspace(0, 1.0, len(clusters.values())))

            # Goto correct frame of the trajectory
            self.universe.trajectory[self.start:self.stop:self.step][i]

            # Prepare positions for cluster plotting
            leaflet_assignment_mask = self.leaflet_assignment[:, i] == 0

            positions = (self.membrane.residues[leaflet_assignment_mask].atoms & self.all_heads).positions

            label_length = len(residue_indexes)

            # Iterate over clusters and plot the residues
            print(f"Number of clusters in frame {i}: {len(clusters.values())}")
            for j, val in enumerate(clusters.values()):
                idx = np.array(list(val), dtype=int)
                ax[k].scatter(positions[idx, 0],
                              positions[idx, 1],
                              s=100, marker="o", color=colors[j], zorder=-10)
            if self.tmd_protein is not None:
                label_length += len(self.tmd_protein["0"])
                for protein in self.tmd_protein["0"]:
                    ax[k].scatter(protein[0], protein[1], s=100, marker="^", color="black", label="TMD Protein")
            ax[k].set_xticks([])
            ax[k].set_yticks([])
            ax[k].set_aspect("equal")

        handles, labels = ax[0].get_legend_handles_labels()
        plt.figlegend(handles=handles, labels=labels, ncol=label_length, loc='lower center', fontsize=15, frameon=False)
        ax[0].set_title("d", fontsize=20, fontweight="bold", loc="left")

        ax[0].set_title(label=f"Frame {frame_list[0]}", fontsize=18)
        ax[1].set_title(label=f"Frame {frame_list[1]}", fontsize=18)
        ax[2].set_title(label=f"Frame {frame_list[2]}", fontsize=18)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=None)
        if self.save_plots:
            plt.savefig("d.pdf")
        plt.show()

    def result_clustering(self):
        """
        Runs hierarchical clustering for each frame and saves result
        """
        self.results["Clustering"] = {'0': {}, '1': {}}

        # Iterate over all frames
        for i in tqdm(range(self.n_frames), total=self.n_frames):

            # Iterate over both leaflets
            for j, leaflet_ in enumerate(['upper', 'lower']):
                # Get order states
                order_states_leaf = self.get_leaflet_step_order(j, i)

                core_lipids = self.assign_core_lipids(weight_matrix_f=self.results[f"{leaflet_}_weight_all"][i],
                                                      g_star_i_f=self.results['Getis_Ord'][j][f'g_star_i_{j}'][i],
                                                      order_states_f=order_states_leaf,
                                                      w_ii_f=self.results["Getis_Ord"][j][f"w_ii_{j}"][i],
                                                      z_score=self.results["z_score"][j])

                clusters = self.hierarchical_clustering(weight_matrix_f=self.results[f"{leaflet_}_weight_all"][i],
                                                        w_ii_f=self.results["Getis_Ord"][j][f"w_ii_{j}"][i],
                                                        core_lipids=core_lipids)
                frame_number = self.start + i * self.step
                leaflet_index_resid_map = self.get_leaflet_step_index_to_resid(j, i)
                cluster_result = []
                for cluster in clusters.values():
                    cluster_result.append([leaflet_index_resid_map[leaflet_index] for leaflet_index in cluster])
                self.results["Clustering"][str(j)][frame_number] = cluster_result

    @staticmethod
    def assign_core_lipids(weight_matrix_f, g_star_i_f, order_states_f, w_ii_f, z_score):

        """
        Assign lipids as core members (aka lipids with a high positive autocorrelation)
        depending on the Getis-Ord spatial local autocorrelation statistic.

        Parameters
        ----------
        weight_matrix_f : numpy.ndarray
            Matrix containing the weight factors between all lipid pairs at one time step
        g_star_i_f : numpy.ndarray
            Getis-Ord spatial local autocorrelation statistic for every lipid at one time step
        order_states_f: numpy.ndarray
            Order states for every lipid at one time step
        w_ii_f: numpy.ndarray
            Self-influence weight factor for every lipid at one time step
        z_score: dict
            Contains boundary of the reaction region


        Returns
        -------
        core_lipids : numpy.ndarray (bool)
           Contains a TRUE value if the lipid is a core member, otherwise it FALSE
        """

        # Define boundary of the reaction region
        z1_a = z_score["z1_a"]
        z_a = z_score["z_a"]

        # Assign core members according to their auto-correlation
        core_lipids = g_star_i_f > z1_a

        # Assign lipids with a mid-range auto-correlation (-z_1-a, z_1-a) * Ordered
        low_corr = (g_star_i_f <= z1_a) & (g_star_i_f >= z_a) & (order_states_f == 1)

        # Add iteratively new lipids to the core members
        n_cores_old = np.inf
        n_cores_new = np.sum(core_lipids)

        # Iterate until self-consistency is reached
        while n_cores_old != n_cores_new:
            # Check how tightly the lipids are connected to the core members
            new_core_lipids = (weight_matrix_f[core_lipids].sum(0) > w_ii_f) & low_corr

            # Assign lipids to core members if condition is full-filled
            core_lipids[new_core_lipids] = True

            # Update number of core lipids
            n_cores_old = n_cores_new
            n_cores_new = np.sum(core_lipids)

        return core_lipids

    @staticmethod
    def hierarchical_clustering(weight_matrix_f, w_ii_f, core_lipids):

        """
        Hierarchical clustering approach to identify spatial related Lo domains.

        Parameters
        ----------
        weight_matrix_f : numpy.ndarray
            Matrix containing the weight factors between all lipid pairs at one time step
        w_ii_f: numpy.ndarray
            Self-influence weight factor for every lipid at one time step
        core_lipids : numpy.ndarray
            Array contains information which lipid is assigned as core member

        Returns
        -------
        clusters : dict
            The dictionary contains each found cluster

        """

        # Merge iteratively clusters
        n_clusters_old = np.inf
        n_clusters_new = np.sum(core_lipids)

        # Get the indices of the core lipids
        core_lipids_id = np.where(core_lipids)[0]

        # Store clusters in a Python dictionary
        # Initialize all core lipids as clusters
        clusters = dict(
            zip(
                core_lipids_id.astype("U"),
                [[id] for id in core_lipids_id]
            )
        )

        # Iterate until self-consistency is reached
        while n_clusters_old != n_clusters_new:

            # Get a list of the IDs of current clusters
            cluster_ids = list(clusters.keys())

            # Iterate over all clusters i
            for i, id_i in enumerate(cluster_ids):

                # If cluster i was already merged and deleted, skip it!
                if id_i not in clusters.keys():
                    continue

                # The cluster weights are defined as the sum
                # over the weights of all lipid members
                cluster_weights_i = np.sum(weight_matrix_f[clusters[id_i]], axis=0)

                # Compare cluster weights to the self-influence of each lipid
                merge_condition_i = cluster_weights_i > w_ii_f

                # Iterate over all clusters j
                for id_j in cluster_ids[(i + 1):]:

                    # Do not merge a cluster with itself
                    if id_i == id_j:
                        continue
                    # If cluster j was already merged and deleted, skip it!
                    if id_j not in clusters.keys():
                        continue

                    # Calculate cluster weights and compare to lipids self-influence
                    cluster_weights_j = np.sum(weight_matrix_f[clusters[id_j]], axis=0)
                    merge_condition_j = cluster_weights_j > w_ii_f

                    # If the condition is fullfilled for any lipid -> Merge the clusters
                    if np.any(merge_condition_i[clusters[id_j]]) or np.any(merge_condition_j[clusters[id_i]]):
                        # Merge cluster j into cluster i
                        clusters[id_i] += clusters[id_j]

                        # Delete cluster j from the cluster dict
                        del clusters[id_j]

            # Update cluster numbers
            n_clusters_old = n_clusters_new
            n_clusters_new = len(clusters.keys())

        return clusters

    # ------------------------------ HELPER FUNCTIONS ---------------------------------------------------------------- #
    def get_leaflet_step_order(self, leaflet, step):
        """
        Receive residue's order state with respect to the leaflet

        Parameters
        ----------
        leaflet : int
            leaflet index
        step: int
            step index

        Returns
        -------
        order_states : numpy.ndarray
            Numpy array contains order state results of the leaflet at step in order of system's residues
        """

        # Init two empty lists for ...
        temp = []  # ... order states prediction
        idxs = []  # ... indices of lipids

        # Iterate over lipids
        for res, data in self.results.train_data_per_type.items():
            # Get indices from resids (e.g. resid 1, is index 0)
            idx = self.get_residue_idx(self.resids_index_map, data[0])

            # Store the indices of the residues in the current leaflet at the current step
            idxs.append(idx[self.leaflet_assignment[idx, step] == leaflet])

            # Get the predicted HMM order states for the residues in the current leaflet at the current step
            temp.append(self.results["HMM_Pred"][res][:, step][self.leaflet_assignment[idx, step] == leaflet])

        # Sort the obtained indices
        sorted_idxs = np.argsort(np.concatenate(idxs))

        # It is not always the case that the lipid order is equal (e.g., depending on resids of cholesterol for
        # example). Here the order states are sorted so that they correspond to the resids in the leaflet selection
        # -> With that they should fit to the order of the lipids in the weight matrix
        order_states = np.concatenate(temp)[sorted_idxs]

        return order_states

    def get_leaflet_step_order_index(self, leaflet, step):
        """
        Receive residue's indexes and positions for a specific leaflet at any frame of the trajectory.

        Parameters
        ----------
        leaflet : int
            leaflet index
        step: int
            step index

        Returns
        -------
        indexes : dict
            dictionary contains numpy.arrays containing residue's indexes for each unique residue type
        positions : dict
            dictionary contains numpy.arrays containing residue's positions for each unique residue type
        """

        leaflet_assignment_step = self.leaflet_assignment[:, step]
        leaflet_assignment_mask = leaflet_assignment_step == leaflet

        indexes = {}
        positions = {}

        # Go to correct frame of the trajectory
        self.universe.trajectory[self.start:self.stop:self.step][step]

        for res in self.unique_resnames:
            indx = np.where(self.membrane.residues[leaflet_assignment_mask].resnames == res)[0]
            if len(indx) != 0:
                indexes[res] = indx
                if res in self.heads.keys():
                    positions[res] = (
                            self.membrane.residues[leaflet_assignment_mask].atoms & self.universe.select_atoms(
                        f"resname {res} and name {self.heads[res]}")).positions
                else:
                    positions[res] = (
                            self.membrane.residues[leaflet_assignment_mask].atoms & self.universe.select_atoms(
                        f"resname {res} and name {self.sterol_heads[res]}")).positions

        return indexes, positions

    def get_leaflet_step_index_to_resid(self, leaflet, step):
        """
        Returns leaflet index to residue id map for a specific leaflet at specific frame of the trajectory

        Parameters
        ----------
        leaflet : int
            leaflet index
        step: int
            step index

        Returns
        -------
        result_map : dict
            dictionary contains leaflet index of residue as key and residue id as value
        """
        leaflet_assignment_step = self.leaflet_assignment[:, step]
        leaflet_assignment_mask = leaflet_assignment_step == leaflet
        result_map = {}
        for resname in self.unique_resnames:
            sys_index = np.where(self.membrane.residues.resnames == resname)[0]
            sys_index = sys_index[leaflet_assignment_mask[sys_index]]
            leaflet_index = np.where(self.membrane.residues[leaflet_assignment_mask].resnames == resname)[0]
            for i in range(0, len(leaflet_index)):
                result_map[leaflet_index[i]] = self.index_resid_map[sys_index[i]]
        return result_map

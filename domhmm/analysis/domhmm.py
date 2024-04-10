"""
LocalFluctuation --- :mod:`elbe.analysis.LocalFluctuation`
===========================================================

This module contains the :class:`LocalFluctuation` class.

"""

from .base import LeafletAnalysisBase
import numpy as np
from sklearn import mixture
from hmmlearn.hmm import GaussianHMM
from scipy.spatial import Voronoi, ConvexHull
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

        # Although sterols maybe do not play a larger role in the future for the domain identification it seems to be
        # a good idea to keep this functionality
        self.resid_selection_sterols = {}

        # Weight matrix storage for each leaflet.
        setattr(self.results, "upper_weight_all", [])
        setattr(self.results, "lower_weight_all", [])

        # Next, a dictionary for EACH selected resid will be created. That's pretty much, but it is important to have
        # the order parameters for each lipid over the whole trajectory for the domain identification

        # Iterate over all residues in the selected membrane
        for resid in self.membrane_unique_resids:

            # Select specific resid
            resid_selection = self.universe.select_atoms(f"resid {resid}")
            # Get its lipid type
            resname = resid_selection.resnames[0]

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
                    # Init storage for SCC values for each lipid
                    getattr(self.results, f'id{resid}')[f'SCC_{i}'] = np.zeros(self.n_frames, dtype=np.float32)

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
                    # Init storage for SCC values for each lipid
                    getattr(self.results, f'id{resid}')[f'SCC_{i}'] = np.zeros(self.n_frames, dtype=np.float32)

                # Store the area per lipid for each lipid
                getattr(self.results, f'id{resid}')[f'APL'] = np.zeros(self.n_frames, dtype=np.float32)
            else:
                raise ValueError(f'{resname} with resid {resid} not found in leaflets or sterol list!')

    def calc_order_parameter(self, chain):

        """
        Calculate average Scc order parameters per acyl chain according to the equation:

        S_cc = (3 * cos( theta )^2 - 1) / 2,

        where theta describes the angle between the z-axis of the system and the vector between two subsequent tail beads.

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
        ridx = np.where(np.diff(chain.resids) > 0)[0] + 1

        pos = np.array(np.split(chain.positions, ridx))

        # Calculate the normalized orientation vector between two subsequent tail beads
        vec = np.diff(pos, axis=1)

        vec_norm = np.sqrt((vec ** 2).sum(axis=-1))

        vec /= vec_norm.reshape(-1, vec.shape[1], 1)
        # TODO z axis multiplication inside. It needs to be changed in future work
        # Choose the z-axis as membrane normal and take care of the machine precision
        dot_prod = np.clip(vec[:, :, 2], -1., 1.)

        # Calculate the order parameter
        s_cc = 0.5 * (3 * dot_prod ** 2 - 1)

        return s_cc.mean(-1)

    def order_parameter(self):
        """
        Calculation of scc order parameter for each chain of each residue.
        """
        # Iterate over each tail with chain id
        for chain, tail in self.resid_tails_selection.items():
            # SCC calculation
            s_cc = self.calc_order_parameter(tail)

            # Saving result with iterating overall corresponding residues
            for i, resid in enumerate(np.unique(tail.resids)):
                self.results[f'id{resid}'][f'SCC_{chain}'][self.index] = s_cc[i]

    def area_per_lipid_vor(self, leaflet, boxdim):

        """
        Calculation of the area per lipid employing Voronoi tessellation on coordinates mapped to the xy plane.
        The function takes also the periodic boundary conditions of the box into account.

        Parameters
        ----------
        leaflet : string
            Index to decide upper/lower leaflet
        boxdim : array
            Length of box vectors in all directions

        Returns
        -------
        vor : Voronoi Tesselation
            Scipy's Voronoi Diagram object
        """

        # Number of points in the plane
        coor_xy = self.surface_lipids_per_frame[str(leaflet)].positions
        ncoor = coor_xy.shape[0]
        # TODO Boxdim is selected in 2D for basic implementation. It will be changed to 3D with required implementations
        bx = boxdim[0]
        by = boxdim[1]
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

        # Save result of area per lipid
        for i in range(0, len(self.surface_lipids_per_frame[str(leaflet)].resnums)):
            resid = self.surface_lipids_per_frame[str(leaflet)].resnums[i]
            getattr(self.results, f'id{resid}')[f'APL'][self.index] = apl[i]

        return vor

    def weight_matrix(self, vor, leaflet):

        """
        Calculate the weight factors between neighbored lipid pairs based on a Voronoi tessellation.

        Parameters
        ----------
        vor : Voronoi Tesselation
            Scipy's Voronoi Diagram object
        leaflet : string
            Index to decide upper/lower leaflet

        Returns
        -------
        weight_matrix : numpy.ndarray
            Weight factors wij between neighbored lipid pairs.
            Is 0 if lipids are not directly neighbored.
        """

        # Number of points in the plane
        coor_xy = self.surface_lipids_per_frame[str(leaflet)].positions
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

        # Setup an empty array to store the weight factors for each lipid
        weight_matrix = np.zeros((ncoor, ncoor))

        # Select all indices of ridges that contain members of the unit cell
        mask_unit_cell = np.logical_or(vor.ridge_points[:, 0] < ncoor, vor.ridge_points[:, 1] < ncoor)

        # Apply the modulus operator since some of the indices in "unit_cell_point" will point to coordinates outside
        # the unit cell. Applying the modulus operator "%" will allow an indexing of the "weight_matrix". However, some
        # of the indices in "unit_cell_point" will be doubled that shouldn't be an issue since the same weight factor is
        # then just put several times in the same entry of the array (no summing or something similar!)
        unit_cell_point = vor.ridge_points[mask_unit_cell] % ncoor

        weight_matrix[unit_cell_point[:, 0], unit_cell_point[:, 1]] = wij[mask_unit_cell]
        weight_matrix[unit_cell_point[:, 1], unit_cell_point[:, 0]] = wij[mask_unit_cell]
        return weight_matrix

    def _single_frame(self):
        """
        Calculate data from a single frame of the trajectory.
        """

        # Make selection of non-flip/flop lipids and flip/flop lipids if there are sterols present

        self.surface_lipids_per_frame = {}

        # Iterate over leafelts
        for leaflet in range(2):
            self.surface_lipids_per_frame[str(leaflet)] = self.leaflet_selection[str(leaflet)]

        if self.surface_lipids_per_frame["0"].select_atoms("group leaf1",
                                                           leaf1=self.surface_lipids_per_frame["1"]):
            raise ValueError("Atoms in both leaflets !")

        # Get number of frame from trajectory
        self.frame = self.universe.trajectory.ts.frame
        # Calculate correct index if skipping step not equals 1 or start point not equals 0
        self.index = self.frame // self.step - self.start

        # ------------------------------ Local Normals/Area per Lipid ------------------------------------------------ #
        boxdim = self.universe.trajectory.ts.dimensions[0:3]
        upper_vor = self.area_per_lipid_vor(leaflet=0, boxdim=boxdim)
        lower_vor = self.area_per_lipid_vor(leaflet=1, boxdim=boxdim)
        # TODO Local normal calculation

        # ------------------------------ Order parameter ------------------------------------------------------------- #
        self.order_parameter()
        # ------------------------------ Weight Matrix --------------------------------------------------------------- #
        upper_weight_matrix = self.weight_matrix(upper_vor, leaflet=0)
        lower_weight_matrix = self.weight_matrix(lower_vor, leaflet=1)
        self.results["upper_weight_all"].append(upper_weight_matrix)
        self.results["lower_weight_all"].append(lower_weight_matrix)

    def _conclude(self):

        """
        Calculate the final results of the analysis

        Extract the obtained data and put them into a clear and accessible data structure
        """

        # -----------------------------------------------------------------------
        # Make a dictionary for the calculated values of each lipid type for each leaflet
        # -----------------------------------------------------------------------

        # Initialize storage dictionary
        self.results.train_data_per_type = {}

        # Iterate over leaflets -> 0 top, 1 bottom
        for i in range(2):

            # Make dictionary for each leaflet
            self.results.train_data_per_type[f"Leaf{i}"] = {}

            for rsn in self.unique_resnames:
                # Create each leaflet's lipid types 3D empty array.
                # Array will fill with order parameters of tails and area per lipid
                num_tails = len(self.tails[rsn])
                self.results.train_data_per_type[f"Leaf{i}"][f"{rsn}"] = [[] for i in range(num_tails + 1)]

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
                    indv_p2 = getattr(self.results, f'id{resid}')[f'SCC_{n_chain}']

                    # Add it to the lipid type list
                    self.results.train_data_per_type[f"Leaf{leaflet}"][f"{rsn}"][n_chain].append(indv_p2)

                # Get area per lipid for specific residue
                apl = getattr(self.results, f'id{resid}')['APL']
                # Add it to the lipid type's result. Index is -1 since area per lipid is the latest element
                self.results.train_data_per_type[f"Leaf{leaflet}"][f"{rsn}"][-1].append(apl)

            elif rsn in self.sterols:
                pass

            # NOTHING?
            else:
                raise ValueError(f'{rsn} with resid {resid} not found in leaflets or sterol list!')

        # -------------------------------------------------------------

        # Transform lists to arrays
        # TODO Concate all lists to one big
        # Iterate over leaflets
        for i in range(2):
            for rsn in self.unique_resnames:
                self.results.train_data_per_type[f"Leaf{i}"][f"{rsn}"]["data"] = np.array([
                    self.results.train_data_per_type[f"Leaf{i}"][f"{rsn}"][j]
                    for j in range(len(self.results.train_data_per_type[f"Leaf{i}"][f"{rsn}"]))
                ])

        # -------------------------------------------------------------

        # TODO Add post-processing parts to here:
        #  GMM
        #  HMM
        #  GetisOrd
        #  Hierarchical Clustering
        # gmm_kwargs = {"tol": 1E-4, "init_params": 'k-means++', "verbose": 0,
        #               "max_iter": 10000, "n_init": 20,
        #               "warm_start": False, "covariance_type": "full"}
        # self.GMM(n_repeats=1, start_frame=1, gmm_kwargs=gmm_kwargs)
        # hmm_kwargs = {"verbose": 0, "tol": 1E-4, "n_iter": 2000,
        #               "algorithm": "viterbi", "covariance_type": "full",
        #               "init_params": "st", "params": "stmc"}
        # self.HMM(n_repeats=1, start_frame=1, hmm_kwargs=hmm_kwargs)

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
        leaflet_resnames = self.unique_resnames

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
        leaflet_resnames = self.unique_resnames

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
        property_flatten = property_[:,
                           start_frame:].flatten()  # Shape change (NLipids, NFrames) -> (NLipids * NFrames,)

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
        leaflet_resnames = self.unique_resnames

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
        leaflet_resnames = self.unique_resnames

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
            # TODO Error on shape of means_ and vars_
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

    # ------------------------------ GETIS-ORD STATISTIC ------------------------------------------------------------- #
    def getis_ord_stat(self, weight_matrix, leaflet, lassign):

        """
        Getis-Ord Local Spatial Autocorrelation Statistics calculation based on the predicted order states
        of each lipid and the weighting factors between neighbored lipids.

        Be aware that this function is far from being elegant but it should do its job.

        Parameters
        ----------
        weight_matrix : numpy.ndarray
            Weight matrix for all lipid in a leaflet at current time step
        leaflet : int
            0 = upper leaflet, 1 = lower leafet
        lassign : numpy.ndarray
            Leaflet assignment for each molecule at each step of time

        Returns
        -------
        g_star_i : list of numpy.ndarrays
            G*i values for each lipid at each time step
        w_ii_all : list of numpy.ndarrays
            Self-influence of the lipids
        """

        # Initialize empty lists to store the G*i values and the self-influence of the lipids in a lipid
        g_star_i = []
        w_ii_all = []

        # Get the weightmatrix of the leaflet at the current time step

        # Number of lipids in the leaflet
        n = weight_matrix.shape[0]

        # In case the code was already executed beforehand
        weight_matrix[range(n), range(n)] = 0.

        # Get the order state of each lipid in the leaflet at the current time step
        # 1. Step: (lassign[ idxDPPC ][:, step] == leaflet) -> Which lipids are in the leaflet
        # 2. Step: orderDPPC[:, step][ ... ] --> What are their order parameters
        # TODO Make it deterministic
        do_dppc = []  # orderDPPC[:, step][lassign[idxDPPC][:, step] == leaflet]
        do_dipc = []  # orderDIPC[:, step][lassign[idxDIPC][:, step] == leaflet]
        do_chol = []  # orderCHOL[:, step][lassign[idxCHOL][:, step] == leaflet]

        # Put all the order states in one array -> The order of the lipids must be the same as in the system!!!
        order_states = np.concatenate([do_dppc, do_dipc, do_chol])

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

        # TODO Results for Clustering
        #  Should be saved in self.result
        g_star_i.append(g_star)
        w_ii_all.append(w_ii)

        return g_star_i, w_ii_all

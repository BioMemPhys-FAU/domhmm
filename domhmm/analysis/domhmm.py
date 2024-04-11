"""
LocalFluctuation --- :mod:`elbe.analysis.LocalFluctuation`
===========================================================

This module contains the :class:`LocalFluctuation` class.

"""

from .base import LeafletAnalysisBase
import numpy as np
import matplotlib.pyplot as plt
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

        # Initialize storage dictionary
        self.results.train_data_per_type = {}

        # Initalize weight matrix storage for each leaflet.
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
            # TODO Refactor this if - else branches
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

        for rsn in self.unique_resnames:
            # Create each leaflet's lipid types 3D empty array.
            # Array will fill with order parameters of tails and area per lipid
            num_tails = len(self.tails[rsn])
            self.results.train_data_per_type[f"{rsn}"] = [[] for _ in range(2)]

        # -------------------------------------------------------------

        # Fill dictionary with obtained data

        # Iterate over all residues in the selected membrane
        for resid in self.membrane_unique_resids:

            # Grab leaflet and resname
            rsn = getattr(self.results, f'id{resid}')['Resname']

            # Check if lipid is a sterol compound or not
            if rsn not in self.sterols:
                self.results.train_data_per_type[f"{rsn}"][0].append(resid)
                # Get area per lipid for specific residue
                apl = getattr(self.results, f'id{resid}')['APL']
                temp_result_array = [apl]
                # Iterate over chains -> For a normal phospholipid that should be 2
                for n_chain in range(len(self.tails[rsn])):
                    # Get individual lipid scc values for corresponding chain
                    indv_scc = getattr(self.results, f'id{resid}')[f'SCC_{n_chain}']
                    temp_result_array.append(indv_scc)

                self.results.train_data_per_type[f"{rsn}"][1].append(np.array(temp_result_array).transpose())

            elif rsn in self.sterols:
                pass

            # NOTHING?
            else:
                raise ValueError(f'{rsn} with resid {resid} not found in leaflets or sterol list!')

        # -------------------------------------------------------------

        # Transform lists to arrays
        for rsn in self.unique_resnames:
            # If number of tail of some residues are more than 2, this line will throw error
            self.results.train_data_per_type[f"{rsn}"][1] = np.array(self.results.train_data_per_type[f"{rsn}"][1])
        # -------------------------------------------------------------
        # TODO Add post-processing parts to here:
        #  GetisOrd
        #  Hierarchical Clustering
        gmm_kwargs = {"tol": 1E-4, "init_params": 'k-means++', "verbose": 0,
                      "max_iter": 10000, "n_init": 20,
                      "warm_start": False, "covariance_type": "full"}
        self.GMM(gmm_kwargs=gmm_kwargs)
        # TODO Change number of iterations to 1000 or 2000
        hmm_kwargs = {"verbose": False, "tol": 1E-4, "n_iter": 20,
                      "algorithm": "viterbi", "covariance_type": "full",
                      "init_params": "st", "params": "stmc"}
        self.HMM(hmm_kwargs=hmm_kwargs)


    # ------------------------------ FIT GAUSSIAN MIXTURE MODEL ------------------------------------------------------ #
    def GMM(self, gmm_kwargs):
        """
        Fit Gaussian Mixture

        Parameters
        ----------
        gmm_kwargs : dict
            Additional parameters for mixture.GaussianMixture
        """
        self.results["GMM"] = {}
        # Iterate over each residue and implement gaussian mixture model
        for res, data in self.results.train_data_per_type.items():
            gmm = mixture.GaussianMixture(n_components=2, **gmm_kwargs).fit(data[1].reshape(-1, 3))
            self.results["GMM"][res] = gmm

        # Check for convergence
        for resname, each in self.results["GMM"].items():
            if not each.converged_:
                print(f"{resname} Gaussian Mixture Model is not converged.")

    # ------------------------------ HIDDEN MARKOV MODEL ------------------------------------------------------------- #

    def HMM(self, hmm_kwargs):
        """
        Create Gaussian based Hidden Markov Models for each residue type
        Parameters
        ----------
        hmm_kwargs : dict
            Additional parameters for hmmlearn.hmm.GaussianHMM
        """
        self.results["HMM"] = {}
        # Iterate over each residue and implement gaussian-hidden markov model
        for resname, data in self.results.train_data_per_type.items():
            hmm = self.fit_hmm(data=data[1], gmm=self.results["GMM"][resname], hmm_kwargs=hmm_kwargs, n_repeats=2)
            self.results["HMM"][resname] = hmm
        # TODO Plot hidden markov model tolerance graph in verbose option
        # self.plot_hmm_result()
        # TODO Decide how to validate HMM (checking result models' means?)

        # Make predictions based on HMM model
        self.predict_states()
    def fit_hmm(self, data, gmm, hmm_kwargs, n_repeats=10, dim=3):

        """
        Fit several HMM models to the data and return the best one.

        Parameters
        ----------
        data : numpy.ndarray
            Data of the single lipid properties for one lipid type at each time step
        gmm : GaussianMixture Model
            Scikit-learn object
        n_repeats : int
            Number of independent fits
        dim : int
            Dimension of lipid property space

        hmm_kwargs: dict
            Additional parameters for Hidden Markov Model

        Returns
        -------
        best_ghmm : GaussianHMM
            hmmlearn object

        """

        # Specify the length of the sequence for each lipid
        n_lipids = data.shape[0]
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
        for resname, ghmm in self.results['HMM'].items():
            plt.semilogy(np.arange(len(ghmm.monitor_.history) - 1), np.diff(np.array(ghmm.monitor_.history)),
                          ls="-", label=resname, lw=2)
        plt.legend(fontsize=15)
        plt.semilogy(np.arange(100), np.repeat(1E-4, 100), color="k", ls="--", lw=2)
        plt.xlim(0, 100)
        plt.ylim(1E-5, 15E5)
        plt.ylabel(r"$\Delta(log(\hat{L}))$", fontsize=18)
        plt.xlabel("Iterations", fontsize=18)
        plt.tick_params(axis="both", labelsize=11)
        plt.text(s="Tolerance 1e-4", x=1, y=1E-4 + 0.00005, color="k", fontsize=15)
        plt.title("a", fontsize=20, fontweight="bold", loc="left")
        plt.show()

    def predict_states(self):
        self.results['HMM_Pred'] = {}
        for resname, data in self.results.train_data_per_type.items():
            shape = data[1].shape
            hmm = self.results['HMM'][resname]
            # Lengths consists of number of frames and number of residues
            lengths = np.repeat(shape[1], shape[0])
            prediction = hmm.predict(data[1].reshape(-1, shape[2]), lengths=lengths).reshape(shape[0], shape[1])
            # Save prediction result of each residue
            self.results['HMM_Pred'][resname] = prediction


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

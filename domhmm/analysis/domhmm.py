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

        # Initialize result storage dictionaries
        self.results.train_data_per_type = {}
        self.results.GMM = {}
        self.results.HMM = {}
        self.results.HMM_Pred = {}
        self.results.Getis_Ord = {}

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

            # Init results for order parameters -> For each resid we should have an array containing the order
            # parameters for each frame
            setattr(self.results, f'id{resid}', {})  # -> Setup an empty dictionary
            getattr(self.results, f'id{resid}')['Resname'] = resname  # -> Store lipid type

            # Iterate over leaflet tails
            n_tails = len(self.tails[resname])
            for i in range(n_tails):
                # Init storage for SCC values for each lipid
                getattr(self.results, f'id{resid}')[f'SCC_{i}'] = np.zeros(self.n_frames, dtype=np.float32)

            # Store the area per lipid for each lipid
            getattr(self.results, f'id{resid}')[f'APL'] = np.zeros(self.n_frames, dtype=np.float32)

            # Check leaflet assignment -> based on RESID
            if resid in self.leaflet_selection["0"].resids and resid not in self.leaflet_selection["1"].resids:
                getattr(self.results, f'id{resid}')['Leaflet'] = 0  # -> Store information about leaflet assignment
            elif resid in self.leaflet_selection["1"].resids and resid not in self.leaflet_selection["0"].resids:
                getattr(self.results, f'id{resid}')['Leaflet'] = 1  # -> Store information about leaflet assignment
            else:
                raise ValueError(f'{resname} with resid {resid} not found in leaflets')

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
            # Array will fill with order parameters of tails, area per lipid and leaflet information
            num_tails = len(self.tails[rsn])
            self.results.train_data_per_type[f"{rsn}"] = [[] for _ in range(3)]

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
                # Add order parameter list and take transpose of it for HMM and GMM training requirements
                self.results.train_data_per_type[f"{rsn}"][1].append(np.array(temp_result_array).transpose())
                # Add leaflet information
                self.results.train_data_per_type[f"{rsn}"][2].append(self.results[f'id{resid}']["Leaflet"])

            elif rsn in self.sterols:
                pass

            # NOTHING?
            else:
                raise ValueError(f'{rsn} with resid {resid} not found in leaflets or sterol list!')

        # -------------------------------------------------------------

        # Transform lists to arrays
        for rsn in self.unique_resnames:
            # If number of tail of some residues are more than 2, this line will throw error
            self.results.train_data_per_type[f"{rsn}"][0] = np.array(self.results.train_data_per_type[f"{rsn}"][0])
            self.results.train_data_per_type[f"{rsn}"][1] = np.array(self.results.train_data_per_type[f"{rsn}"][1])
            self.results.train_data_per_type[f"{rsn}"][2] = np.array(self.results.train_data_per_type[f"{rsn}"][2])
        # -------------------------------------------------------------
        gmm_kwargs = {"tol": 1E-4, "init_params": 'k-means++', "verbose": 0,
                      "max_iter": 10000, "n_init": 20,
                      "warm_start": False, "covariance_type": "full"}
        self.GMM(gmm_kwargs=gmm_kwargs)
        hmm_kwargs = {"verbose": False, "tol": 1E-4, "n_iter": 1000,
                      "algorithm": "viterbi", "covariance_type": "full",
                      "init_params": "st", "params": "stmc"}
        self.HMM(hmm_kwargs=hmm_kwargs)
        self.getis_ord()
        self.clustering()

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
        # Iterate over each residue and implement gaussian-hidden markov model
        for resname, data in self.results.train_data_per_type.items():
            hmm = self.fit_hmm(data=data[1], gmm=self.results["GMM"][resname], hmm_kwargs=hmm_kwargs, n_repeats=2)
            self.results["HMM"][resname] = hmm
        # Plot result of hmm
        self.plot_hmm_result()

        # Make predictions based on HMM model
        self.predict_states()
        # Validate states and result prediction
        self.state_validate()
        # Plot prediction result
        self.predict_plot()

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
        for resname, gmm in self.results["HMM"].items():
            means = gmm.means_
            diff_percents = (means[1, 0] - means[0, 0]) / means[0, 0]
            if diff_percents > 0.1:
                self.results['HMM_Pred'][resname] = np.abs(self.results['HMM_Pred'][resname] - 1)

    def predict_plot(self):
        t = np.linspace(8, 10, self.n_frames)
        for resname in self.unique_resnames:
            plt.plot(t, self.results['HMM_Pred'][resname].mean(0), label=resname)
        plt.xticks([8, 8.5, 9, 9.5, 10])
        plt.xlabel(r"t ($\mu$s)", fontsize=18)
        plt.ylabel(r"$\bar{O}_{Lipid}$", fontsize=18)
        plt.legend(fontsize=15, ncols=1, loc="lower left")
        plt.ylim(0, 1)
        plt.xlim(8, 10)
        plt.title("b", fontsize=20, fontweight="bold", loc="left")
        plt.show()

    # ------------------------------ GETIS-ORD STATISTIC ------------------------------------------------------------- #
    def getis_ord(self):
        self.getis_ord_stat(self.results["upper_weight_all"], 0)
        self.getis_ord_stat(self.results["lower_weight_all"], 1)
        self.getis_ord_plot()

    def getis_ord_stat(self, weight_matrix_all, leaflet):
        """
            Getis-Ord Local Spatial Autocorrelation Statistics calculation based on the predicted order states
            of each lipid and the weighting factors between neighbored lipids.

            Parameters
            ----------
            weight_matrix_all : numpy.ndarray
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
        # Get the number of frames
        nframes = self.n_frames

        # Initialize empty lists to store the G*i values and the self-influence of the lipids in a lipid
        g_star_i = []
        w_ii_all = []

        # Iterate over frames
        for step in range(nframes):
            # Get the weightmatrix of the leaflet at the current time step
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

            g_star_i.append(g_star)
            w_ii_all.append(w_ii)
        self.results['Getis_Ord'][leaflet] = {f"g_star_i_{leaflet}": g_star_i, f"w_ii_{leaflet}": w_ii_all}

    def getis_ord_plot(self):
        resnum = len(self.unique_resnames)
        g_star_i_temp = [[] for _ in range(resnum)]
        for step in range(self.n_frames):
            index_dict_0 = self.get_leaflet_step_order_index(leaflet=0)
            index_dict_1 = self.get_leaflet_step_order_index(leaflet=1)
            temp_index_list_0 = [0]
            temp_index_list_1 = [0]
            for resname in self.unique_resnames:
                temp_index_list_0.append(temp_index_list_0[-1] + len(index_dict_0[resname]) - 1)
                temp_index_list_1.append(temp_index_list_1[-1] + len(index_dict_1[resname]) - 1)
            for i in range(resnum):
                g_star_i_temp[i] += list(np.append(self.results['Getis_Ord'][0]['g_star_i_0'][step]
                                                   [temp_index_list_0[i]:temp_index_list_0[i + 1]],
                                                   self.results['Getis_Ord'][1]['g_star_i_1'][step]
                                                   [temp_index_list_1[i]:temp_index_list_1[i + 1]]))

        for i in range(resnum):
            plt.hist(g_star_i_temp[i], bins=np.linspace(-3, 3, 201), density=True, histtype="step", lw=2,
                     label=self.unique_resnames[i])

        plt.legend(fontsize=15, loc="upper left", ncols=2)

        plt.xlim(-3, 3)
        plt.ylim(0, .9)

        xl = plt.xlabel("$G^*_i$", fontsize=18)
        plt.ylabel("$p(G^*_i)$", fontsize=18)
        plt.tick_params(labelsize=11)

        plt.title("a", fontsize=20, fontweight="bold", loc="left")
        plt.show()

    def permut_getis_ord_stat(self, weight_matrix_all, leaflet):

        """
        Getis-Ord Local Spatial Autocorrelation Statistics calculation based on the predicted order states
        of each lipid and the weighting factors between neighbored lipids.

        Parameters
        ----------
        weight_matrix_all : numpy.ndarray
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

    # ------------------------------ HIERARCHICAL CLUSTERING --------------------------------------------------------- #
    def clustering(self):
        """
        Runs hierarchical clustering and plots clustering results in different frames.
        """
        # TODO Decide on which frames to plot
        frame_list = [3, 50, 98]
        fig, ax = plt.subplots(1, len(frame_list), figsize=(20, 5))

        # Iterate over three frames illustrate the clustering results
        for k, i in enumerate(frame_list):
            order_states_0 = self.get_leaflet_step_order(0, i)

            # Clustering
            # ----------------------------------------------------------------------------------------------------------------------
            core_lipids = self.assign_core_lipids(weight_matrix_f=self.results["upper_weight_all"][i],
                                                  g_star_i_f=self.results['Getis_Ord'][0]['g_star_i_0'][i],
                                                  order_states_f=order_states_0,
                                                  w_ii_f= self.results["Getis_Ord"][0]["w_ii_0"][i])

            clusters = self.hierarchical_clustering(weight_matrix_f=self.results["upper_weight_all"][i],
                                                    w_ii_f=self.results["Getis_Ord"][0]["w_ii_0"][i],
                                                    core_lipids=core_lipids)

            # Plot coordinates
            # ----------------------------------------------------------------------------------------------------------------------
            residue_indexes = self.get_leaflet_step_order_index(leaflet = 0)
            positions = self.leaflet_selection['0'].positions
            for resname, index in residue_indexes.items():
                ax[k].scatter(positions[index, 0],
                          positions[index, 1], marker="s", alpha=1, s=5, label=resname)

            # Choose color scheme for clustering coloring
            colors = plt.cm.viridis_r(np.linspace(0, 1.0, len(clusters.values())))

            # Iterate over clusters and plot the residues
            print(f"Number of clusters in frame {i}: {len(clusters.values())}")
            for j, val in enumerate(clusters.values()):
                idx = np.array(list(val), dtype=int)
                ax[k].scatter(positions[idx, 0],
                              positions[idx, 1],
                              s=100, marker="o", color=colors[j], zorder=-10)

            # Plot cosmetics
            ax[k].set_ylim(-5, 138)
            ax[k].set_xlim(-5, 138)

            ax[k].set_xticks([])
            ax[k].set_yticks([])

            ax[1].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=15, frameon=False)

            ax[k].set_aspect("equal")

        plt.subplots_adjust(wspace=-0.45)
        ax[0].set_title("a", fontsize=20, fontweight="bold", loc="left")
        ax[1].set_title("b", fontsize=20, fontweight="bold", loc="left")
        ax[2].set_title("c", fontsize=20, fontweight="bold", loc="left")

        ax[0].text(s=r"$t=8\, \mu s$", x=71.5, y=144, fontsize=18, ha="center", va="center")
        ax[1].text(s=r"$t=9\, \mu s$", x=71.5, y=144, fontsize=18, ha="center", va="center")
        ax[2].text(s=r"$t=10\, \mu s$", x=71.5, y=144, fontsize=18, ha="center", va="center")

        plt.show()

    def assign_core_lipids(self, weight_matrix_f, g_star_i_f, order_states_f, w_ii_f):

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


        Returns
        -------
        core_lipids : numpy.ndarray (bool)
           Contains a TRUE value if the lipid is a core member, otherwise it FALSE
        """

        # Define boundary of the rection region
        z1_a = 2.017  # 1.750 #1.307
        z_a = -1.271

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

    def hierarchical_clustering(self, weight_matrix_f, w_ii_f, core_lipids):

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
                if id_i not in clusters.keys(): continue

                # The cluster weights are defined as the sum
                # over the weights of all lipid members
                cluster_weights_i = np.sum(weight_matrix_f[clusters[id_i]], axis=0)

                # Compare cluster weights to the self-influence of each lipid
                merge_condition_i = cluster_weights_i > w_ii_f

                # Iterate over all clusters j
                for id_j in cluster_ids[(i + 1):]:

                    # Do not merge a cluster with itself
                    if id_i == id_j: continue
                    # If cluster j was already merged and deleted, skip it!
                    if id_j not in clusters.keys(): continue

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
        leaflet : numpy.ndarray
            leaflet index
        step: numpy.ndarray
            step index

        Returns
        -------
        order_states : numpy.ndarray
            Numpy array contains order state results of the leaflet at step in order of system's residues
        """
        temp = []
        for res, data in self.results.train_data_per_type.items():
            temp.append(self.results["HMM_Pred"][res][:, step][data[2] == leaflet])

        order_states = np.concatenate(temp)
        return order_states

    def get_leaflet_step_order_index(self, leaflet):
        """
        Receive residue's indexes in order state result with respect to the leaflet

        Parameters
        ----------
        leaflet : numpy.ndarray
            leaflet index
        step: numpy.ndarray
            step index

        Returns
        -------
        order_states : numpy.ndarray
            Numpy array contains residue indexes of the leaflet at step in order of system's residues
        """
        result = {}
        for res, data in self.results.train_data_per_type.items():
            indexes = data[0][data[2] == leaflet]
            # Decreasing one since Python array index system
            result[res] = indexes - 1
        return result

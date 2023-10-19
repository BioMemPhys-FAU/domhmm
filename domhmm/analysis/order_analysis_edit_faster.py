import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import leaflet
from MDAnalysis.analysis import distances
from numba import jit
from tqdm import tqdm
import networkx as nx
import argparse
import sys

sigma = 8
cadelx = 2
cadely = 2

#-----------------------------Function for analysis-----------------------------#

def lipid_parser(path2file, name):

    chain2_1 = {}
    chain2_2 = {}
    chain2_c = {}

    chain3_1 = {}
    chain3_2 = {}
    chain3_c = {}

    with open(path2file + '/' + name + '.dat', 'r') as lipid:

        for line in lipid:

            list_ = line.strip().split(' ')

            if "[" in line and "]" in line: 
                chain = list_[1]
            elif len(line) == 1: pass
            else:

                sele_string_1 = f'(resname ' + name.upper() + f' and name {list_[1]})'
                sele_string_2 = f'(resname ' + name.upper() + f' and name {list_[2]})'
                sele_string_c = f'(resname ' + name.upper() + f' and name {list_[0]})'

                if chain == 'chain2': 
                    chain2_1[list_[0]] = sele_string_1
                    chain2_2[list_[0]] = sele_string_2
                    chain2_c[list_[0]] = sele_string_c
                elif chain == 'chain3': 
                    chain3_1[list_[0]] = sele_string_1
                    chain3_2[list_[0]] = sele_string_2
                    chain3_c[list_[0]] = sele_string_c

                else: print('Gotta problem over here!')

    return chain2_1, chain2_2, chain2_c, chain3_1, chain3_2, chain3_c


def calc_normal(cloud):

        mean_point_cloud = np.sum(cloud, 0) / len(cloud)
        cloud = cloud - mean_point_cloud

        M = np.cov(cloud.T) / cloud.shape[0]
        eigvals, eigvec = np.linalg.eig(M)

        min_eigval = np.argmin(eigvals)
        min_eig_vec = eigvec[:, min_eigval]

        normal = min_eig_vec / np.linalg.norm(min_eig_vec)

        return normal


@jit(nopython = True, fastmath = True)
def order_math(c_h1, c_h2, c_c, normal):

    vec_c1 = c_h1 - c_c
    vec_c2 = c_h2 - c_c

    #Norm them...
    norm_vec_c1 = np.sqrt(np.sum(vec_c1**2, axis = 1))
    norm_vec_c2 = np.sqrt(np.sum(vec_c2**2, axis = 1))

    i = 0
    for vec1, vec2, nvec1, nvec2 in zip(vec_c1, vec_c2, norm_vec_c1, norm_vec_c2):
        vec_c1[i] /= nvec1
        vec_c2[i] /= nvec2
        i += 1

    #...and calculate the angle between them
    dot_c1 = np.sum(vec_c1 * normal, 1)
    dot_c2 = np.sum(vec_c2 * normal, 1)

    #Angle in radians
    theta_c1 = np.arccos(dot_c1)
    theta_c2 = np.arccos(dot_c2)

    #Input for np.cos in radians
    s_ch_c1 = (3 * np.cos(theta_c1)**2 - 1) / 2
    s_ch_c2 = (3 * np.cos(theta_c2)**2 - 1) / 2

    #Average the order parameter for two hydrogens
    s_ch_c = (s_ch_c1 + s_ch_c2) / 2

    return s_ch_c

#@jit(nopython = True, fastmath = True)
def calc_order(uni, head_selection, head_coms, r1, acyls, Lx, Ly, Lz):


    uniq_resids = np.unique( head_selection.resids )
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    
    scd_2 = np.zeros(( len(uniq_resids), 14))
    scd_3 = np.zeros(( len(uniq_resids), 14))
 #   coms = np.zeros((len(uniq_resids), 3))

    coms = head_selection.center_of_mass(compound = "residues")

    for i, resid in enumerate(uniq_resids):

        #Get selection only for the specific resid and the specific lipid resname
 #       resid_selection = uni.select_atoms(f'resid {resid}') & head_selection
#
  #      resn = resid_selection.center_of_mass()
   #     coms[i] = np.copy(resn)
#        
#        #---------------------------------------------POINT CLOUD---------------------------------------------#
#    
#        dist = distance(x0 = resn, x1 = head_coms, dimensions = bs[0:3])
#
#        point_cloud = head_coms[dist <= r1]
#
#        point_cloud[:, 0] = np.where(point_cloud[:, 0] - resn[0] > Lx/2, point_cloud[:, 0] - Lx, point_cloud[:, 0])
#        point_cloud[:, 1] = np.where(point_cloud[:, 1] - resn[1] > Ly/2, point_cloud[:, 1] - Ly, point_cloud[:, 1])
#        point_cloud[:, 2] = np.where(point_cloud[:, 2] - resn[2] > Lz/2, point_cloud[:, 2] - Lz, point_cloud[:, 2])
#
#        point_cloud[:, 0] = np.where(point_cloud[:, 0] - resn[0] < -1 * Lx/2, point_cloud[:, 0] + Lx, point_cloud[:, 0])
#        point_cloud[:, 1] = np.where(point_cloud[:, 1] - resn[1] < -1 * Ly/2, point_cloud[:, 1] + Ly, point_cloud[:, 1])
#        point_cloud[:, 2] = np.where(point_cloud[:, 2] - resn[2] < -1 * Lz/2, point_cloud[:, 2] + Lz, point_cloud[:, 2])
#
#        dists = np.sqrt(np.sum( (point_cloud - resn)**2 , 1))
#        max_dist = dists.max()
#        min_dist = dists.min()
#
#        assert max_dist <= r1, 'Scary. Max distance in point cloud is larger than cutoff'
#        assert np.allclose(min_dist, 0.0), 'Central point is not in point cloud.'
#
#        #---------------------------------------------NORMAL CALCULATION---------------------------------------------#
#
#        normal = calc_normal(cloud = np.copy(point_cloud))
#
#        #ax.set_aspect('equal')
#        #ax.scatter3D(head_coms[:, 0], head_coms[:, 1], head_coms[:, 2], color = 'green')
#        #ax.scatter3D(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color = 'red')
#        #ax.scatter3D(resn[0], resn[1], resn[2], color = 'blue')
#        #ax.quiver(resn[0], resn[1], resn[2], resn[0] + normal[0], resn[1] + normal[1], resn[2] + normal[2], color = 'orange')
#        #ax.scatter(head_coms[:, 0], head_coms[:, 1], color = 'green')
#        #ax.scatter(point_cloud[:, 0], point_cloud[:, 1], color = 'red')
#        #ax.scatter(resn[0], resn[1], color = 'blue')
#
#        #plt.show()
#
#        normal = normal.astype(dtype = np.float32, casting = "same_kind", copy = True).reshape(1, -1)

        normal = np.array([0., 0., 1.]).reshape(1, -1)

        c_h21 = acyls[resid]["H21"].positions
        c_h22 = acyls[resid]["H22"].positions
        c_h2c = acyls[resid]["C2"].positions

        c_h31 = acyls[resid]["H31"].positions
        c_h32 = acyls[resid]["H32"].positions
        c_h3c = acyls[resid]["C3"].positions

#        print(acyls[resid])


        s_ch_c2 = order_math(c_h1 = c_h21, c_h2 = c_h22, c_c = c_h2c, normal = normal)
        s_ch_c3 = order_math(c_h1 = c_h31, c_h2 = c_h32, c_c = c_h3c, normal = normal)

        scd_2[i] = s_ch_c2
        scd_3[i] = s_ch_c3

    #print(np.allclose(coms, coms_test))


    return coms, scd_2, scd_3


def get_distances(pos, boxdim, molnum):
    """
    Calculate a distance matrix
    pos := Array with cartesian coordinates
    boxdim := Box dimensions in xyz
    molnum := Number of peptides in system
    """
    distmat = np.zeros((molnum, molnum))
    for i in range(molnum):
        for j in range(i+1, molnum):
            dist = distances.distance_array(pos[i], pos[j], box = boxdim)
            distmat[i,j]=distmat[j,i]=dist[0][0]
    return distmat

@jit(nopython = True, fastmath = True)
def distance(x0, x1, dimensions):

    """
    Vectorized distance calculation under periodic boundary conditions

    x0 := Array with cartesian coordinates
    x1 := Array with cartesian coordinates
    dimensions := Box dimensions

    Source: https://stackoverflow.com/questions/11108869/optimizing-python-distance-calculation-while-accounting-for-periodic-boundary-co
    """

    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

@jit(nopython = True, fastmath = True)
def periodic_cog(pos, boxdim):

    """
    Calculate the center of geometry over periodic conditions

    pos := Array with cartesian coordinates
    boxdim := Box dimensions in xyz
    """

    #Transform coordinates to radians
    theta = 2 * np.pi * pos / boxdim

    n = len(theta)

    #Put them on an unit circle
    xi = np.cos(theta)
    zeta = np.sin(theta)

    #Calculate mean of spherical coordinates
    xi_mean = np.sum(xi, 0) / n #np.mean(xi, 0)
    zeta_mean = np.sum(zeta, 0) / n

    #Get angle of calculated mean
    theta_mean = np.arctan2(-zeta_mean, -xi_mean) + np.pi

    #Transform it back to cartesian coordinates
    cog = boxdim * theta_mean / 2 / np.pi

    return cog



#-----------------------------Load stuff-----------------------------#

parser = argparse.ArgumentParser(description="Lipid environment of transmembrane peptides.")
parser.add_argument("-f", dest = "traj", help='Please insert a trajectory. (*.xtc, *.pdb, *.gro, *.trr)', type = str)
parser.add_argument('-s', dest='struc', help='Please insert a structure file. (*.gro, *.tpr, *pdb)', type = str)
parser.add_argument("-r", dest = "r_co", help="Cutoff for local surface.", type = int)
parser.add_argument("-m", dest = "molnum", help="Number of peptides.", type = int)
parser.add_argument("-l", dest = "path2lipids", help="Path to lipids.", type = str)
parser.add_argument('-o', dest='out', help='Main name for output files (e.g. <name>_*.npy).', type = str)   
parser.add_argument('-b', dest='begin', help='Start frame for analysis.', type = int)   
parser.add_argument('-e', dest='end', help='End frame for analysis.', type = int)   
parser.add_argument('-dt', dest='dt', help='Skipping frames.', type = int)   
args = parser.parse_args()

#-----------------------------Load more stuff-----------------------------#

u = mda.Universe(args.struc,args.traj)  # always start with a Universe

print("Loaded trajectory!")
print("")

#Create selection string
protein = u.select_atoms("name CA")
n_bb    = len(protein)
nacc    = n_bb//args.molnum
ccutoff = 25 #Cluster distance cutoff

print(protein.resnames)

head_pc = "(name C11 or name C12 or name C13 or name C14 or name C15 or name N)"
head_pe = "(name C11 or name C12 or name N)"
head_ps = "(name C11 or name C12 or name C13 or name N or name O13A or name O13B)"
head_sapi = "(name C11 or name C12 or name C13 or name C14 or name C15 or name C16 or name O2 or name O3 or name O4 or name O5 or name O6 or name P3 or name OP32 or name OP33 or name OP34 or name P5 or name OP52 or name OP53 or name OP54)"

#Test case
#head_pc = "(name P)"
#head_pe = "(name P)"
#head_ps = "(name P)"
#head_sapi = "(name P)" 


head_lipids = {"PSM":head_pc,"PAPC":head_pc,"NSM":head_pc,"LSM":head_pc,"SOPC":head_pc,"PLPC":head_pc,"SAPS":head_ps,"PLA20":head_pe,"DPPC":head_pc,"PAPS":head_ps,"PDOPE":head_pe,"POPC":head_pc,"POPE":head_pe,"SAPE":head_pe,"SAPI2A":head_sapi} 

no_lipids_first = {"PLPC":22,"SOPC":10,"PAPC":8,"SAPS":2,"PSM":18,"NSM":14,"LSM":12,"PLA20":4}
no_lipids_second = {"PLPC":13,"POPC":5,"DPPC":3,"POPE":3,"PDOPE":11,"SAPE":5,"PAPS":19,"SAPS":2,"SAPI2A":3,"PSM":2,"PLA20":17}

#-----------------------------Acyl chains-----------------------------#

acyl_lipids_chain2_1 = {}
acyl_lipids_chain2_2 = {}
acyl_lipids_chain2_c = {}

acyl_lipids_chain3_1 = {}
acyl_lipids_chain3_2 = {}
acyl_lipids_chain3_c = {}

for lipid in head_lipids.keys(): acyl_lipids_chain2_1[lipid], acyl_lipids_chain2_2[lipid], acyl_lipids_chain2_c[lipid], acyl_lipids_chain3_1[lipid], acyl_lipids_chain3_2[lipid], acyl_lipids_chain3_c[lipid] = lipid_parser(path2file = args.path2lipids, name = lipid.lower())

#-----------------------------Assign leaflets-----------------------------#

#Create lipid selection and assign leaflets
po4 = u.select_atoms("name P")

#Assign leaflets
leaf = leaflet.LeafletFinder(u, "name P",  pbc = True)
pho_first = leaf.group(0)
pho_second = leaf.group(1)

#Unique lipid names in each leaflet
uniq_resnames_first  = np.unique(pho_first.resnames)
uniq_resnames_second = np.unique(pho_second.resnames)

#Unique lipid ids in each leaflet
resids_first  = np.unique(pho_first.resids)
resids_second = np.unique(pho_second.resids)

#Empty selection for headgroups
heads_first = u.select_atoms("")
heads_second = u.select_atoms("")

acyls_first = {}
acyls_second = {}

#Iterate over resids in first leaflet
print("Prepare selections for the first leaflet!")
for resid_first in tqdm(resids_first):

    #Get resname from resid
    resname_first = np.unique(u.select_atoms("resid {}".format(resid_first)).resnames)

    #Check if only one lipid was selected
    assert len(resname_first) == 1, "More than one lipid found!"

    #Get resname
    resname_first = resname_first[0]

    #Check if resname is really in leaflet
    assert resname_first in uniq_resnames_first, "Lipid not in leaflet!"

    #Prepare selection string
    sele = head_lipids[resname_first] + " and " + "resid {}".format(resid_first)

    #Make selection
    mda_sele = u.select_atoms(sele)

    #Check if the right number of atoms was selected
    assert head_lipids[resname_first].count("or") + 1 == mda_sele.n_atoms, "Not all atoms are selected!"
    
    #Append selection
    heads_first += mda_sele #-> Is unordered

    if resname_first not in acyls_first.keys(): acyls_first[resname_first] = {}
    else: pass

    if resid_first not in acyls_first[resname_first].keys(): acyls_first[resname_first][resid_first] = {}
    else: print("Double resid!")


    sh21, sh22, sc2 = u.select_atoms(""), u.select_atoms(""), u.select_atoms("")
    sh31, sh32, sc3 = u.select_atoms(""), u.select_atoms(""), u.select_atoms("")

    for h21, h22, c2, h31, h32, c3 in zip(acyl_lipids_chain2_1[resname_first].values(), acyl_lipids_chain2_2[resname_first].values(), acyl_lipids_chain2_c[resname_first].values(), acyl_lipids_chain3_1[resname_first].values(), acyl_lipids_chain3_2[resname_first].values(), acyl_lipids_chain3_c[resname_first].values()):

        sh21 += u.select_atoms(h21)
        sh22 += u.select_atoms(h22)
        sc2 += u.select_atoms(c2)

        sh31 += u.select_atoms(h31)
        sh32 += u.select_atoms(h32)
        sc3 += u.select_atoms(c3)

    first_resid =  u.select_atoms(f"resid {resid_first}")

    acyls_first[resname_first][resid_first]["H21"] = sh21 & first_resid
    acyls_first[resname_first][resid_first]["H22"] = sh22 & first_resid    
    acyls_first[resname_first][resid_first]["C2"] = sc2 & first_resid    

    acyls_first[resname_first][resid_first]["H31"] = sh31 & first_resid    
    acyls_first[resname_first][resid_first]["H32"] = sh32 & first_resid    
    acyls_first[resname_first][resid_first]["C3"] = sc3 & first_resid    

#Iterate over resids in second leaflet
print("Prepare selections for the second leaflet!")
for resid_second in tqdm(resids_second):
    
    #Get resname from resid
    resname_second = np.unique(u.select_atoms("resid {}".format(resid_second)).resnames)
    
    #Check if only one lipid was selected
    assert len(resname_second) == 1, "More than one lipid found!"

    #Get resname
    resname_second = resname_second[0]
    
    #Check if resname is really in leaflet
    assert resname_second in uniq_resnames_second, "Lipid not in leaflet!"

    #Prepare selection string
    sele = head_lipids[resname_second] + " and " + "resid {}".format(resid_second)

    #Make selection
    mda_sele = u.select_atoms(sele)
    
    #Check if the right number of atoms was selected
    assert head_lipids[resname_second].count("or") + 1 == mda_sele.n_atoms, "Not all atoms are selected!"

    #Append selection #-> Is unordered
    heads_second += mda_sele

    if resname_second not in acyls_second.keys(): acyls_second[resname_second] = {}
    else: pass

    if resid_second not in acyls_second[resname_second].keys(): acyls_second[resname_second][resid_second] = {}
    else: print("Double resid!")


    sh21, sh22, sc2 = u.select_atoms(""), u.select_atoms(""), u.select_atoms("")
    sh31, sh32, sc3 = u.select_atoms(""), u.select_atoms(""), u.select_atoms("")

    for h21, h22, c2, h31, h32, c3 in zip(acyl_lipids_chain2_1[resname_second].values(), acyl_lipids_chain2_2[resname_second].values(), acyl_lipids_chain2_c[resname_second].values(), acyl_lipids_chain3_1[resname_second].values(), acyl_lipids_chain3_2[resname_second].values(), acyl_lipids_chain3_c[resname_second].values()):

        sh21 += u.select_atoms(h21)
        sh22 += u.select_atoms(h22)
        sc2 += u.select_atoms(c2)

        sh31 += u.select_atoms(h31)
        sh32 += u.select_atoms(h32)
        sc3 += u.select_atoms(c3)

    sec_resid =  u.select_atoms(f"resid {resid_second}")

    acyls_second[resname_second][resid_second]["H21"] = sh21  & sec_resid     
    acyls_second[resname_second][resid_second]["H22"] = sh22  & sec_resid     
    acyls_second[resname_second][resid_second]["C2"] = sc2  & sec_resid     

    acyls_second[resname_second][resid_second]["H31"] = sh31  & sec_resid     
    acyls_second[resname_second][resid_second]["H32"] = sh32  & sec_resid     
    acyls_second[resname_second][resid_second]["C3"] = sc3  & sec_resid    

#for keys, vals in zip(acyls_second["SAPS"].keys(), acyls_second["SAPS"].values()):
#    print(vals)

with open(args.out + "_resnames_first.dat", "w") as W:

    for i, name in enumerate(uniq_resnames_first): W.write(f"{i:5d}-{name:10s}\n")

with open(args.out + "_resnames_second.dat", "w") as W:

    for i, name in enumerate(uniq_resnames_second): W.write(f"{i:5d}-{name:10s}\n")

#-----------------------------Assign storage-----------------------------#

store_sdc_2_first = {}
store_sdc_3_first = {}

for lipid1 in uniq_resnames_first:
    
    store_sdc_2_first[lipid1] = np.zeros( (len(u.trajectory[args.begin:args.end:args.dt]), no_lipids_first[lipid1], len(acyl_lipids_chain2_1[lipid1].keys()) + 1))
    store_sdc_3_first[lipid1] = np.zeros( (len(u.trajectory[args.begin:args.end:args.dt]), no_lipids_first[lipid1], len(acyl_lipids_chain3_1[lipid1].keys()) + 1))

store_sdc_2_second = {}
store_sdc_3_second = {}

for lipid2 in uniq_resnames_second:
    
    store_sdc_2_second[lipid2] = np.zeros( (len(u.trajectory[args.begin:args.end:args.dt]), no_lipids_second[lipid2], len(acyl_lipids_chain2_1[lipid2].keys()) + 1 ))
    store_sdc_3_second[lipid2] = np.zeros( (len(u.trajectory[args.begin:args.end:args.dt]), no_lipids_second[lipid2], len(acyl_lipids_chain3_1[lipid2].keys()) + 1 ))



#---------------------Start analysis---------------------#

for step, ts in tqdm(enumerate(u.trajectory[args.begin:args.end:args.dt]), total = len(u.trajectory[args.begin:args.end:args.dt])):

    #Get protein positions -> Clip atom coordinates to actual box size
    bb_pos = protein.positions % ts.dimensions[0:3]
    bb_xy  = bb_pos[:, 0:2]

    Lx, Ly, Lz = ts.dimensions[0], ts.dimensions[1], ts.dimensions[2]

    #Get positions of beads in upper and lower leaflet and for cholesterol
    pho_first_pos  = pho_first.positions % ts.dimensions[0:3]
    pho_second_pos = pho_second.positions % ts.dimensions[0:3]

    #---------------------Protein analysis---------------------#

    #Store protein center of geometry
    bb_xyz_cog = np.zeros((args.molnum, 3))
    bb_xyz_cog_start = np.zeros((args.molnum, 3))
    bb_xyz_cog_end = np.zeros((args.molnum, 3))

    #Iterate over tmds
    for i in range(0, n_bb, nacc):

        #Get tmd index
        idx = i//nacc

        #Calculate center of geometry over periodic boundary conditions
        bb_cog = periodic_cog(pos=bb_pos[i:i+nacc], boxdim=ts.dimensions[0:3])

        #Calculate center of geometry over periodic boundary conditions for the first three backbone beads
        bb_cog_start = periodic_cog(pos = bb_pos[i:i+nacc][-30:-27], boxdim=ts.dimensions[0:3])   
        #Calculate center of geometry over periodic boundary conditions for the last three backbone beads
        bb_cog_end = periodic_cog(pos = bb_pos[i:i+nacc][-3:], boxdim=ts.dimensions[0:3])

        #Store periodic center of geometries
        bb_xyz_cog[idx] = bb_cog
        bb_xyz_cog_start[idx] = bb_cog_start
        bb_xyz_cog_end[idx] = bb_cog_end

        #Project to xy plane
        xy_cog = np.array([bb_cog[0], bb_cog[1], 0.0]).reshape((1, 3))
        xy_cog_start = np.array([bb_cog_start[0], bb_cog_start[1], 0.0]).reshape((1, 3))
        xy_cog_end = np.array([bb_cog_end[0], bb_cog_end[1], 0.0]).reshape((1, 3))


        #Check if start/end termini coincide with leaflets -> Done

        #---------------------First leaflet---------------------#

        heads_first_com = heads_first.center_of_mass(compound = 'residues')
        heads_second_com = heads_second.center_of_mass(compound = 'residues')

        assert heads_first_com.shape[0] == pho_first_pos.shape[0], 'Not equal numbers of heads and phosphates in first leaflet'
        assert heads_second_com.shape[0] == pho_second_pos.shape[0], 'Not equal numbers of heads and phosphates in second leaflet'

        #Iterate over lipids in first leaflet
        for j, uniq_resname in enumerate(uniq_resnames_first):

            #Get all selections from one type of lipid
            resn_pos = heads_first & u.select_atoms("resname " + uniq_resname)

            assert len(np.unique(resn_pos.resnames)) == 1, "Bullshit!"

            coms, scd2, scd3 = calc_order(uni = u, head_selection=resn_pos, head_coms = heads_first_com,
                                          r1 = args.r_co,
                                          acyls = acyls_first[uniq_resname], Lx = Lx, Ly = Ly, Lz = Lz)

            coms_2d = np.hstack([coms[:, 0:2], np.zeros((len(coms[:, 0]), 1))])

            dist2ca = distance(x0 = coms_2d, x1 = xy_cog_start, dimensions = ts.dimensions[0:3])

            scd2 = np.hstack( [ dist2ca.reshape((-1, 1)), scd2 ] )
            scd3 = np.hstack( [ dist2ca.reshape((-1, 1)), scd3 ] )

            store_sdc_2_first[uniq_resname][step] = scd2
            store_sdc_3_first[uniq_resname][step] = scd3

        #Iterate over lipids in second leaflet
        for j, uniq_resname in enumerate(uniq_resnames_second):

            #Get all selections from one type of lipid
            resn_pos = heads_second & u.select_atoms("resname " + uniq_resname)

            assert len(np.unique(resn_pos.resnames)) == 1, "Bullshit!"

            coms, scd2, scd3 = calc_order(uni = u, head_selection=resn_pos, head_coms = heads_second_com,
                                          r1 = args.r_co,
                                          acyls = acyls_second[uniq_resname], Lx = Lx, Ly = Ly, Lz = Lz)

            coms_2d = np.hstack([coms[:, 0:2], np.zeros((len(coms[:, 0]), 1))])

            dist2ca = distance(x0 = coms_2d, x1 = xy_cog_end, dimensions = ts.dimensions[0:3])

            scd2 = np.hstack( [ dist2ca.reshape((-1, 1)), scd2 ] )
            scd3 = np.hstack( [ dist2ca.reshape((-1, 1)), scd3 ] )

            store_sdc_2_second[uniq_resname][step] = scd2
            store_sdc_3_second[uniq_resname][step] = scd3

for lipid1 in uniq_resnames_first: np.save(arr = store_sdc_2_first[lipid1], file = args.out + '_' + lipid1 + '_first_sdc2.npy')
for lipid1 in uniq_resnames_first: np.save(arr = store_sdc_3_first[lipid1], file = args.out + '_' + lipid1 + '_first_sdc3.npy')

for lipid2 in uniq_resnames_second: np.save(arr = store_sdc_2_second[lipid2], file = args.out + '_' + lipid2 + '_second_sdc2.npy')
for lipid2 in uniq_resnames_second: np.save(arr = store_sdc_3_second[lipid2], file = args.out + '_' + lipid2 + '_second_sdc3.npy')









import numpy as np

def clustering_step(frames, get_leaflet_step_order, assign_core_lipids, hierarchical_clustering, start, stop, results, leaflet):

    """
    Runs full clustering step (core_lipids + hierarchical clustering) for ONE frame.

    Parameters
    ----------
    leaflet : int
        leaflet number (0/1)
    frames : numpy.ndarray
        array of frames
    """

    if leaflet == 0: leaflet_ = 'upper'
    if leaflet == 1: leaflet_ = 'lower'

    store = []
    frame_store =[]

    #Iterate over frames
    for frame in frames:

        #Get order states
        order_states_leaf = get_leaflet_step_order(leaflet, frame)
        
        core_lipids = assign_core_lipids(weight_matrix_f=results[f"{leaflet_}_weight_all"][frame],
                                              g_star_i_f=results['Getis_Ord'][leaflet][f'g_star_i_{leaflet}'][frame],
                                              order_states_f=order_states_leaf,
                                              w_ii_f=results["Getis_Ord"][leaflet][f"w_ii_{leaflet}"][frame],
                                              z_score=results["z_score"][leaflet])

        clusters = hierarchical_clustering(weight_matrix_f=results[f"{leaflet_}_weight_all"][frame],
                                                w_ii_f=results["Getis_Ord"][leaflet][f"w_ii_{leaflet}"][frame],
                                                core_lipids=core_lipids)
        frame_number = start + frame * step
        store.append( list(clusters.values()) )
        frame_store.append( frame_number)

    store =store
    frame_store = frame_store

    print(f'Finished clustering on frames {frames[0]}-{frames[-1]}')

    return (frame_store, store)

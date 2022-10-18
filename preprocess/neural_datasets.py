import scipy.io # needed for older versions of MATLAB files
import h5py # MATLAB files > v7.3
import numpy as np
import pickle

import glob
import sys
sys.path.append("..") # access to library

from neuroprob.utils.signal import ConsecutiveArrays, TrueIslands, WrapPi, linear_interpolate




def toroidal_grid_cells(datadir, dataset, start, end, savedir):
    """
    rat_r_day1_sessions:
    #1,  open_field_1,       start=7457, end=16045, valid_times=[7457,14778;14890,16045]
    #2,  foraging_maze_1,    start=16925, end=20704, valid_times=[16925,18026;18183,20704]
    #3,  foraging_maze_2,    start=20895, end=21640
    #4,  sleep_box_1,        start=21799, end=23771
    
    rat_r_day2_sessions:
    #1,  sleep_box_1,        start=396,  end=9941
    #2,  open_field_1,       start=10617, end=13004
    #3,  sleep_box_2,        start=13143, end=15973
    
    rat_q_sessions:
    #1,  sleep_box_1,        start=0,    end=5794
    #2,  linear_track_1,     start=6685, end=9481
    #3,  sleep_box_2,        start=9576, end=18812
    #4,  linear_track_2,     start=18977, end=25355
    #5,  sleep_box_3,        start=25403, end=27007
    #6,  open_field_1,       start=27826, end=31223
    """
    moserdata = np.load(datadir+dataset+'.npz', allow_pickle=True)

    start_time = moserdata['t'][start]
    time = moserdata['t'][start:end] - start_time # relative time
    x = moserdata['x'][start:end]
    y = moserdata['y'][start:end]
    z = moserdata['z'][start:end]
    hd = np.unwrap(moserdata['azimuth'][start:end])
    
    modules = []
    for k in moserdata.keys():
        if k[-4:-1] == 'mod':
            modules.append(int(k[-1]))
    
    ### resample at 1 ms ###
    tbin = 0.001
    time_fine = np.arange(int(np.floor(time[-1]/tbin)))*tbin
    x = linear_interpolate(time, x, time_fine)
    y = linear_interpolate(time, y, time_fine)
    z = linear_interpolate(time, z, time_fine)
    hd = linear_interpolate(time, hd, time_fine)

    spike_ind = []
    spike_module = []
    for m in modules:
        spikes = moserdata['spikes_mod{}'.format(m)].item()
        clusters = len(spikes)
        for c in range(clusters):
            spktimes = spikes[c]-start_time
            spike_ind.append(np.round(spktimes / tbin).astype(int))
            spike_module.append(m)
    
    savef = savedir + dataset + '.p'
    pickle.dump((tbin, time_periods, x, y, z, hd, spike_ind, spike_module), open(savef, 'wb'), pickle.HIGHEST_PROTOCOL)




def hc23(datadir, dataset, animal_id, session, day, task, bout, savedir):
    """
    GENERAL INFORMATION: 
    Data are rat's positional data and recordings obtained from superficial layers of the medial entorhinal cortex (layer II/III). 
    Data are for 6 rats, from 1-7 sessions each, in which each session contains two "tasks" across two days. 

    The Open Field task required rats to randomly forage around a 100 x 100 cm arena for small cereal rewards. 
    The Overnight task was simply a rat placed in a 60 x 60 cm box for overnight recordings, with food, water and 
    enrichment items provided. 

    The Open Field session was repeated the following morning (Day 2), and the units' firing analyzed, to confirm 
    stability of recordings. Data from day 1 only was analyzed for the publication. Data from day 2 was used only 
    to confirm stability of overnight recordings.

    Each Open Field session included three 20-minute bouts of task performance, with 10 minutes
    of rest before and after each bout. During 'rest,' rats were removed from the track/box and placed in an
    elevated and towel lined flower pot. Data from these interwoven 10 minutes rest sessions are not provided here.

    For additional details, please see the publication: 
    Trettel, S.G., Trimper, J.B., Hwaun, E., Fiete, I.R., & Colgin, L.L. (2019). Grid cell co-activity patterns 
    during sleep reflect spatial overlap of grid fields during active behaviors. Nature Neuroscience, 22(4), 609-617.
    https://www.ncbi.nlm.nih.gov/pubmed/30911183



    ------------------------------------------------------
    RAW DATA DIRECTORY ORGANIZATION: 
    Within the main folder, each of the 6 rats has a subdirectory named according to the rat's ID. 

    Within each rat's subdirectory, a subdirectory exists for each experimental session. 

    Within each session's subdirectory, a subdirectory exists for each of two recording days. 
    Also within this folder is a text file indicating the names of the unit cluster files, which 
    are located within each bout subdirectory (to be described later) and contain the spike times 
    for each unit. Names indicate tetrode and unit number (e.g., TT8_1.t = tetrode 8, unit 1). Non-grid cells 
    are additionally tagged with '_NonGC.t' whereas confirmed grid cells do not include this tag. 

    Within each Day 1 subdirectory, a subdirectory for each 'task' can be found. A text file is also
    included noting the date at which that session was recorded. 

    Within each 'task' subdirectory (i.e., OpenField, Overnight), a subdirectory exists for 
    each bout of 'task performance.' I am employing '' here because the Overnight 'task' did not require any
    particular behaviors of the rat. For OpenField data, these bouts are called 'Begin' and
    for Overnight, these bout subdirectories are titled 'Sleep.' 

    Data contained in Sleep subdirectories corresponds to times when two independent reviewers both agreed, 
    based on viewing videos of the animals' behavior, that the rat was asleep rather than simply immobile. 
    Thus, data is not provided for the entirety of the overnight recording session, but only when rats were asleep. 

    Each bout directory contains the *.t files (unit spike times) and the *.nvt file (raw position data for 
    each LED on the rat's headstage). Neuralynx functions can be used to read in this raw data. Functions 
    available on the Colgin Lab GitHub page (e.g., Readtfile and read_in_coords) demonstrate how to employ
    these Neuralynx functions. 

    LFPs are not provided here as they are still being analyzed for a paper in preparation.

    GitHub page: https://github.com/jtrimper/ColginLabCode



    ------------------------------------------------------
    MATLAB DATA STRUCTURE ORGANIZATION: 

    1) File name: 'Rat_Data_Struct_040819.mat'
    2) Description: This data structure contains rat's positional data for open field bouts and spike-time information for each unit during open field and sleep bouts.
    3) Highest level of organization: rat
    4) Subfield structure: 
    -rat
    --name: string specifying rat's ID (e.g., Rat-20)
    --session
    ----day
    ------task: 2 'tasks' (1 = Open Field; 2 = Overnight)
    --------name
    --------bout
    ----------coords: nx3 matrix, where (:,1) = time for each video frame, (:,2) = x position for each frame, and (:,3) = y position for each frame; [available only for OpenField session]
    ----------runTimes OR REMTimes OR nRemTimes: nx2 matrix indicating the start time (:,1) and stop time (:,2), in seconds, in which the rat was in each behavioral state. See publication for functional definition of each behavioral state. Available for day 1 only.
    ----------unit: data for each unit
    ------------ID: tetrode and cluster number
    ------------type: scalar indicating whether unit is a grid cell (1) or a non-grid cell (2)
    ------------spkTms: nx1 vector indicating time for each action potential, in seconds
    
    References:
    
    [1] Trimper, John; Trettel, Sean G.; Hwaun, Ernie; Fiete, Ila R.; Colgin, Laura Lee (2019): Grid Cell and Non-Grid Cell Recordings from Layer II/III of Medial Entorhinal Cortex from 6 rats (1-7 sessions each) During Active Exploration and Overnight Sleep. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.4496375.v1 
    """    
    mat = scipy.io.loadmat(datadir+'hc-23/{}.mat'.format(dataset))
    
    print('Rat ID: {}'.format(mat['rat'][0, animal_id][0][0]))
    print('Task: {}'.format(mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][0][0]))
    
    savef = savedir + '{}_{}_s{}_d{}_b{}.p'.format(mat['rat'][0, animal_id][0][0], 
                                                   mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][0][0], 
                                                   session+1, day+1, bout+1)
    
    start_time = mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][0][0, 0]
    end_time = mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][0][-1, 0]
    time = mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][0][:, 0] - start_time
    time_periods = mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][2]
    
    x = mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][0][:, 1]
    y = mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][0][:, 2]
    dx = x[1:]-x[:-1]
    dx = np.concatenate((dx, dx[-1:]))
    dy = y[1:]-y[:-1]
    dy = np.concatenate((dy, dy[-1:]))
    dtime = time[1:]-time[:-1]
    dtime = np.concatenate((dtime, dtime[-1:]))
    
    s = np.sqrt(dx**2 + dy**2)/dtime
    dir_t = np.unwrap(np.angle(np.exp(dx + dy*1j)))
    
    
    ### resample at 1 ms ###
    tbin = 0.001
    time_fine = np.arange(int(np.floor(time[-1]/tbin)))*tbin
    x = linear_interpolate(time, x, time_fine)
    y = linear_interpolate(time, y, time_fine)
    s = linear_interpolate(time, s, time_fine)
    dir_t = linear_interpolate(time, dir_t, time_fine)

    spike_ind = []
    cell_type = []
    units = len(mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][1][0, :])
    for unit in range(units):
        spktimes = mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][1][0, unit][2][:, 0] - start_time
        spike_ind.append(np.round(spktimes / tbin).astype(int))
        cell_type.append(mat['rat'][0, animal_id][1][0, session][0][0, day][0][0, task][1][0, bout][1][0, unit][1][0, 0])
    
    pickle.dump((tbin, time_periods, x, y, s, dir_t, spike_ind, cell_type), open(savef, 'wb'), pickle.HIGHEST_PROTOCOL)
    
    



def sargolini_EC(datadir, session_id, savefile):
    """
    The sample includes conjunctive cells and head direction cells from layers III and V of medial entorhinal cortex and have  been published in Sargolini et al. (Science, 2006).

    The files are in matlab format. They include spike and position times for recorded cells from rats that were running in a 1 x 1 m
    enclosure. The cells were recorded in the dorsocaudal 25% portion of the medial entorhinal cortex. Position is given for two LEDs
    to enable calculation of head direction.

    The cell id is based on tetrode number and cell number (i.e: t2c7).

    The file naming convention is as follow:

    Rat number - session number _ cell id (i.e: 11084-03020501_t2c1).

    Each session duration is normally 10 minutes, but some sessions are combination of 2 or more 10 minutes sessions, this is marked in
    the file name by using "+" between the session numbers (i.e: 11207-21060501+02_t6c1). Note that the 6 first digits in the session
    number is the date of the recording.

    When loading the files into Matlab you get the following variables:

    x1      Array with the x-positions for the first tracking LED.
    y1      Array with the y-positions for the first tracking LED.
    x2      Array with the x-positions for the second tracking LED.
    y2      Array with the y-positions for the second tracking LED.
    t       Array with the position timestamps.
    ts      Array with the cell spike timestamps.

    The position data have been smoothed with a moving mean filter to remove tracking jitter.


    You can use the data for whatever you want but we take no responsibility for what is published!

    Please refer to our web site where you obtained the data in your Methods section when you write up the results.

    Best regards,

    Raymond Skjerpeng and Edvard Moser
    Kavli Institute for Systems Neuroscience
    Centre for the Biology of Memory
    Norwegian University of Science and Technology

    Correspondence: Edvard Moser (edvard.moser@ntnu.no)


    Published : DOI: 10.1126/science.1125572, Citation: Sargolini, F. et al, 2006, "Conjunctive Representation of Position, Direction, and Velocity in Entorhinal Cortex", Science 312 (5774), 758-612 (doi:10.1126/science.1125572 (primary)
    """
    spiketimes = []
    
    names = glob.glob(datadir+'EC_Sargolini/{}_*.mat'.format(session_id))
    print('Files: {}'.format(len(names)))
    print(names)
    
    for name in names:
        mat = scipy.io.loadmat(name)
        spiketimes.append(mat['ts'][:, 0])
      
    timesteps = mat['t'].shape[0]
    time = mat['t'][:, 0]
    x_1 = np.empty((timesteps, 2))
    x_2 = np.empty((timesteps, 2))
    x_1[:, 0] = mat['x1'][:, 0]
    x_1[:, 1] = mat['y1'][:, 0]
    x_2[:, 0] = mat['x2'][:, 0]
    x_2[:, 1] = mat['y2'][:, 0]
    
    pos = (x_2 + x_1)/2.
    
    # remove invalid indices
    inds = list(range(timesteps))
    for i in np.where(pos[:, 0] != pos[:, 0])[0]:
        inds.remove(i)
    pos = pos[inds, :]
    time = time[inds]
    
    dpos = pos[1:]-pos[:-1]
    dpos = np.concatenate((dpos, dpos[-1:, :]), axis=0)
    #speed = (pos[2:]+pos[:-2]-2*pos[1:-1])/2. # midpoint approximation
    dtime = time[1:]-time[:-1]
    dtime = np.concatenate((dtime, dtime[-1:]))
    
    speed = np.sqrt(((dpos/dtime[:, None])**2).sum(-1))
    hd = np.unwrap(np.angle(np.exp(dpos[:, 0] + 1j*dpos[:, 1])))
    
    
    ### resample to 1 ms ###
    tbin = 0.001
    spike_ind = []
    for sp in spiketimes:
        spike_ind.append(np.round(sp / tbin).astype(int))
    
    time_fine = np.arange(int(np.floor(time[-1]/tbin)))*tbin
    x = linear_interpolate(time, pos[:, 0], time_fine)
    y = linear_interpolate(time, pos[:, 1], time_fine)
    s = linear_interpolate(time, speed, time_fine)
    hd = linear_interpolate(time, hd, time_fine)
    
    pickle.dump((tbin, x, y, speed, hd, spike_ind), open(savefile, 'wb'), pickle.HIGHEST_PROTOCOL)





def hc5_open_field(datadir, dataset, savefile):
    """
    [hc5] open field exploration with random foraging task
    
    Pyramidal cells were distinguished from theta cells in the CAI region by a number of criteria  
    (Ranck,  1973; Fox and Ranck, 1981; Buzsiki  et al.,  1983; McNaughton et al.,  1983a). To be 
    classified as a pyramidal cell, a unit was required to 1 ) be recorded simultaneously with other 
    pyramidal cells; 2) fire at least a small number of complex spike bursts during the recording 
    session; 3) have a spike width (peak to valley) of at least 300 psec;  and 4) have an overall 
    mean rate below 5 Hz during the recording ses- sion. To be classified as  a theta cell, a unit 
    was required to 1) fire no complex spike bursts; 2) have a spike width less than 300 psec; and 
    3) fire with a mean rate above 5 Hz during the recording session.
    
    Each spike, if theta existed when it occurred, was assigned a nominal phase, according to the 
    fraction of the time between the preceding  and  following  theta  peaks at which  it  occurred. 
    Precisely, the phase assigned to an event at time t was 360 . (t - to)/(tl - to), where to and 
    tl are the times of the preceding  and following peaks of the filtered  reference EEG signal. 
    Note that the phase was always a number between 0 and 360. 
    
    that hippocampal pyramidal cell population  activity is  modulated  in a consistent way by the 
    theta  rhythm,  with  the  peak  occurring  simultaneously throughout  the dorsal hippocampus 
    (CA1 and probably CA3 as
    Global phase zero was defined to be the point in the theta cycle corresponding to maximal pyramidal 
    cell activity. Thus, we measured the theta rhythm using EEG, and used population pyramidal cell 
    activity, integrated across an entire data set, to decide which point on the EEG wave to call phase 
    zero. [1]
    
    
    theta_hilb = mat['Spike'][0,0][4][:, 0]
    x_spike = mat['Spike'][0,0][5][:, 0]
    y_spike = mat['Spike'][0,0][6][:, 0]
    hd_spike = mat['Spike'][0,0][10][:, 0]
    s_spike = mat['Spike'][0,0][11][:, 0]
    
    x_s = np.delete(x_spike, idz)
    y_s = np.delete(y_spike, idz)
    s_s = np.delete(s_spike, idz)
    hd_s = np.delete(hd_spike, idz)
    
    checked spike-track correspondence here
    plt.plot(t_spike[:10], np.delete(mat['Spike'][0,0][4][:, 0], idz)[:10]) # 4, 12, 10
    ff = int(t_spike[9]/sample_bin)
    plt.plot(np.arange(ff)*sample_bin, theta_t[:ff])
    
    
    
    References:
    
    [1] Pastalkova E, Wang Y, Mizuseki K, Buzsáki G. (2015)
        `Simultaneous extracellular recordings from left and right hippocampal areas CA1 and right entorhinal cortex from a rat performing a left / right alternation task and other behaviors', 
        CRCNS.org.
        http://dx.doi.org/10.6080/K0KS6PHF
    
    [2] `Theta Phase Precession in Hippocampal Neuronal Populations and the Compression of Temporal Sequences`,
    Skaggs et al. 1996

    Right EC: channels 1-32
    • Right hippocampus (CA1): channels 33-64
    • Left hippocampus (CA1): channels 65-128 (except for channels 105-107)
    • Wheel TTL pulse: channels 105+106
    • SYNC pulse: channel 107

    """
    mat = scipy.io.loadmat(datadir+dataset)

    x_track = mat['Track'][0,0][0][:, 0] # in mm
    y_track = mat['Track'][0,0][1][:, 0]
    hd_track = mat['Track'][0,0][6][:, 0]
    #s_track = mat['Track'][0,0][7][:, 0] # in mm/s too inaccurate
    theta_track = mat['Track'][0,0][-2][:, 0]
    eeg_track = mat['Track'][0,0][-1][:, 0] # differs in index for datasets
    x_spike = mat['Spike'][0,0][5][:, 0]
    y_spike = mat['Spike'][0,0][6][:, 0]
    
    
    sample_bin = 1.0/1250 # in s

    # remove outside session regions in track data
    idz = np.where((x_track == 0.0) & (y_track == 0.0))[0]
    idz_consec = ConsecutiveArrays(idz)
    if len(idz_consec) > 0:
        idz_islands = int(idz_consec.max())
        idz_size = np.empty(idz_islands)
        for i in range(idz_islands):
            idz_size[i] = (idz_consec == i+1).sum()
            if idz_size[i] < 10: # this may be part of trajectory
                idz = np.delete(idz, np.where(idz_consec == i+1))
    else:
        idz = []

    start = idz_size[0] if idz_consec[0] == 1 and idz[0] == 0 else 0
    end = len(x_track) - idz_size[1] if idz_consec[-1] != 0 and idz[-1] == len(x_track)-1 else 0
    x_t = np.delete(x_track, idz)
    y_t = np.delete(y_track, idz)
    #s_t = np.delete(s_track, idz)
    hd_t = np.delete(hd_track, idz)
    eeg_t = np.delete(eeg_track, idz)
    theta_t = np.delete(theta_track, idz)
    track_samples = len(x_t)

    # the spike data seems to contain periods of no movement (no tracking outside session?)
    idz = np.where((x_spike == 0.0) & (y_spike == 0.0))[0]
    idz_consec = ConsecutiveArrays(idz)
    if len(idz_consec) > 0:
        idz_islands = int(idz_consec.max())
        idz_size = np.empty(idz_islands)
        for i in range(idz_islands):
            idz_size[i] = (idz_consec == i+1).sum()
            if idz_size[i] < 10: # this may be part of trajectory
                idz = np.delete(idz, np.where(idz_consec == i+1))
    else:
        idz = []
        
    
    t_spike = np.delete(mat['Spike'][0,0][0][:, 0], idz) # in track samples
    t_spike -= t_spike.min() # set start to 0 ms
    clu_id = np.delete(mat['Spike'][0,0][3][:, 0] - 1, idz)
    spike_samples = len(t_spike)
    units = clu_id.max()+1

    sep_t_spike_ = []
    for u in range(units):
        sep_t_spike_.append(t_spike[clu_id == u])
        
        
    ### resample at 1 ms ###
    tbin = 0.001
    sep_t_spike = []
    for sp in sep_t_spike_:
        sep_t_spike.append(np.round(sp*sample_bin / tbin).astype(int))
    
    time = np.arange(track_samples)*sample_bin
    time_fine = np.arange(int(np.ceil(time[-1]/tbin)))*tbin
    x_t = linear_interpolate(time, x_t, time_fine)
    y_t = linear_interpolate(time, y_t, time_fine)
    eeg_t = linear_interpolate(time, eeg_t, time_fine)
    theta_t = linear_interpolate(time, np.unwrap(theta_t), time_fine)
    hd_t = linear_interpolate(time, np.unwrap(hd_t), time_fine)
    
    sample_bin = tbin # reset
    track_samples = len(time_fine)
        
        
    # empirical velocities
    vx_t = (x_t[1:] - x_t[:-1]) / sample_bin # mm/s
    vy_t = (y_t[1:] - y_t[:-1]) / sample_bin # mm/s
    emp_s_t = np.sqrt(vx_t**2 + vy_t**2)

    # pauses in the animal 
    pause_ind, pause_size = TrueIslands((vx_t == 0) & (vy_t ==0))

    # check trajectory continuity with empirical acceleration
    emp_a_t = (emp_s_t[1:] - emp_s_t[:-1]) / sample_bin # mm/s^2

    # note that s_t has -1.0s at the start and end, and it is unreliable compared to emp_s_t
    inds = np.where(np.abs(emp_a_t) > 1e3)[0] # potential tracking errors, realistic acceleration bound
    for i in inds:
        ind = np.where(pause_ind == i+1)[0] # error if the tracker doesn't move
        if ind.size > 0:
            ind = ind[0]
            ex = x_t[pause_ind[ind]]
            ey = y_t[pause_ind[ind]]
            ix = x_t[pause_ind[ind]-1]
            iy = y_t[pause_ind[ind]-1]
            dx = (ex-ix)/(pause_size[ind]) # linear interpolation
            dy = (ey-iy)/(pause_size[ind])
            for g in range(1,pause_size[ind]):
                x_t[pause_ind[ind]-1 + g] = ix + g*dx
                y_t[pause_ind[ind]-1 + g] = iy + g*dy

            #plt.plot(x_t[pause_ind[ind]-10:pause_ind[ind]], 
            #         y_t[pause_ind[ind]-10:pause_ind[ind]], marker='o')
            #plt.scatter(ex, ey, marker='+', s=1000, color='r')
            #plt.scatter(ix, iy, marker='+', s=1000, color='r')
            #plt.show()

    # reliable velocity estimate with O(delta^2), max rat speed ~3.6 m/s in nature
    emp_s_t = np.empty(track_samples)
    dir_t = np.empty(track_samples)

    vx_t = np.empty(track_samples)
    vy_t = np.empty(track_samples)
    vx_t[1:-1] = (x_t[2:] + x_t[:-2] - 2*x_t[1:-1]) / sample_bin / 2 # mm/s
    vy_t[1:-1] = (y_t[2:] + y_t[:-2] - 2*y_t[1:-1]) / sample_bin / 2 # mm/s
    vx_t[0] = vx_t[1] # copy ends
    vy_t[0] = vy_t[1] # copy ends
    vx_t[track_samples-1] = vx_t[track_samples-2]
    vy_t[track_samples-1] = vy_t[track_samples-2]

    emp_s_t = np.sqrt(vx_t**2 + vy_t**2)
    dir_t = np.angle(vx_t + vy_t*1j)

    #emp_s_s = emp_s_t[t_spike]
    #dir_s = dir_t[t_spike]
    pause_ind, pause_size = TrueIslands(emp_s_t == 0)

    # Deduce cell (spiking unit SU) quality
    shank_id = mat['Clu'][0,0][0][0, :]
    local_clu = mat['Clu'][0,0][1][0, :]
    # mat['Clu'][0,0][2][0, :] is cluster id, range(1,units+1)
    FR_waveshape = mat['Clu'][0,0][3][0, :]# based on FR and wave shape
    SpkWidthC = mat['Clu'][0,0][4][0, :]
    refract_viol = mat['Clu'][0,0][5][0, :] # ISI percentage < 2 ms
    #sess_avg_rate = mat['Clu'][0,0][6][0, :] # fire rate in Hz

    """ Mahalanobis distance of noise (Isolation Distance)
    Isolation Distance was first introduced by Harris et al. (2001)
    and applied to hippocampal data sets. If a cluster contains nC
    cluster spikes, the Isolation Distance of the cluster is the D^2
    value of the n_C closest noise spike. Isolation Distance is 
    therefore the radius of the smallest ellipsoid from
    the cluster center containing all of the cluster spikes and an
    equal number of noise spikes. As such, Isolation Distance
    estimates how distant the cluster spikes are from the other
    spikes recorded on the same electrode. Isolation Distance is
    not defined for cases in which the number of cluster spikes is
    greater than the number of noise spikes."""
    isolation_dist = mat['Clu'][0,0][8][0, :]

    """local coefficient of variation LV classification:
    regular spiking for LV ∈ [0, 0.5], irregular
    for LV ∈ [0.5, 1], and bursty spiking for LV > 1.
    S. Shinomoto, K. Shima, and J. Tanji. Differences in spiking patterns
    among cortical neurons. Neural Computation, 15(12):2823–2842, 2003."""
    ISI = []
    for u in range(units):
        ISI.append((sep_t_spike[u][1:] - sep_t_spike[u][:-1])*sample_bin*1000) # ms

    dir_t = WrapPi(dir_t, True)
    hd_t = WrapPi(hd_t, True)
    theta_t = WrapPi(theta_t, True)
    
    offset = theta_t[0]
    pthet = 0
    for t in range(track_samples):
        if theta_t[t] - offset < pthet: # we went through a cycle
            offset -= 2*np.pi
        theta_t[t] = theta_t[t] - offset
        pthet = theta_t[t]
    
    # Hilbert transform
    #analytic_signal = hilbert(eeg_t)
    #hilbert_amp = np.abs(analytic_signal)
    #hilbert_theta = np.unwrap(np.angle(analytic_signal))
    
    left_x = 0.0 # mm
    right_x = 540.0
    bottom_y = 0.0
    top_y = 435.0 # check this
        
    pickle.dump((sample_bin, track_samples, x_t, y_t, emp_s_t, dir_t, \
        hd_t, eeg_t, theta_t, pause_ind, pause_size, \
        sep_t_spike, clu_id, t_spike, spike_samples, units, \
        shank_id, local_clu, FR_waveshape, SpkWidthC, \
        refract_viol, isolation_dist, \
        left_x, right_x, bottom_y, top_y), open(savefile, 'wb'), pickle.HIGHEST_PROTOCOL)






def hc_2_3(datadir, session_id, savefile, shanks, linear_interpolate=True, speed_limit=1e3):
    """
    [hc2] as in [1] and [hc3] as in [2] loading, both use a similar format.
    
    :param string datadir: the directory where the data files are located
    :param string session_id: the name of the session ID which file names are based on
    :param string savefile: the output file location
    :param int shanks: the number of shanks used to record (shank id is at the end of file name)
    :param bool linear_interpolate: interpolate the behaviour linearly as we upsample, otherwise 
                                    we have stepwise constant behaviour after upsampling
    :param float speed_limit: the upper limit of animal speed before it becomes a tracking error
    
    References:
    
    [1] Mizuseki K, Sirota A, Pastalkova E, Buzsáki G. (2009): 
        `Multi-unit recordings from the rat hippocampus made during open field foraging.'
        http://dx.doi.org/10.6080/K0Z60KZ9
        
    [2] Mizuseki, K., Sirota, A., Pastalkova, E., Diba, K., Buzsáki, G. (2013)
        `Multiple single unit recordings from different rat hippocampal and entorhinal regions while the animals were performing multiple behavioral tasks.', 
        CRCNS.org.
        http://dx.doi.org/10.6080/K09G5JRZ
    """
    spiketimes = []
    spikeclusters = []
    totclusters = []
    for k in range(1, shanks+1):
        f_res = open(datadir+session_id+".res.{}".format(k), "r").read()
        f_clu = open(datadir+session_id+".clu.{}".format(k), "r").read()
        
        clu = np.array(f_clu.split(sep='\n')[:-1]).astype(np.float)
        totclusters.append(clu[0]-2) # index 0 represents artifacts, 1 – noise/nonclusterable units, 2 and above – isolated units
        clu = clu[1:]
        C0 = np.where(clu == 0)[0]
        C1 = np.where(clu == 1)[0]
        indx = list(C0) + list(C1)
        
        spikeclusters.append(np.delete(clu, indx)-2)
        res = np.array(f_res.split(sep='\n')[:-1]).astype(np.float)
        spiketimes.append(np.delete(res, indx))
        
    totclusters = np.array(totclusters).astype(int)
    units = totclusters.sum()

    sep_t_spike = []
    track_samples = 0
    u = 0
    for k in range(shanks):
        for l in range(totclusters[k]):
            arr = np.sort(spiketimes[k][spikeclusters[k] == l]).astype(int)
            if (len(arr) < 2): # silent neurons
                units -= 1
                print('Less than 2 spikes on channel shank {} cluster {}, excluded from data.'.format(k, l))
                continue
                
            u += 1
            sep_t_spike.append(arr)
            if track_samples < arr[-1]:
                track_samples = arr[-1]
        k += 1
        
    sample_bin = 1./20000
    
    f_pos = open(datadir+session_id+".whl", "r").read()
    l = f_pos.split(sep='\n')[:-1]
    pos = np.array([i.split('\t') for i in l]).astype(np.float)
    pos_1 = pos[:, :2]
    pos_2 = pos[:, 2:]
    
    dp = pos_1 - pos_2
    hd_beh = np.angle(dp[:, 0]+dp[:, 1]*1j) % (2*np.pi)
    x_beh = (pos_1[:, 0]+pos_2[:, 0])/2.
    y_beh = (pos_1[:, 1]+pos_2[:, 1])/2.
    
    inval_ind, inval_size = TrueIslands((pos_1[:, 0] == -1.0) | (pos_1[:, 1] == -1.0) | 
                                        (pos_2[:, 0] == -1.0) | (pos_2[:, 1] == -1.0))
    
    # interpolator for invalid data
    for i, ind in enumerate(inval_ind):
        if ind == 0 or ind+inval_size[i] == len(hd_beh): # if at start or end leave -1 in
            x_beh[ind:ind+inval_size[i]] = -1
            y_beh[ind:ind+inval_size[i]] = -1
            hd_beh[ind:ind+inval_size[i]] = -1
            continue
            
        # interpolate with geodesic distances
        dhd = hd_beh[ind+inval_size[i]]-hd_beh[ind-1]
        if dhd > np.pi:
            dhd -= 2*np.pi
        elif dhd < -np.pi:
            dhd += 2*np.pi
        for ii in range(inval_size[i]):
            hd_beh[ii+ind] = (hd_beh[ind-1] + dhd*(ii+1)/(inval_size[i]+1)) % (2*np.pi)

        dx = x_beh[ind+inval_size[i]]-x_beh[ind-1]
        dy = y_beh[ind+inval_size[i]]-y_beh[ind-1]
        for ii in range(inval_size[i]):
            x_beh[ii+ind] = x_beh[ind] + dx*(ii+1)/(inval_size[i]+1)
            y_beh[ii+ind] = y_beh[ind] + dy*(ii+1)/(inval_size[i]+1)
            
            
    # empirical velocities used to smoothen sudden jumps
    behav_tbin = 1./39.06
    vx_beh = (x_beh[1:] - x_beh[:-1]) / behav_tbin # mm/s
    vy_beh = (y_beh[1:] - y_beh[:-1]) / behav_tbin # mm/s
    s_beh = np.sqrt(vx_beh**2 + vy_beh**2)
    s_beh[(x_beh[1:] == -1) | (x_beh[:-1] == -1)] = -1 # invalid
    
    inval_ind_, inval_size_ = TrueIslands(s_beh == -1) # at edges, ignore these time points
    assert len(inval_size_) == 2 # locations at which this or next time step is invalid

    inds = np.where(s_beh[inval_size[0]:inval_ind[-1]] > speed_limit)[0]+inval_size_[0] # potential tracking errors
    assert len(inds) % 2 == 0 # pairs indicating segments
    for k in range(len(inds)//2):
        x_prev = inds[2*k]
        x_next = inds[2*k+1]+1
        steps = (x_next-x_prev)
        t = np.arange(1, steps)

        dx = (x_beh[x_next]-x_beh[x_prev])/steps
        x_beh[inds[2*k]+1:inds[2*k+1]+1] = x_beh[x_prev]+dx*t
        
        dy = (y_beh[x_next]-y_beh[x_prev])/steps
        y_beh[inds[2*k]+1:inds[2*k+1]+1] = y_beh[x_prev]+dy*t
    
    # resample behaviour from 39.06 Hz to 1250 Hz
    beh_sample_bin = 0.0008
    beh_samples = int(np.rint(track_samples/16)) # 20 kHz to 1250 Hz
    x_t = np.empty((beh_samples))
    y_t = np.empty((beh_samples))
    hd_t = np.empty((beh_samples))
    
    if linear_interpolate:
        dx = np.concatenate((x_beh[1:]-x_beh[:-1], [0]))
        dy = np.concatenate((y_beh[1:]-y_beh[:-1], [0]))
        dhd = np.concatenate((hd_beh[1:]-hd_beh[:-1], [0]))
        dhd[dhd > np.pi] -= 2*np.pi # geodesic
        dhd[dhd < -np.pi] += 2*np.pi
        
        ind = np.floor(np.arange(beh_samples)*beh_sample_bin/behav_tbin).astype(int)
        ind_change = np.where((ind[1:]-ind[:-1]) == 1)[0]
        ind_change = np.concatenate(([0], ind_change))
        
        step_sizes = ind_change[1:]-ind_change[:-1]+1
        step_sizes = np.concatenate((step_sizes, [beh_samples-(step_sizes-1).sum()+1])) # sums to beh_samples
        
        _beh = np.stack((x_beh, y_beh, hd_beh))
        _d = np.stack((dx, dy, dhd))
        _d[:, :inval_size_[0]] = 0 # set differences in invalid regions to zero
        _d[:, inval_ind_[1]:] = 0
        
        step_counter = np.concatenate([np.arange(s-1) for s in step_sizes])
        
        _p = _beh[:, ind] + _d[:, ind]/step_sizes[None, ind]*step_counter[None, :]
        x_t = _p[0]
        y_t = _p[1]
        hd_t = _p[2]
        
        # reset to -1 for invalid indices
        ind = np.round(np.arange(beh_samples)*beh_sample_bin/behav_tbin).astype(int)
        ind[-1] -= 1 # last point will be the previous one
        
        inv_ind = np.where(x_beh == -1)[0]
        for ii in inv_ind:
            x_t[(ind == ii)] = -1
            y_t[(ind == ii)] = -1
        
    else: # copy between the points the previous sampled behaviour
        ind = np.round(np.arange(beh_samples)*beh_sample_bin/behav_tbin).astype(int)
        ind[-1] -= 1 # last point will be the previous one
        x_t = x_beh[ind]
        y_t = y_beh[ind]
        hd_t = hd_beh[ind]
        
    ISI = []
    for u in range(units):
        ISI.append((sep_t_spike[u][1:] - sep_t_spike[u][:-1])*sample_bin*1000) # ms

    refract_viol = np.empty((units))
    viol_ISI = 2.0 # ms
    for u in range(units):
        refract_viol[u] = (ISI[u] <= viol_ISI).sum()/len(ISI[u])
        
    # eeg at 1250 Hz
    eeg_1250Hz = np.fromfile(open(datadir+session_id+".eeg", "rb"), dtype=np.int16)
    eeg_channels = int(np.rint(len(eeg_1250Hz)/beh_samples)) # 20 kHz for the rest
    eeg_t = np.reshape(eeg_1250Hz, (-1, eeg_channels)).T # channel, time
        
        
    save_data = (beh_sample_bin, beh_samples, x_t, y_t, hd_t, eeg_t, \
                units, sample_bin, track_samples, sep_t_spike, \
                refract_viol)
    pickle.dump(save_data, open(savefile, 'wb'), pickle.HIGHEST_PROTOCOL)
    



def peyrache_th1(datadir, mouse_id, session_id, savefile, channels, phase, interpolate_invalid=True, 
        linear_interpolate=True):
    """
    [th1] head direction cell detection in anterior nuclei of thalamus and postsubiculum in mice
    In “pos” and “ang” files, -1 values indicate that LED detection failed. 
    time of the first video frame was randomly misaligned by 0–60 ms 
    
    The behaviour is recorded at a different frequency, resample to get it at the same frequency as 
    spike recordings.
    
    left_x = 0. # mm
    right_x = 530.
    bottom_y = 0.
    top_y = 460.
    
    :param tuple channel_tuple: tuple indicating channels belonging to neuron type (channels, neuron_type)
    :param bool remove_invalid: indicates whether to remove invalid data segments or interpolate
    
    References:
    
    [1] Peyrache, A., Petersen P., Buzsáki, G. (2015)
        `Extracellular recordings from multi-site silicon probes in the anterior thalamus and subicular formation of freely moving mice', 
        CRCNS.org.
        http://dx.doi.org/10.6080/K0G15XS1
    
    """
    f_ang = open(datadir+"PositionFiles/"+mouse_id+"/"+session_id+"/"+session_id+".ang", "r").read()
    f_pos = open(datadir+"PositionFiles/"+mouse_id+"/"+session_id+"/"+session_id+".pos", "r").read()
    # PosFiles, AngFiles are the same data
    
    l = f_pos.split(sep='\n')[:-1]
    pos = np.array([i.split('\t') for i in l]).astype(np.float)

    l = f_ang.split(sep='\n')[:-1]
    ang = np.array([i.split('\t') for i in l]).astype(np.float)

    f_wake = open(datadir+session_id+"/"+session_id+".states.Wake", "r").read()
    l = f_wake.split(sep='\n')[:-1]
    wake = np.array([i.split('\t') for i in l]).astype(np.float)
    assert len(wake) == 1 # only one wake session

    f_REM = open(datadir+session_id+"/"+session_id+".states.REM", "r").read()
    l = f_REM.split(sep='\n')[:-1]
    REM = np.array([i.split('\t') for i in l]).astype(np.float) # (session, times (ms))


    f_SWS = open(datadir+session_id+"/"+session_id+".states.SWS", "r").read()
    l = f_SWS.split(sep='\n')[:-1]
    SWS = np.array([i.split('\t') for i in l]).astype(np.float) # (session, times (ms))


    spiketimes = []
    spikeclusters = []
    totclusters = []
    for key in channels:
        for e in channels[key]: # assume clusters are separate neurons for each file
            f_res = open(datadir+session_id+"/"+session_id+".res.{}".format(e), "r").read()
            f_clu = open(datadir+session_id+"/"+session_id+".clu.{}".format(e), "r").read()
            spiketimes.append(np.array(f_res.split(sep='\n')[:-1]).astype(np.float))
            arr = np.array(f_clu.split(sep='\n')[:-1]).astype(np.float)
            spikeclusters.append(arr[1:])
            totclusters.append(arr[0])
    
    totclusters = np.array(totclusters).astype(int)
    units = totclusters.sum()

    sep_t_spike = []
    track_samples = 0
    neuron_groups = {}
    k = 0
    u = 0
    for key, value in channels.items():
        neurons = []
        for _ in range(len(value)):
            for l in range(totclusters[k]):
                arr = np.sort(spiketimes[k][spikeclusters[k] == l]).astype(int)
                if (len(arr) == 0): # silent neurons
                    units -= 1
                    print('Empty channel shank {} cluster {}.'.format(k, l))
                    continue
                    
                neurons += [u]
                u += 1
                sep_t_spike.append(arr)
                if track_samples < arr[-1]:
                    track_samples = arr[-1]
            k += 1
        neuron_groups[key] = neurons
        
    sample_bin = 1./20000
    behav_times = ang[:, 0]
    
    if phase == 'wake':
        left_T = wake[0,0]
        right_T = wake[0,1]
    elif phase == 'prewake':
        left_T = 0
        right_T = wake[0,0]
    elif phase == 'postwake':
        left_T = wake[0,1]
        right_T = behav_times[-1]
    else:
        raise ValueError('Recording segment not known.')

    window = (behav_times < right_T) & (behav_times >= left_T)
    x_beh = pos[window, 1]
    y_beh = pos[window, 2]
    hd_beh = ang[window, 1]
    use_times = behav_times[window] - left_T
    x_beh[x_beh != x_beh] = -1.0 # NaNs
    y_beh[y_beh != y_beh] = -1.0

    # extract spikes for the region used
    use_t_spike = []
    use_samples = 0
    for u in range(units):
        times = sep_t_spike[u]*sample_bin
        use_t_spike.append(sep_t_spike[u][(times < right_T) & (times >= left_T)] - int(np.round(left_T/sample_bin)))
        if use_samples < use_t_spike[u].max()+1:
            use_samples = use_t_spike[u].max()+1
            
    # interpolator
    if interpolate_invalid:
        inval_ind, inval_size = TrueIslands(hd_beh == -1.0)
        for i, ind in enumerate(inval_ind):
            if ind == 0 or ind+inval_size[i] == len(hd_beh): # leave in -1 at ends
                continue

            # interpolate with geodesic distances
            dhd = hd_beh[ind+inval_size[i]]-hd_beh[ind-1]
            if dhd > np.pi:
                dhd -= 2*np.pi
            elif dhd < -np.pi:
                dhd += 2*np.pi
            for ii in range(inval_size[i]):
                hd_beh[ii+ind] = (hd_beh[ind-1] + dhd*(ii+1)/(inval_size[i]+1)) % (2*np.pi)

        inval_ind, inval_size = TrueIslands(x_beh == -1.0)
        for i, ind in enumerate(inval_ind):
            if ind == 0 or ind+inval_size[i] == len(hd_beh): # leave in -1 at ends
                continue

            # interpolate with geodesic distances
            dx = x_beh[ind+inval_size[i]]-x_beh[ind-1]
            dy = y_beh[ind+inval_size[i]]-y_beh[ind-1]
            for ii in range(inval_size[i]):
                x_beh[ii+ind] = x_t[ind] + dx*(ii+1)/(inval_size[i]+1)
                y_beh[ii+ind] = y_t[ind] + dy*(ii+1)/(inval_size[i]+1)

    # resample
    x_t = np.empty((use_samples))
    y_t = np.empty((use_samples))
    hd_t = np.empty((use_samples))
    
    behav_tbin = use_times[1]-use_times[0]
    print("Behaviour time bin size {} s.".format(behav_tbin))
    
    if linear_interpolate:
        dx = np.concatenate((x_beh[1:]-x_beh[:-1], [0]))
        dy = np.concatenate((y_beh[1:]-y_beh[:-1], [0]))
        dhd = np.concatenate((hd_beh[1:]-hd_beh[:-1], [0]))
        dhd[dhd > np.pi] -= 2*np.pi # geodesic
        dhd[dhd < -np.pi] += 2*np.pi
        ind = np.floor(np.arange(use_samples)*sample_bin/behav_tbin).astype(int)
        ind_change = np.where((ind[1:]-ind[:-1]) == 1)[0]
        ind_change = np.concatenate(([0], ind_change))
        step_sizes = ind_change[1:]-ind_change[:-1]+1
        step_sizes = np.concatenate((step_sizes, [use_samples-(step_sizes-1).sum()+1]))
        _beh = np.stack((x_beh, y_beh, hd_beh))
        _d = np.stack((dx, dy, dhd))
        
        step_counter = np.concatenate([np.arange(s-1) for s in step_sizes])
        
        _p = _beh[:, ind] + _d[:, ind]/step_sizes[None, ind]*step_counter[None, :]
        x_t = _p[0]
        y_t = _p[1]
        hd_t = _p[2]
        
        # reset to -1 for invalid indices
        ind = np.round(np.arange(use_samples)*sample_bin/behav_tbin).astype(int)
        ind[-1] -= 1 # last point will be the previous one
        
        inv_ind = np.where(hd_beh == -1)[0]
        for ii in inv_ind:
            hd_t[(ind == ii)] = -1
            
        inv_ind = np.where(x_beh == -1)[0]
        for ii in inv_ind:
            x_t[(ind == ii)] = -1
            y_t[(ind == ii)] = -1
        
    else:
        ind = np.round(np.arange(use_samples)*sample_bin/behav_tbin).astype(int)
        ind[-1] -= 1 # last point will be the previous one
        x_t = x_beh[ind]
        y_t = y_beh[ind]
        hd_t = hd_beh[ind]
                
    ISI = []
    for u in range(units):
        ISI.append((use_t_spike[u][1:] - use_t_spike[u][:-1])*sample_bin*1000) # ms

    refract_viol = np.empty((units))
    viol_ISI = 2.0 # ms
    for u in range(units):
        refract_viol[u] = (ISI[u] <= viol_ISI).sum()/len(ISI[u])
        
    save_data = (sample_bin, use_samples, x_t, y_t, hd_t, \
                units, use_t_spike, track_samples, sep_t_spike, \
                refract_viol, neuron_groups, \
                wake, SWS, REM)
    pickle.dump(save_data, open(savefile, 'wb'), pickle.HIGHEST_PROTOCOL)
    
    
    
def perich_pmd1(dataset, savefile):
    r"""
    mat['Data']['neural_data_M1'][0, 0][:, 0][:, :] reaches x neurons x time_bins
    mat['Data']['neural_data_PMd'] PMd instead of M1
    mat['Data']['kinematics'][0, 0][:, 0][:, :] reaches x time_bins x [x, y, vx, vy, ax, ay, time] units cm and sec
    
    References:
    
    [1] `Extracellular neural recordings from macaque primary and dorsal premotor motor cortex during a sequential reaching task.',
        Matthew G. Perich, Patrick N. Lawlor, Konrad P. Kording, Lee E. Miller (2018); 
        http://dx.doi.org/10.6080/K0FT8J72
    
    """
    mat = scipy.io.loadmat(dataset)

    sample_bin = 0.01 # ms
    reaches = mat['Data']['neural_data_M1'][0, 0][:, 0].shape[0]
    spktrain_PMd = mat['Data']['neural_data_PMd'][0, 0][:, 0]
    
    spktrain_M1 = []
    spktrain_PMd = []
    kinematics = []
    trial_num = []
    reach_num = []
    target_on_ind = []
    cue_on_time = []
    reach_start_end = []
    for i in range(reaches):
        spktrain_M1.append(mat['Data']['neural_data_M1'][0, 0][i, 0])
        spktrain_PMd.append(mat['Data']['neural_data_PMd'][0, 0][i, 0])
        kinematics.append(mat['Data']['kinematics'][0, 0][i, 0])
        trial_num.append(mat['Data']['trial_num'][0, 0][i, 0][0, 0])
        reach_num.append(mat['Data']['reach_num'][0, 0][i, 0][0, 0])
        target_on_ind.append(mat['Data']['target_on'][0, 0][i, 0][:, 0].nonzero()[0][0])
        reach_start_end.append([mat['Data']['reach_st'][0, 0][i, 0][0, 0], mat['Data']['reach_end'][0, 0][i, 0][0, 0]])
        cue_on_time.append(mat['Data']['cue_on'][0, 0][1, 0][0, 0])
        
    block_info = mat['Data']['block_info'][0, 0]
    
    save_data = (sample_bin, spktrain_M1, spktrain_PMd, kinematics, \
                 trial_num, reach_num, target_on_ind, cue_on_time, reach_start_end, reaches, \
                 block_info)
    pickle.dump(save_data, open(savefile, 'wb'), pickle.HIGHEST_PROTOCOL)
    
    
    
    
def odoherty_m1(dataset, savefile):
    r"""
    https://zenodo.org/record/3854034#.YdhkgdvLcax
 
    
    References:
    
    @misc{ODoherty:2017,
      author = {O'{D}oherty, Joseph E. and Cardoso, Mariana M. B. and Makin, Joseph G. and Sabes, Philip N.},
      title  = {Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex electrophysiology},
      doi    = {10.5281/zenodo.788569},
      url    = {https://doi.org/10.5281/zenodo.788569},
      month  = may,
      year   = {2017}
    }
    
    """
    
    return
    
    
    
    


def ecker_V1(datadir, savefile, clean_select):
    r"""
    date = mat['data'][8, 0][0, 0][0][0]
    ID = mat['data'][8, 0][0, 0][1][0]
    orientation[k], contrast[k] = mat['data'][8, 0][0, 0][2][0,k]
    contamination = mat['data'][8, 0][0, 0][3][0]
    tetrode = mat['data'][8, 0][0, 0][4][0]
    spikes = mat['data'][8, 0][0, 0][5]    single units x conditions x time bins x repetitions
    times mat['data'][8, 0][0, 0][6][0]    ms units
    
    condition shape[0]
    trials = spikes.shape[1]
    time_bins = spikes.shape[2]
    units = spikes.shape[3]
    
    References:
    
    [1] `Decorrelated Neuronal Firing in Cortical Microcircuits', 
        Alexander S. Ecker  and Philipp Berens  and Georgios A. Keliris  and Matthias Bethge and Nikos K. Logothetis  and Andreas S. Tolias;
        Science 2010
    """
    data_static = datadir + 'macaque_V1/data_v1_binned_static/data_v1_binned_static.mat'
    data_moving = datadir + 'macaque_V1/data_v1_binned_moving/data_v1_binned_moving.mat'

    bin_size = 10 # ms
    datasets_static = []
    datasets_moving = []
    
    mat = scipy.io.loadmat(data_static)
    datasets = mat['data'].shape[0]
    for dataset_id in range(datasets):
        dd = mat['data'][dataset_id, 0][0, 0]
        
        date = dd[0][0]
        ID = dd[1][0]
        if ID == 'H':
            c_ind = 0
            o_ind = 1
        else:
            c_ind = 1
            o_ind = 0
    
        conditions = len(dd[2][0])
        contrast = [dd[2][0, k][c_ind][0, 0] for k in range(conditions)]
        orientation =[dd[2][0, k][o_ind][0, 0] for k in range(conditions)]

        clean = dd[3][0] < clean_select
        spikes = np.transpose(dd[5][clean, :, :, :], (1, 3, 2, 0)) # cond, trial, time, unit

        datasets_static.append((date, ID, contrast, orientation, spikes))

        
    mat = scipy.io.loadmat(data_moving)
    datasets = mat['data'].shape[0]
    for dataset_id in range(datasets):
        dd = mat['data'][dataset_id, 0][0, 0]
        
        date = dd[0][0]
        ID = dd[1][0]
        ID = dd[1][0]
        if ID == 'H':
            c_ind = 0
            o_ind = 1
        else:
            c_ind = 1
            o_ind = 0
    
        conditions = len(dd[2][:, 0])
        contrast = [dd[2][:, 0][k][c_ind][0, 0] for k in range(conditions)]
        direction = [dd[2][:, 0][k][o_ind][0, 0] for k in range(conditions)]
        
        clean = dd[3][:, 0] < clean_select
        spikes = np.transpose(dd[5][clean, :, :, :], (1, 3, 2, 0))

        datasets_moving.append((date, ID, contrast, direction, spikes))

        
    save_data = (bin_size, datasets_static, datasets_moving)
    pickle.dump(save_data, open(savefile, 'wb'), pickle.HIGHEST_PROTOCOL)


    

def s1(datadir, savefile):
    """
    Chowdhury, Raeed; Glaser, Joshua; Miller, Lee (2020), Data from: Area 2 of primary somatosensory cortex encodes kinematics of the whole arm, Dryad, Dataset, https://doi.org/10.5061/dryad.nk98sf7q7
    Abstract

    Proprioception, the sense of body position, movement, and associated forces, remains poorly understood, despite its critical role in movement. Most studies of area 2, a proprioceptive area of somatosensory cortex, have simply compared neurons' activities to the movement of the hand through space. By using motion tracking, we sought to elaborate this relationship by characterizing how area 2 activity relates to whole arm movements. We found that a whole-arm model, unlike classic models, successfully predicted how features of neural activity changed as monkeys reached to targets in two workspaces. However, when we then evaluated this whole-arm model across active and passive movements, we found that many neurons did not consistently represent the whole arm over both conditions. These results suggest that 1) neural activity in area 2 includes representation of the whole arm during reaching and 2) many of these neurons represented limb state differently during active and passive movements.


    %% Set up meta info
    if ispc
        dataroot = 'G:\raeed\project-data\limblab\s1-kinematics';
    else
        dataroot = '/data/raeed/project-data/limblab/s1-kinematics';
    end
    
    file_info = dir(fullfile(dataroot,'reaching_experiments','*COactpas*.mat'));
    filenames = horzcat({file_info.name})';

    % plotting variables
    monkey_names = {'C','H'};
    arrayname = 'S1';
    neural_signals = [arrayname '_FR'];
    model_aliases = {'ext','handelbow'};
    models_to_plot = {neural_signals,'ext','handelbow'};
    model_titles = {'Actual Firing','Hand-only','Whole-arm'};
    session_colors = [...
        102,194,165;...
        252,141,98;...
        141,160,203]/255;

    %% Loop through trial data files to clean them up
        trial_data_cell = cell(1,length(filenames));
        for filenum = 1:length(filenames)
            %% load and preprocess data
            td = load(fullfile(dataroot,'reaching_experiments',[filenames{filenum}]));

            % rename trial_data for ease
            td = td.trial_data;

            % first process marker data
            % find times when markers are NaN and replace with zeros temporarily
            for trialnum = 1:length(td)
                markernans = isnan(td(trialnum).markers);
                td(trialnum).markers(markernans) = 0;
                td(trialnum) = smoothSignals(td(trialnum),struct('signals','markers'));
                td(trialnum).markers(markernans) = NaN;
                clear markernans
            end

            % get marker velocity
            td = getDifferential(td,struct('signals','markers','alias','marker_vel'));

            % get speed and ds
            td = getNorm(td,struct('signals','vel','field_extra','_norm'));
            td = getDifferential(td,struct('signals','vel_norm','alias','dvel_norm'));

            % remove unsorted neurons
            unit_ids = td(1).([arrayname '_unit_guide']);
            unsorted_units = (unit_ids(:,2)==0);
            new_unit_guide = unit_ids(~unsorted_units,:);
            for trialnum = 1:length(td)
                td(trialnum).(sprintf('%s_unit_guide',arrayname)) = new_unit_guide;

                spikes = td(trialnum).(sprintf('%s_spikes',arrayname));
                spikes(:,unsorted_units) = [];
                td(trialnum).(sprintf('%s_spikes',arrayname)) = spikes;
            end

            % prep trial data by getting only rewards and trimming to only movements
            % split into trials
            td = splitTD(...
                td,...
                struct(...
                    'split_idx_name','idx_startTime',...
                    'linked_fields',{{...
                        'trialID',...
                        'result',...
                        'bumpDir',...
                        'tgtDir',...
                        'ctrHoldBump',...
                        'ctrHold',...
                        }},...
                    'start_name','idx_startTime',...
                    'end_name','idx_endTime'));
            [~,td] = getTDidx(td,'result','R');
            td = reorderTDfields(td);

            % clean nans out...?
            nanners = isnan(cat(1,td.tgtDir));
            td = td(~nanners);
            fprintf('Removed %d trials because of missing target direction\n',sum(nanners))
            biggers = cat(1,td.ctrHoldBump) & abs(cat(1,td.bumpDir))>360;
            td = td(~biggers);
            fprintf('Removed %d trials because bump direction makes no sense\n',sum(biggers))

            % remove trials where markers aren't present
            bad_trial = false(length(td),1);
            for trialnum = 1:length(td)
                if any(any(isnan(td(trialnum).markers)))
                    bad_trial(trialnum) = true;
                end
            end
            td(bad_trial) = [];
            fprintf('Removed %d trials because of missing markers\n',sum(bad_trial))

            % remove trials where muscles aren't present
            bad_trial = false(length(td),1);
            for trialnum = 1:length(td)
                if any(any(isnan(td(trialnum).muscle_len) | isnan(td(trialnum).muscle_vel)))
                    bad_trial(trialnum) = true;
                end
            end
            td(bad_trial) = [];
            fprintf('Removed %d trials because of missing muscles\n',sum(bad_trial))

            % for C_20170912, trial structure is such that active and passive are part of the same trial--split it up
            if strcmpi(td(1).monkey,'C') && contains(td(1).date_time,'2017/9/12')
                td_copy = td;
                [td_copy.ctrHoldBump] = deal(false);
                td = cat(2,td,td_copy);
                clear td_copy
            end

            % split into active and passive
            [~,td_act] = getTDidx(td,'ctrHoldBump',false);
            [~,td_pas] = getTDidx(td,'ctrHoldBump',true);

            % find the relevant movmement onsets
            td_act = getMoveOnsetAndPeak(td_act,struct(...
                'start_idx','idx_goCueTime',...
                'start_idx_offset',20,...
                'peak_idx_offset',20,...
                'end_idx','idx_endTime',...
                'method','peak',...
                'peak_divisor',10,...
                'min_ds',1));
            td_pas = getMoveOnsetAndPeak(td_pas,struct(...
                'start_idx','idx_bumpTime',...
                'start_idx_offset',-5,... % give it some wiggle room
                'peak_idx_offset',-5,... % give it some wiggle room
                'end_idx','idx_goCueTime',...
                'method','peak',...
                'peak_divisor',10,...
                'min_ds',1));
            % throw out all trials where bumpTime and movement_on are more than 3 bins apart
            bad_trial = isnan(cat(1,td_pas.idx_movement_on)) | abs(cat(1,td_pas.idx_movement_on)-cat(1,td_pas.idx_bumpTime))>3;
            td_pas = td_pas(~bad_trial);
            fprintf('Removed %d trials because of bad movement onset\n',sum(bad_trial))

            % even out sizes and put back together
            minsize = min(length(td_act),length(td_pas));
            td_act = td_act(1:minsize);
            td_pas = td_pas(1:minsize);
            td_trim = cat(2,td_act,td_pas);

            % remove low firing neurons
            td_trim = removeBadNeurons(td_trim,struct(...
                'min_fr',1,...
                'fr_window',{{'idx_movement_on',0;'idx_movement_on',11}},...
                'calc_fr',true));

            trial_data_cell{filenum} = td_trim;
        end

    %% Plot trial info (hand speed and example rasters)
        for filenum = 1:length(trial_data_cell)
            %% load and preprocess data
            td = trial_data_cell{filenum};

            % trim to just movements
            td = trimTD(td,{'idx_movement_on',-50},{'idx_movement_on',60});

            %% Plot out hand speed
            figure('defaultaxesfontsize',18)
            for trial = 1:length(td)
                timevec = ((1:length(td(trial).vel_norm))-td(trial).idx_movement_on)*td(trial).bin_size;
                if td(trial).ctrHoldBump
                    plot(timevec,td(trial).vel_norm,'r')
                else
                    plot(timevec,td(trial).vel_norm,'k')
                end
                hold on
            end
            plot(zeros(2,1),ylim,'--k','linewidth',2)
            hold on
            plot(repmat(0.12,2,1),ylim,'--k','linewidth',2)
            xlabel('Time from movement onset (s)')
            ylabel('Hand speed (cm/s)')
            set(gca,'box','off','tickdir','out','xtick',[-0.5 0 0.12 0.5])
            set(gcf,'renderer','Painters')
            suptitle(sprintf('Monkey %s %s',td(1).monkey, td(1).date_time))

            %% Plot out example rasters for each direction
            dirs = unique(cat(1,td.tgtDir));
            figure('defaultaxesfontsize',18)
            for dirnum = 1:length(dirs)
                % pick a random active and random passive trial with this direction
                act_idx = getTDidx(td,'tgtDir',dirs(dirnum),'ctrHoldBump',false,'rand',1);
                pas_idx = getTDidx(td,'bumpDir',dirs(dirnum),'ctrHoldBump',true,'rand',1);
                td_temp = td([act_idx pas_idx]);

                for trialnum = 1:length(td_temp)
                    spikes = getSig(td_temp(trialnum),'S1_spikes')';
                    timevec = ((1:size(spikes,2))-td_temp(trialnum).idx_movement_on)*td_temp(trialnum).bin_size;
                    % active on left, passive on right
                    subplot(length(dirs),length(td_temp),(dirnum-1)*length(td_temp)+trialnum)
                    % neurons
                    for neuronnum = 1:size(spikes,1)
                        spike_times = timevec(spikes(neuronnum,:)>0);
                        scatter(spike_times,repmat(neuronnum,size(spike_times)),5,'k','filled')
                        hold on
                    end
                    plot(zeros(1,2),[0 size(spikes,1)+1],'--k')
                    plot(ones(1,2)*0.12,[0 size(spikes,1)+1],'--k')
                    xlabel('Time from movement onset (s)')
                    set(gca,'box','off','tickdir','out','xtick',[-0.5 0 0.12 0.5],'ytick',[])
                end
                subplot(length(dirs),length(td_temp),(dirnum-1)*length(td_temp)+1)
                ylabel(sprintf('Direction %f',dirs(dirnum)))
            end
            suptitle(sprintf('Monkey %s %s',td(1).monkey, td(1).date_time))
        end


        # S1
    %% Set up meta info and load trial data
        if ispc
            dataroot = 'G:\raeed\project-data\limblab\s1-kinematics';
        else
            dataroot = '/data/raeed/project-data/limblab/s1-kinematics';
        end

        % load data
        file_info = dir(fullfile(dataroot,'reaching_experiments','*TRT*'));
        filenames = horzcat({file_info.name})';

        % save directory information (for convenience, since this code takes a while)
        savefile = true;
        if savefile
            savedir = fullfile(dataroot,'reaching_experiments','EncodingResults');
            if ~exist(savedir,'dir')
                mkdir(savedir)
            end
            run_date = char(datetime('today','format','yyyyMMdd'));
            savename = sprintf('encoderResults_run%s.mat',run_date);
        end

        arrayname = 'S1';
        monkey_names = {'C','H','L'};
        included_models = {'ext','handelbow','ego','joint','musc','extforce'}; % models to calculate encoders for
        models_to_plot = {'ext','handelbow'}; % main models of the paper
        not_plot_models = setdiff(included_models,models_to_plot);

        % colors for pm, dl conditions and sessions
        cond_colors = [...
            231,138,195;...
            166,216,84]/255;
        session_colors = [...
            102,194,165;...
            252,141,98;...
            141,160,203]/255;

    %% Loop through trial data files to clean up
        trial_data_cell = cell(1,length(filenames));
        for filenum = 1:length(filenames)
            %% Load data
            td = load(fullfile(dataroot,'reaching_experiments',[filenames{filenum}]));

            % rename trial_data for ease
            td = td.trial_data;

            % first process marker data
            % find times when markers are NaN and replace with zeros temporarily
            for trialnum = 1:length(td)
                markernans = isnan(td(trialnum).markers);
                td(trialnum).markers(markernans) = 0;
                td(trialnum) = smoothSignals(td(trialnum),struct('signals','markers'));
                td(trialnum).markers(markernans) = NaN;
                clear markernans
            end

            % get marker velocity
            td = getDifferential(td,struct('signals','markers','alias','marker_vel'));

            % remove unsorted neurons
            unit_ids = td(1).S1_unit_guide;
            unsorted_units = (unit_ids(:,2)==0);
            new_unit_guide = unit_ids(~unsorted_units,:);
            for trialnum = 1:length(td)
                td(trialnum).(sprintf('%s_unit_guide',arrayname)) = new_unit_guide;

                spikes = td(trialnum).(sprintf('%s_spikes',arrayname));
                spikes(:,unsorted_units) = [];
                td(trialnum).(sprintf('%s_spikes',arrayname)) = spikes;
            endtrial_data

            % add firing rates in addition to spike counts
            td = addFiringRates(td,struct('array',arrayname));

            % prep trial data by getting only rewards and trimming to only movements
            % split into trials
            td = splitTD(...
                td,...
                struct(...
                    'split_idx_name','idx_startTime',...
                    'linked_fields',{{...
                        'trialID',...
                        'result',...
                        'spaceNum',...
                        'bumpDir',...
                        }},...
                    'start_name','idx_startTime',...
                    'end_name','idx_endTime'));
            [~,td] = getTDidx(td,'result','R');
            td = reorderTDfields(td);

            % for active movements
            % remove trials without a target start (for whatever reason)
            td(isnan(cat(1,td.idx_targetStartTime))) = [];
            td = trimTD(td,{'idx_targetStartTime',0},{'idx_endTime',0});

            % remove trials where markers aren't present
            bad_trial = false(length(td),1);
            for trialnum = 1:length(td)
                if any(any(isnan(td(trialnum).markers)))
                    bad_trial(trialnum) = true;
                end
            end
            td(bad_trial) = [];
            fprintf('Removed %d trials because of missing markers\n',sum(bad_trial))

            % remove trials where muscles aren't present
            bad_trial = false(length(td),1);
            for trialnum = 1:length(td)
                if any(any(isnan(td(trialnum).muscle_len) | isnan(td(trialnum).muscle_vel)))
                    bad_trial(trialnum) = true;
                end
            end
            td(bad_trial) = [];
            fprintf('Removed %d trials because of missing muscles\n',sum(bad_trial))

            trial_data_cell{filenum} = td;
        end

    %% Plot example rasters
        num_trials = 2;
        for filenum = 4%1:length(trial_data_cell)
            %% Load data
            td = trial_data_cell{filenum};

            %% choose a random few trials and plot
            figure('defaultaxesfontsize',18)
            max_x = 0;
            for spacenum = 1:2
                [~,td_temp] = getTDidx(td,'spaceNum',spacenum,'rand',num_trials);
                spikes = getSig(td_temp,'S1_spikes')';
                timevec = (1:size(spikes,2))*td_temp(1).bin_size;
                subplot(1,2,spacenum)
                for neuronnum = 1:size(spikes,1)
                    spike_times = timevec(spikes(neuronnum,:)>0);
                    scatter(spike_times,repmat(neuronnum,size(spike_times)),5,'k','filled')
                    hold on
                end
                trial_end = 0;
                for trialnum = 1:num_trials
                    plot(td_temp(trialnum).bin_size*repmat(td_temp(trialnum).idx_otHoldTime,2,1)+trial_end,...
                        repmat([0;size(spikes,1)],1,length(td_temp(trialnum).idx_otHoldTime)),...
                        '--','color',cond_colors(spacenum,:))
                    plot(td_temp(trialnum).bin_size*repmat(td_temp(trialnum).idx_targetStartTime,2,1)+trial_end,...
                        [0;size(spikes,1)],...
                        '--k')
                    plot(td_temp(trialnum).bin_size*repmat(td_temp(trialnum).idx_endTime,2,1)+trial_end,...
                        [0;size(spikes,1)],...
                        '--k')
                    trial_end = trial_end + (td_temp(trialnum).idx_endTime-1)*td_temp(trialnum).bin_size;
                end
                max_x = max(max_x,trial_end);
                xlabel 'Time (s)'
                set(gca,'box','off','tickdir','out')
            end
            for spacenum = 1:2
                subplot(1,2,spacenum)
                xlim([0 max_x+0.5]);
            end
        end
    """
    
    
    return





def rat_A1(datadir, save_file):
    """
    Baratham, Vyassa L.; Dougherty, Maximilian E.; Ledochowitsch, Peter; Maharbiz, 
    Michel M.; Bouchard, Kristofer E (2021); Recordings and simulations of ECoG 
    responses from rat auditory cortex during presentation of pure tone pips. 
    CRCNS.org  https://doi.org/10.6080/K0VT1Q93
    """
    
    
    return




def widloski(datadir, savedir, session, tbin=0.001):
    """
    There are three sessions recorded back to back on the same day. You can restrict analysis to any one of them using the session times:
    Session 1: 11365 - 14408 sec
    Session 2: 18791 - 21200 sec
    Session 3 (the open field) : 25742 - 27802 sec

    The "behavior" file contains "positions" variable concatenated for all three sessions.
    1st col: time stamps (in sec); 2nd/3rd cols: x and y positions; 4th col: speed; 5th col: head direction.

    The "clusters" file is a struct, with each element of the struct a putative cluster. One of the fields is "spkTime" (in sec). There are also fields containing cluster metrics, which you'll want to use to filter clusters for decoding. I use: 
     noise_overlap<0.03 & isolation>0.9 & peak_snr>1.5 + some min threshold on firing rate (like >0.01 Hz for any given session).

    The "binDecoding_08" file contains the result of a sliding decoder (80 ms wide, shifted in 5 ms increments) applied to the entirety of each session. The data consists of the changing center of mass of the posterior as well as the posterior peak value and "spread" (how punctate the posterior is) as a function of time. There are 3 elements in the struct called "decoder_binDecoding", one for each session.

    Lastly, if you only want replays, then look at the "replayEvents" file. Like the "binDecoding_08" file, there is a struct with 3 elements corresponding to the three sessions, except that individual candidate replays have been separated out. Each replay has many subfields, but the main ones to care about are "timePoints" (the beginning and end of the event), "replay" (the center of mass changing across time bins (same as in "bindDecoding_08")), and the "dispersion" (characterizes how spread out in space the event is).  I select replays that have a minimum duration of 100 msec and a dispersion of > 12.
    """
    # inclusive boundaries
    if session == 1:
        start, end = 0, 89789
    elif session == 2:
        start, end = 89790, 153419
    elif session == 3:
        start, end = 153420, 215220
    else:
        raise ValueError('Sessions go from 1 to 3')
        
    dataset = 'widloski_sess{}'.format(session)
    session -= 1
        
    # data
    behaviour_file = datadir+'behavior.mat'
    decoded_file = datadir+'binDecoding_08.mat'
    clusters_file = datadir+'clusters.mat'
    replay_file = datadir+'replayEvents.mat'

    behaviour = scipy.io.loadmat(behaviour_file)
    decoded = h5py.File(decoded_file, 'r')
    clusters = scipy.io.loadmat(clusters_file)
    replay = scipy.io.loadmat(replay_file)

    
    timestamps = behaviour['positions'][start:end, 0]
    x = behaviour['positions'][start:end, 1]
    y = behaviour['positions'][start:end, 2]
    s = behaviour['positions'][start:end, 3]
    hd = np.unwrap(behaviour['positions'][start:end, 4])
    
    
    # clusters
    num_clusters = clusters['clusters'].shape[1]
    names = ['bursting_parent','dur_sec', 'firing_rate', 'isolation', 
             'noise_overlap', 'num_events', 'overlap_cluster', 
             'peak_amp', 'peak_noise', 'peak_snr', 't1_sec', 't2_sec']

    activity_type = []
    properties = []
    spktimes = []
    for c in range(num_clusters):
        activity_type.append(clusters['clusters'][0, c][0][-1, 0][0])

        values = [v[0, 0] for v in clusters['clusters'][0, c][3][0, 0].item()]
        properties.append(dict(zip(names, values)))

        spktimes.append(clusters['clusters'][0, c][5][:, 0])
    
    
    # replays
    num_replays = replay['decoder_replay'][0, session].item()[0].shape[1]
    dtypes = [
        'indNaN', 'indData', 'timeBins', 'timePoints', 'duration', 'maxJump_NaN',  
        'maxJump_NaNremoved', 'maxJump_NaNremoved_time', 'replay', 'distance', 'linCorr', 
        'curvature', 'linearity', 'dispersion', 'mvl', 'alpha', 'diffusionCoef', 
        'spikeDensityPeak', 'spikeDensityMean', 'posteriorSpreadMax', 'posteriorSpreadMean', 
        'ratLoc', 'ratSpeed', 'ratHD', 'meanAngDisplacement_futPath', 
        'meanAngDisplacement_pastPath', 'angularDisplacement_past', 'angularDisplacement_future'
    ]

    replay_dicts = []
    for r in range(num_replays):
        values = list(replay['decoder_replay'][0, session].item()[0][0, r].item())
        replay_dicts.append(dict(zip(dtypes, values)))

       
    ### resample at 1 ms ###
    time_fine = np.arange(
        int(np.floor(timestamps[0]/tbin)), 
        int(np.floor(timestamps[-1]/tbin)))*tbin
    x = linear_interpolate(timestamps, x, time_fine)
    y = linear_interpolate(timestamps, y, time_fine)
    s = linear_interpolate(timestamps, s, time_fine)
    hd = linear_interpolate(timestamps, hd, time_fine)

    spike_ind = []
    for c in range(num_clusters):
        st = spktimes[c] - time_fine[0]
        inds = np.round(st / tbin).astype(int)
        inds = inds[(inds >= start) & (inds <= end)]
        spike_ind.append(inds)
    
    savef = savedir + dataset + '.p'
    pickle.dump((tbin, x, y, s, hd, spike_ind, 
                 activity_type, spktimes, properties, replay_dicts), 
                open(savef, 'wb'), pickle.HIGHEST_PROTOCOL)


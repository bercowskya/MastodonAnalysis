import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import scipy as sc
import untangle
import xml.etree.ElementTree as ET
from scipy.stats import gaussian_kde, pearsonr, gaussian_kde
import scipy.spatial.distance


# Obtain all the features that are in the .xml file which has been generated when the data
# is converted to .hdf5 using BigDataViewer/BigStitcher/MultiviewReconstruction in Fiji
class xml_features:
    def __init__(self, path_xml):
        # Parse .xml file
        obj = untangle.parse(path_xml)
        # Data Features
        self.channels = len(obj.SpimData.SequenceDescription.ViewSetups.Attributes)
        ch = self.channels
        self.dim = 3
        
        if ch > 1:
            self.width = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[0])
            self.height = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[1])
            self.n_slices = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].size.cdata.split()[2])

            self.x_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.split()[0])
            self.y_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.split()[1])
            self.z_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.size.cdata.split()[2])
            
            self.units = obj.SpimData.SequenceDescription.ViewSetups.ViewSetup[0].voxelSize.unit.cdata
        else:
            self.width = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[0])
            self.height = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[1])
            self.n_slices = int(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.size.cdata.split()[2])

            self.x_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.split()[0])
            self.y_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.split()[1])
            self.z_pixel = float(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.size.cdata.split()[2])
            
            self.units = obj.SpimData.SequenceDescription.ViewSetups.ViewSetup.voxelSize.unit.cdata
            

        #self.channels = len(obj.SpimData.SequenceDescription.ViewSetups.ViewSetup)
      
        while True:
            try:
                self.n_frames = len(obj.SpimData.SequenceDescription.Timepoints.integerpattern.cdata.split())
                break
            except AttributeError:
                pass  # fallback to dict
            try:
                self.n_frames = int(obj.SpimData.SequenceDescription.Timepoints.last.cdata.split()[0])
                break
            except KeyError:
                raise AttributeError("There is something wrong with the .xml file") from None
        



# Obtain all the information that comes in the .csv files (-vertices and -edges) that is 
# generated with Mastodon once you have computed the features and saved as .csv
class csv_features:
    def __init__(self, path_csv, path_xml):

        xml_params = xml_features(path_xml) # To use the parameters from the other function
       
        # Read data from file 'file_path/file_name.csv' 
        data = pd.read_csv('%s-vertices.csv'%(path_csv), header=[0,1],skip_blank_lines=True,encoding = 'unicode_escape')
        edges = pd.read_csv('%s-edges.csv'%(path_csv), header=[0,1],skip_blank_lines=True,encoding = 'unicode_escape')   

        n_features = len(data.keys()) # All features that appear in data

        '''
        Rearanging and capturing all the features from the .csv files called -vertices and -edges
        '''

        # Number of links in each spot
        for i in range(n_features):
            if data.keys()[i][0]=='Spot N links':
                self.n_links = np.array(data[data.keys()[i]][1:], dtype=int)
                break

        # ID --> For linking spot source to spot target (for cell divisions)
        for i in range(n_features):
            if data.keys()[i][0]=='ID':
                self.IDs = np.array(data[data.keys()[1]][1:], dtype=int)
                break

        # Spot source ID
        for i in range(n_features):
            if edges.keys()[i][1]=='Source spot id' and len(edges.Label)>1: # len(edges.Label) --> In case there are no tracks
                self.spot_source = np.array(edges[edges.keys()[i]][1:], dtype=int)
                break

        # Spot target ID        
        for i in range(n_features):
            if edges.keys()[i][1]=='Target spot id' and len(edges.Label)>1:        
                self.spot_target = np.array(edges[edges.keys()[i]][1:], dtype=int)
                break

        # Spot frame
        for i in range(n_features):
            if data.keys()[i][0]=='Spot frame':
                self.frames = np.array(data[data.keys()[i]][1:], dtype=int)
                break
                
        # Index to start the rest of the features
        for i in range(n_features):
            if data.keys()[i][0]=='Spot gaussian-filtered intensity':
                ind = i
                break

        # Spot gaussian-filtered intensity: mean, std and median
        aux = ind
        self.mean = []
        self.std = []
        self.median = []

        for i in range(xml_params.channels):
            self.mean.append(np.array(list(map(float,data[data.keys()[aux]][1:]))))
            self.std.append(np.array(list(map(float,data[data.keys()[aux+xml_params.channels]][1:]))))
            self.median.append(np.array(list(map(float,data[data.keys()[aux+(xml_params.channels*2)]][1:]))))
            aux += 1

        ind += xml_params.channels*3

        # X,Y,Z Position for each individual spot
        self.pos = []
        for i in range(xml_params.dim):
            self.pos.append(np.array(data[data.keys()[ind+i]][1:], dtype = float))

        ind += (xml_params.dim+xml_params.channels+1)

        # Track ID
        self.track_id = np.array(data[data.keys()[ind]][1:], dtype=int)
        self.n_tracks = len(np.unique(self.track_id))
        ind +=1

        # Number of spots per track
        self.n_spots = np.array(data[data.keys()[ind]][1:], dtype=int)
        ind +=1

        # Tags (if any)
        self.tags = []
        if len(data.keys())==ind:
            print('There are no Tags.')
        else:
            # Tags are in the end of the .csv file, if any
            for i in range(len(data.keys())-ind):
                if len(data.keys()[ind+i]):
                    self.tags += [data.keys()[ind+i]]



# Order tracks acoording to whether they divide or not: 
# using the data obtained from csv_features
class ordering_tracks:
    def __init__(self, path_csv, path_xml):
        xml_params = xml_features(path_xml) # To use the parameters from the other function
        csv_params = csv_features(path_csv, path_xml) # To use the parameters from the other function

        # ORDER TRACKS BY THEIR ID
        # Number of tracks
        len_id = len(np.unique(csv_params.track_id))

        # Dictionaries to save mean, std and XYZ coordinates, links, source and target ids of spots
        mean_by_track = {key:[] for key in np.arange(xml_params.channels)}
        median_by_track = {key:[] for key in np.arange(xml_params.channels)}
        frames_by_track = []
        std_by_track = {key:[] for key in np.arange(xml_params.channels)}
        pos_by_track = {key:[] for key in np.arange(xml_params.dim)} # for XYZ Data or XY Data
        links_by_track = []
        source_by_track = []
        target_by_track = []
        ids_by_track = []
        cells_track = [] # Keep tracks of the spots which have a track
        
        for i in range(len_id):
            idx = np.where(csv_params.track_id == np.unique(csv_params.track_id)[i])[0]
            
            # In case there are spots with tracks
            if len(idx)>1:
                sorted_frames = csv_params.frames[idx].argsort()
                
                # Save the sorted frames
                frames_by_track.append(np.sort(csv_params.frames[idx]))
                
                # Save the sorted links (1, 2 or 3 links in case of division)
                links_by_track.append(csv_params.n_links[idx][sorted_frames])

                # IDs by tracks
                ids_by_track.append(csv_params.IDs[idx][sorted_frames])

                # Sorted spot source ID and spot target ID
                ind_ids = csv_params.IDs[idx][sorted_frames]
                source_by_track.append(csv_params.spot_source[np.array([ind for ind,element in enumerate\
                                                                        (csv_params.spot_source) if element in ind_ids[:-1]])])
                target_by_track.append(csv_params.spot_target[np.array([ind for ind,element in enumerate\
                                                                        (csv_params.spot_target) if element in ind_ids[1:]])])

                # Save the coordinates
                for j in range(xml_params.dim):
                    pos_by_track[j].append(csv_params.pos[j][idx][sorted_frames])

                # Save the mean, median and std according to their channel
                for j in range(xml_params.channels):
                    mean_by_track[j].append(csv_params.mean[j][idx][sorted_frames])
                    median_by_track[j].append(csv_params.median[j][idx][sorted_frames])
                    std_by_track[j].append(csv_params.std[j][idx][sorted_frames])
                
                # cells with tracks
                cells_track.append(i)
                
            # If there are spots without any tracks    
            else:
                continue


        # ORDER TRACKS ACCORDING WHETHER THEY DIVIDE OR NOT
        self.spots_features = {key:[] for key in ['Frames', 'Mean', 'Median', 'ID', 'DivID', 'X', 'Y']}
        # DivID : If division, the an ID equal to its sibling. If not, nan              
        DivID = 0
        self.n_tracks_divs = 0 # Number of tracks including divisions
        self.n_division_tracks = 0 # Number of tracks with divisions

        for i in range(len(cells_track)):
            # Are there any divisions in the track?  
            # (e.g. the spot divides in two different opportunities during all the timeseries)
            n_divs = len(list(map(int,np.where(links_by_track[i]>2)[0])))
            
            # How many times the spot divides per division?
            # (e.g. in one specific division, in how many daughters the spot divided?)
            #n_divs_cell = links_by_track[links_by_track[i]>2]
            
            if n_divs == 0: # There is no cell division
                self.spots_features['Frames'].append(frames_by_track[i])
                self.spots_features['Mean'].append(mean_by_track[0][i])
                self.spots_features['Median'].append(median_by_track[0][i])
                self.spots_features['ID'].append(ids_by_track[i])
                self.spots_features['DivID'].append(0)
                self.spots_features['X'].append(pos_by_track[0][i])
                self.spots_features['Y'].append(pos_by_track[1][i])
                self.spots_features['Z'].append(pos_by_track[2][i])
                
                self.n_tracks_divs += 1
                
            else: # Otherwise, there is cell division
                DivID += 1
                div_vect = [] # one vector for [each division+1] we want to keep track of
                val0 = np.where(links_by_track[i]==3)[0][0] # index of first division
                
                # save the IDs up to first division for all tracks
                for j in range(n_divs+1):
                    div_vect.append(ids_by_track[i][:val0+1].tolist()) # IDS[0:first division]
                
                # store the list of already saved IDs to not use them again
                list_idx_sources_used = [] # To save the indices of used sources spots
                list_used = ids_by_track[i][:val0+1].tolist() # list of IDs used --> finish while loop when all IDs used
                # while we have not stored all IDs, loop across tracks and fill them with targets (if not in list_used)
                
                while not(all(elem in list_used for elem in ids_by_track[i])):
                    
                    for j in range(n_divs+1):
                        idx = np.where(source_by_track[i] == div_vect[j][-1])[0]
                        # In the exact moment of division
                        if len(idx) > 1: # point of division
                            cond = True
                            k=0
                            while cond: 
                                if idx[k] not in list_idx_sources_used:
                                    list_idx_sources_used.append(idx[k])
                                    idx = idx[k]
                                    cond = False
                                else:
                                    k += 1
                                    
                        # In the cases where there is no division
                        if np.size(idx) == 1:
                            div_vect[j].append(int(target_by_track[i][idx]))
                            list_used.append(int(target_by_track[i][idx]))
                            
                        # This means it finished at least one of the tracks
                        if np.size(idx) == 0:
                            continue
                            
                # Save each division tracks with its corresponding division ID and its frames and mean
                for j in range(n_divs+1):
                    #inds = np.where(np.array(div_vect[j]) == np.array(ids_by_track[i]))[0] # Indices for the IDS of the tracks in one of the divisions
                    inds = [np.where(ids_by_track[i]==div_vect[j][ind])[0][0] for ind in range(len(div_vect[j]))]
                    self.spots_features['Frames'].append(frames_by_track[i][inds])
                    self.spots_features['Mean'].append(mean_by_track[0][i][inds])
                    self.spots_features['ID'].append(ids_by_track[i][inds])
                    self.spots_features['DivID'].append(DivID)
                    self.n_tracks_divs += 1
                    self.n_division_tracks += 1        



# Obtain the tracks in the same format as in csv_reader but coming from and _mamut.xml file which is the same
# for TrackMate, MaMuT and Mastodon
class xml_reader:
    def __init__(self, path_xml, path_real_xml):
        obj = untangle.parse(path_real_xml)
        xml_params = xml_features(path_xml) # To use the parameters from the other function

        dim = xml_params.dim
        channels = xml_params.channels
        n_frames = xml_params.n_frames
        mean = {key:[] for key in np.arange(channels)}
        std = {key:[] for key in np.arange(channels)}
        pos = {key:[] for key in np.arange(dim)}
        frames = []
        IDs = []
        axis = ['X', 'Y', 'Z']

        for i in range(n_frames):
            n_spots_in_frame = len(obj.TrackMate.Model.AllSpots.SpotsInFrame[i])
            for j in range(n_spots_in_frame):
                for k in range(channels):
                    mean[k].append(float(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['Spot_gaussian-filtered_intensity_Mean_ch%d'%k]))
                    std[k].append(float(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['Spot_gaussian-filtered_intensity_Std_ch%d'%k]))
                for k in range(dim):
                    pos[k].append(float(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['Spot_position_%s'%axis[k]]))
                frames.append(int(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['FRAME']))
                IDs.append(int(obj.TrackMate.Model.AllSpots.SpotsInFrame[i].Spot[j]['ID']))
                
        count = 0 # For all the number of tracks      

        spot_source = []
        spot_target = []
        n_tracks = []
        while True:
            # Check if there are only spots or also tracks
            try:
                len(obj.TrackMate.Model.AllTracks.Track[count].Edge)
                break
            except AttributeError:
                n_tracks = 0
                raise AttributeError("There are no tracks, just independent spots") from None
            # Keep in the loop until the last track 

        if len(n_tracks) == 0 : 
            while True:
                try:
                    len(obj.TrackMate.Model.AllTracks.Track[count].Edge)
                except IndexError:
                    break # End of tracks

                # Temporary save source and targets to later add them to another list
                aux_source = []
                aux_target = [] 
                for i in range(len(obj.TrackMate.Model.AllTracks.Track[count].Edge)): # For all spots in a track
                    aux_source.append(int(float(obj.TrackMate.Model.AllTracks.Track[count].Edge[i]['Link_target_IDs_Source_spot_id'])))
                    aux_target.append(int(float(obj.TrackMate.Model.AllTracks.Track[count].Edge[i]['Link_target_IDs_Target_spot_id'])))

                spot_source.append(aux_source)
                spot_target.append(aux_target)
                count += 1
        n_tracks = count

        ### Order according Track 
        # Dictionaries to save mean, std and XYZ coordinates, links, source and target ids of spots
        mean_by_track = {key:[] for key in np.arange(fts.channels)}
        frames_by_track = []
        std_by_track = {key:[] for key in np.arange(fts.channels)}
        pos_by_track = {key:[] for key in np.arange(fts.dim)} # for XYZ Data or XY Data
        links_by_track = []
        source_by_track = []
        target_by_track = []
        ids_by_track = []

        for i in range(n_tracks):
            
            # Obtain the indices of all the spots in one track "i"
            indices =  [idx for idx,element in enumerate(IDs) if element in spot_source[i]+[spot_target[i][-1]]]
            # Order these tracks according to their timepoint
            indices_ordered = np.array(frames)[indices]
            
            # Order the source and track ids according to the ordered indices
            source_by_track.append(np.array(spot_source[i])[indices_ordered])
            target_by_track.append(np.array(spot_target[i])[indices_ordered])
            
            # Perform the same for mean, std, frames, and position 
            frames_by_track.append(np.array(frames)[indices])
            ids_by_track.append(np.array(IDs)[indices_ordered])
            for j in range(channels):
                mean_by_track[j].append(np.array(mean[j])[indices_ordered])
                std_by_track[j].append(np.array(std[j])[indices_ordered])
            for j in range(dim):
                pos_by_track[j].append(np.array(pos[j])[indices_ordered])

        ### Order according to cell divisions


# Smooting function with a defined window length and stype of window
def smoothing_filter(x, window_len, window='hanning'):
    """
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y
        


# Calculate the peaks using a series of rules:
#
class peak_detection:
    def __init__(self, x, th_dist, th_dist_, window, prominence_val, prominence_val_, width_val, width_val_ ):

        # Initialize the variables to save the data analysis results
        T_plus_final = []
        T_minus_final = []
        A_plus_final = []
        A_minus_final = []
        cycles_final = []
        self.n_cycles = []

        # Initialize the variables to save the data analysis results that will be used to plot 
        self.cycles_T_minus_plot = []
        self.cycles_T_plus_plot = []
        self.cycles_A_minus_plot = []
        self.cycles_A_plus_plot = []

        # Counter in case there are no peaks detected it will go from 0 to 1
        self.count = 0 
        
        def pair_wise_dist(r):
            dists = scipy.spatial.distance.pdist(r[:, None], 'cityblock')
            return dists

        def distance_neighbors(r):
            pwd_peaks = pair_wise_dist(r)
            peaks_neighbors = np.zeros(len(r)-1)
            count = len(r)-2
            aux = 0
            for i in range(len(r)-1):
                peaks_neighbors[i] = pwd_peaks[i+aux] # Only take distance of neighboring peals
                aux = aux + count
                count -= 1
            return peaks_neighbors
                
        
        cA = smoothing_filter(x,window) # To find Maxima
        cA_ = cA*(-1) # To find minima
        
        # Maxima
        self.peaks = []
        self.peaks, _ = signal.find_peaks(cA, prominence=prominence_val, width=width_val)
        self.peaks = np.unique(self.peaks)
        # Minima
        self.peaks_ = []
        self.peaks_, _ = signal.find_peaks(cA_, prominence=prominence_val_, width=width_val_)
        self.peaks_ = np.unique(self.peaks_)

        # If there is no minima or no maxima, then let the user know
        if np.size(self.peaks) ==0 or np.size(self.peaks_) ==0:
            print("No peaks detected")
            self.count += 1

        else:
            if self.peaks[0]<self.peaks_[0]:
                self.peaks_ = np.concatenate((np.array([0]), self.peaks_)) # Add a 0 at the beginning
            if self.peaks_[-1]<self.peaks[-1]:
                self.peaks_ = np.concatenate((self.peaks_, np.array([len(cA)-1]))) # Add the last point 
            # Distance between the values next to each other
            # Indices where the distance between points is smaller than a threshold --> They must be the same point!
            
            # Maxima
            ind_avg = np.where(distance_neighbors(self.peaks) < th_dist)[0]
            while len(ind_avg) > 0:
                peaks_final = list(self.peaks)
                for i,val in enumerate(ind_avg):
                    peaks_final[i] = np.mean([peaks_final[i],peaks_final[i+1]])
                
                peaks_final = [peaks_final[i] for i in range(len(peaks_final)) if i-1 not in ind_avg]
                    
                peaks_final = np.array(peaks_final, dtype=int)        
                ind_avg = np.where(distance_neighbors(peaks_final) < th_dist)[0]
                # If there are still some other values to average, restart the cycle
                self.peaks = peaks_final
            
            # Minima
            ind_avg_ = np.where(distance_neighbors(self.peaks_) < th_dist_)[0]
            while len(ind_avg_) > 0:
                peaks_final_ = list(self.peaks_)
                for i,val in enumerate(ind_avg_):

                    peaks_final_[i] = np.mean([peaks_final_[i],peaks_final_[i+1]])
                
                peaks_final_ = [peaks_final_[i] for i in range(len(peaks_final_)) if i-1 not in ind_avg_]
                    
                peaks_final_ = np.array(peaks_final_, dtype=int)        
                ind_avg_ = np.where(distance_neighbors(peaks_final_) < th_dist_)[0]
                # If there are still some other values to average, restart the cycle
                self.peaks_ = peaks_final_
                    
            # There has to be min - max - min - max - min --> Always start and end with min
            peaks_aux = list(self.peaks_)
            for i in range(len(self.peaks)-1):
                ind_del = np.where((self.peaks_>self.peaks[i]) & (self.peaks_<self.peaks[i+1]))[0]
                if len(ind_del) > 1:
                    peaks_aux.remove(max(self.peaks_[ind_del]))
                    
            self.peaks_ = np.array(peaks_aux) # We had to convert to list to apply the remove funcion

            # Add the first value of the signal in case it does not start with a minimum
            if self.peaks_[0] > self.peaks[0]:
                self.peaks_ = np.concatenate((np.array([int(cA[0])]), self.peaks_))

            
            # Peaks analysis
            
            T_plus = []
            T_minus = []
            A_plus = []
            A_minus = []
            for i in range(len(self.peaks)):
                T_plus.append(self.peaks[i]-self.peaks_[i]) 
                T_minus.append(self.peaks_[i+1]-self.peaks[i])
                A_plus.append(abs(cA[self.peaks[i]]-cA[self.peaks_[i]]))
                A_minus.append(abs(cA[self.peaks_[i+1]]-cA[self.peaks[i]]))
            
            T_plus_final.append(np.array(T_plus))
            T_minus_final.append(np.array(T_minus))
            A_plus_final.append(np.array(A_plus))
            A_minus_final.append(np.array(A_minus))
            cycles_final.append(len(self.peaks))

            # Analysis of T+ and T- 
            # 1. According to the cycle number:
            cycles_T_minus = {key: [] for key in np.arange(1,max(cycles_final)+1)}
            cycles_T_plus = {key: [] for key in np.arange(1,max(cycles_final)+1)}
            cycles_A_minus = {key: [] for key in np.arange(1,max(cycles_final)+1)}
            cycles_A_plus = {key: [] for key in np.arange(1,max(cycles_final)+1)}

            for i in range(len(T_minus_final)):
                for j,val in enumerate(T_minus_final[i]):
                    cycles_T_minus[j+1].append(val)
                    cycles_T_plus[j+1].append(T_plus_final[i][j])
                    cycles_A_minus[j+1].append(A_minus_final[i][j])
                    cycles_A_plus[j+1].append(A_plus_final[i][j])
            
            self.cycles_T_minus_plot = [cycles_T_minus[i+1] for i in range(len(cycles_T_minus))]        
            self.cycles_T_plus_plot = [cycles_T_plus[i+1] for i in range(len(cycles_T_plus))] 
            self.cycles_A_minus_plot = [cycles_A_minus[i+1] for i in range(len(cycles_A_minus))]        
            self.cycles_A_plus_plot = [cycles_A_plus[i+1] for i in range(len(cycles_A_plus))] 
            
            self.n_cycles = max(np.unique(cycles_final))

class bulk_peak_analysis():
    def __init__(self, cycles_T_minus_plot, cycles_T_plus_plot, cycles_A_minus_plot, cycles_A_plus_plot):

        # Calculate the median and standard deviation
        self.mean_T_minus_plot = np.zeros(max(cycles))
        self.std_T_minus_plot = np.zeros(max(cycles))
        self.mean_T_plus_plot = np.zeros(max(cycles))
        self.std_T_plus_plot = np.zeros(max(cycles))
        self.mean_A_minus_plot = np.zeros(max(cycles))
        self.std_A_minus_plot = np.zeros(max(cycles))
        self.mean_A_plus_plot = np.zeros(max(cycles))
        self.std_A_plus_plot = np.zeros(max(cycles))

        for i in range(max(cycles)):
            # Auxiliars to calculate the meand and std 
            A_plus = []
            A_minus = []
            T_plus = []
            T_minus = []

            for j in range(len(cycles_A_minus_plot)):
                try:
                    A_plus.append(cycles_A_plus_plot[j][i])
                    A_minus.append(cycles_A_minus_plot[j][i])
                    T_plus.append(cycles_T_plus_plot[j][i])
                    T_minus.append(cycles_T_minus_plot[j][i])
                except IndexError:
                    continue


            self.mean_T_minus_plot[i] = np.nanmedian(T_minus)
            self.std_T_minus_plot[i] = np.nanstd(T_minus)
            self.mean_T_plus_plot[i] = np.nanmedian(T_plus)
            self.std_T_plus_plot[i] = np.nanstd(T_plus)
            self.mean_A_minus_plot[i] = np.nanmedian(A_minus)
            self.std_A_minus_plot[i] = np.nanstd(A_minus)
            self.mean_A_plus_plot[i] = np.nanmedian(A_plus)
            self.std_A_plus_plot[i] = np.nanstd(A_plus)





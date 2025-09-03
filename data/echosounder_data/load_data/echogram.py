import os
import pickle
import numpy as np
import torch
from scipy.signal import convolve2d as conv2d
from . import data_paths

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# from data.echosounder_data.load_data.plotting import setup_matplotlib
# from data.echosounder_data.load_data.normalization import db

from .plotting import setup_matplotlib
from ..preprocessing.normalization import db

class Echogram():
    """ Object to represent a echogram """

    def __init__(self, path):
        self.object_ids_with_label = {}  # Id (index) to echogram with the given label

        # Load meta data
        def load_meta(folder, name):
            with open(os.path.join(folder, name) + '.pkl', 'rb') as f:
                f.seek(0)
                return pickle.load(f, encoding='latin1')

        self.path = path
        self.name = os.path.split(path)[-1]
        self.frequencies  = load_meta(path, 'frequencies').squeeze().astype(int)
        self.frequencies_of_interest = self.frequencies
        self.range_vector = load_meta(path, 'range_vector').squeeze()
        self.time_vector  = load_meta(path, 'time_vector').squeeze()
        self.heave = load_meta(path, 'heave').squeeze()
        self.data_dtype = load_meta(path, 'data_dtype')
        self.label_dtype = load_meta(path, 'label_dtype')
        self.shape = load_meta(path, 'shape')
        self.objects = load_meta(path, 'objects')
        self.n_objects = len(self.objects)
        self.year = int(self.name[9:13])
        self._seabed = None
        self._statistics = None

        self.date = np.datetime64(self.name[9:13] + '-' + self.name[13:15] + '-' + self.name[15:17] + 'T' + self.name[19:21] + ':' + self.name[21:23]) #'yyyy-mm-ddThh:mm'

        #Check which labels that are included
        self.label_types_in_echogram = np.unique([o['fish_type_index'] for o in self.objects])

        #Make dictonary that points to objects with a given label
        for object_id, object in enumerate(self.objects):
            label = object['fish_type_index']
            if label not in self.object_ids_with_label.keys():
                self.object_ids_with_label[label] = []
            self.object_ids_with_label[label].append(object_id)

    def get_regular_time_grid_idx(self, dt=1.0):
        """ Returns indices of time vector that gives a regular time grid by nearest neighbor in time """

        ### Not in use - To be implemented later ###

        # Requires changes to e.g.
        ## label/data memmap/numpy,
        ## get_seabed
        ## echogram.objects

        dt = dt / (60 * 60 * 24)  # Convert from seconds to days
        time_vec = self.time_vector
        start = time_vec[0]
        stop = time_vec[-1]
        regular_time = np.arange(start, stop, dt)
        return [np.argmin(np.abs(time_vec - t)) for t in regular_time]

    def label_memmap(self):
        """ Returns memory map array with labels """

        ### 'labels.dat' replaced by 'labels_heave.dat' - reversing heave corrections (waves making the ship go up and down) ###
        # return np.memmap(self.path + '/labels.dat', dtype=self.label_dtype, mode='r', shape=tuple(self.shape))
        # raw = np.memmap(self.path + '/labels_heave.dat', dtype=self.label_dtype, mode='r', shape=tuple(self.shape))
        # raw = np.array(raw)

        # # # Replace all occurrences of 27 with 2
        # raw[raw == 27] = 2

        # # # Define the values to be replaced with 0
        # values_to_replace = [5027, 6009, 6010, 6007, 9999]
        # # Use numpy's vectorized operations to replace the specified values with 0
        # mask = np.isin(raw, values_to_replace)
        # raw[mask] = -1
        return np.memmap(self.path + '/labels_heave.dat', dtype=self.label_dtype, mode='r', shape=tuple(self.shape))

    def data_memmaps(self, frequencies_of_interest):
        return [np.memmap(self.path + '/data_for_freq_' + str(int(f)) + '.dat', dtype=self.data_dtype, mode='r', shape=tuple(self.shape)) for f in frequencies_of_interest]

    def data_numpy_transformed(self, data_transforms=None, frequencies_of_interest=None):
        # Update frequencies_of_interest attribute if provided
        if frequencies_of_interest is not None:
            self.frequencies_of_interest = frequencies_of_interest

        """ Returns numpy array with data (H x W x C) """
        data = self.data_memmaps(frequencies_of_interest=self.frequencies_of_interest)  # Get memory maps
        data = [np.array(d[:]) for d in data]  # Read memory map into memory
        for d in data:
            d.setflags(write=1)  # Set write permissions to array
        data = [np.expand_dims(d, -1) for d in data]  # Add channel dimension
        data = np.concatenate(data, -1)
        
        # Only apply transforms if they are provided
        if data_transforms is not None:
            if not isinstance(data_transforms, list):
                data_transforms = [data_transforms]
            # Apply each transformation sequentially (skip if transform is None)
            for transform in data_transforms:
                if transform is not None:
                    data = transform(data)
                    
        return data.astype('float32')

    def label_numpy(self):
        """ Returns numpy array with labels (H x W)"""
        label = self.label_memmap()
        label = np.array(label[:])
        label.setflags(write=1)

        # # Replace all occurrences of 27 with 2
        label[label == 27] = 2

        # # Define the values to be replaced with 0
        values_to_replace = [5027, 6009, 6010, 6007, 9999]
        # Use numpy's vectorized operations to replace the specified values with 0
        mask = np.isin(label, values_to_replace)
        label[mask] = -1
        return label

    def get_seabed(self, save_to_file=True, ignore_saved=False):
        """
        Returns seabed approximation line as maximum vertical second order gradient
        :param save_to_file: (bool)
        :param ignore_saved: (bool) If True, this function will re-estimate the seabed even if there exist a saved seabed
        :return:
        """

        if self._seabed is not None and not ignore_saved:
            return self._seabed

        elif os.path.isfile(os.path.join(self.path, 'seabed.npy')) and not ignore_saved:
            self._seabed = np.load(os.path.join(self.path, 'seabed.npy'))
            return self._seabed

        else:

            def set_non_finite_values_to_zero(input):
                input[np.invert(np.isfinite(input))] = 0
                return input

            def seabed_gradient(data):
                gradient_filter_1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
                gradient_filter_2 = np.array([[1, 5, 1], [-2, -10, -2], [1, 5, 1]])
                grad_1 = conv2d(data, gradient_filter_1, mode='same')
                grad_2 = conv2d(data, gradient_filter_2, mode='same')
                return np.multiply(np.heaviside(grad_1, 0), grad_2)

            # Number of pixel rows at top of image (noise) not included when computing the maximal gradient
            n = 10 + int(0.05 * self.shape[0])
            # Vertical shift of seabed approximation line (to give a conservative line)
            a = int(0.004 * self.shape[0])

            data = set_non_finite_values_to_zero(self.data_numpy_transformed())
            seabed = np.zeros((data.shape[1:]))
            for i in range(data.shape[2]):
                seabed[:, i] = -a + n + np.argmax(seabed_gradient(data[:, :, i])[n:, :], axis=0)

            # Repair large jumps in seabed altitude
            repair_threshold = -8

            # Set start/stop for repair interval [i_edge:-i_edge] to avoid repair at edge of echogram
            i_edge = 2

            sb_max = np.max(data[n:, :, :], axis=0)
            sb_max = np.log(1e-10 + sb_max)
            sb_max -= np.mean(sb_max, axis=0)
            sb_max *= 1 / np.std(sb_max, axis=0)

            for f in range(sb_max.shape[1]):

                i = i_edge
                while i < sb_max.shape[0] - i_edge:

                    # Get interval [idx_0, idx_1] where seabed will be repaired for frequency f
                    if sb_max[i, f] < repair_threshold:
                        idx_0 = i
                        while i < sb_max.shape[0]:
                            if sb_max[i, f] < repair_threshold:
                                i += 1
                            else:
                                break
                        idx_1 = i - 1
                        # Replace initial seabed values with mean value before/after repair interval
                        if idx_0 <= i_edge:
                            seabed[idx_0:idx_1 + 1, f] = seabed[idx_1 + 1, f]
                        elif idx_1 >= sb_max.shape[0] - i_edge:
                            seabed[idx_0:idx_1 + 1, f] = seabed[idx_0 - 1, f]
                        else:
                            seabed[idx_0:idx_1 + 1, f] = np.mean(seabed[[idx_0 - 1, idx_1 + 1], f])
                    i += 1

            self._seabed = np.rint(np.median(seabed, axis=1)).astype(int)
            if save_to_file:
                np.save(os.path.join(self.path, 'seabed.npy'), self._seabed)
            return self._seabed

    def get_statistics(self, save_to_file=False, ignore_saved=False):
        # Returns echogram statistics as dict of dicts.
        ## Keys (str): 'min', 'max', 'mean', 'median, 'std', 'count_non_finite_values'
        ## Sub-keys (int): frequencies
        # Warning: non-finite values (nan, inf) are set to zero before calculating each statistic (except 'count_non_finite_values').

        if self._statistics is not None and not ignore_saved:
            return self._statistics

        elif os.path.isfile(os.path.join(self.path, 'statistics.pkl')) and not ignore_saved:
            with open(os.path.join(self.path, 'statistics.pkl'), 'rb') as f:
                f.seek(0)
                self._statistics = pickle.load(f, encoding='latin1')
                return self._statistics

        else:
            statistics = {
                'min': dict(),
                'max': dict(),
                'mean': dict(),
                'median': dict(),
                'std': dict(),
                'count_non_finite_values': dict()
            }

            data = self.data_numpy_transformed()
            freqs = list(self.frequencies)

            for f in self.frequencies:
                statistics['count_non_finite_values'][f] = np.sum(np.invert(np.isfinite(data)))

            data[np.invert(np.isfinite(data))] = 0

            min = np.min(data, axis=(0, 1))
            max = np.max(data, axis=(0, 1))
            mean = np.mean(data, axis=(0, 1), dtype='float64')
            median = np.median(data, axis=(0, 1))
            std = np.std(data, axis=(0, 1), dtype='float64')

            for f in freqs:
                idx = freqs.index(f)
                statistics['min'][f] = min[idx]
                statistics['max'][f] = max[idx]
                statistics['mean'][f] = mean[idx]
                statistics['median'][f] = median[idx]
                statistics['std'][f] = std[idx]

            self._statistics = statistics

            if save_to_file:
                with open(os.path.join(self.path, 'statistics') + '.pkl', 'wb') as file:
                    pickle.dump(self._statistics, file)

            return self._statistics

    def select_mask(self):
        seabed = np.array(self.get_seabed())
        # Create a grid of row indices
        row_indices = np.arange(self.shape[0]).reshape(-1, 1)
        # Create a mask that selects all rows less than the seabed (rows below the seabed will be False)
        mask = row_indices < seabed
        # Set the top 5 rows to False
        mask[:5, :] = False
        return np.asarray(mask, dtype=bool)

    def visualize_raw(self,
                  predictions=None,
                  frequencies=None,
                  draw_seabed=True,
                  show_grid=True,
                  show_name=True,
                  show_freqs=True,
                  show_labels_str=True,
                  return_fig=True,
                  figure=None,
                  filedir='./'):
                #   pred_contrast=1.0,
                # show_predictions_str=True,
                #   labels_original=None,
                #   labels_refined=None,
                #   labels_korona=None,
        """ Visualize echogram, and optionally predictions """

        ### Parameters
        # predictions (2D numpy array): each value is a proxy for probability of fish at given pixel coordinate
        # pred_contrast (positive float): exponent for prediction values to adjust contrast (predictions -> predictions^pred_contrast)
        ###

        # Get data
        if frequencies is None:
            frequencies = self.frequencies_of_interest


        # Tick labels Y
        tick_labels_y = self.range_vector
        tick_labels_y = tick_labels_y - np.min(tick_labels_y)
        tick_idx_y = np.arange(start=0, stop=len(tick_labels_y), step=int(len(tick_labels_y) / 4))

        # Format settings
        color_seabed = {'seabed': 'white'}
        lw = {'seabed': 1}
        cmap_labels = mcolors.ListedColormap(['yellow', 'black', 'red', 'green'])
        boundaries_labels = [-200, -0.5, 0.5, 1.5, 2.5]
        norm_labels = mcolors.BoundaryNorm(boundaries_labels, cmap_labels.N, clip=True)

        global_title = ''
        if show_name:
            global_title += self.name + ' '

        for j, (label, data, seabed, time_vector, mask) in enumerate(self.generate_echogram()):
            # Initialize plot
            plt = setup_matplotlib()
            if figure is not None:
                plt.clf()
            plt.tight_layout()

            # Tick labels X
            tick_labels_x = time_vector * 24 * 60
            tick_labels_x = tick_labels_x - np.min(tick_labels_x)
            tick_idx_x = np.arange(start=0, stop=len(tick_labels_x), step=int(len(tick_labels_x) / 6))

            # Number of subplots
            n_plts = data.shape[2] + 2 # per channel + selected mask + label
            if predictions is not None:
                if type(predictions) is np.ndarray:
                    n_plts += 1
                elif type(predictions) is list:
                    n_plts += len(predictions)

            # Channels
            plt.figure(figsize=(label.shape[1]//50, 
                                label.shape[0]//50*(len(frequencies)+2+(sum(1 for var in [predictions] if var is not None)))))
            major_font_size = label.shape[1]//250
            minor_font_size = major_font_size//2

            for i in range(data.shape[2]):
                if i == 0:
                    main_ax = plt.subplot(n_plts, 1, i + 1)
                    plt.suptitle(global_title, fontsize=major_font_size)
                else:
                    plt.subplot(n_plts, 1, i + 1, sharex = main_ax, sharey = main_ax)
                if show_freqs:
                    plt.text(0.5, 0.9, str(frequencies[i]) + ' kHz', fontsize=major_font_size, ha='center', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
                plt.imshow(data[:, :, i], cmap='jet', aspect='auto')

                # Hide grid
                if not show_grid:
                    plt.axis('off')
                else:
                    plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=minor_font_size)
                    plt.xticks(tick_idx_x, [int(tick_labels_x[j]) for j in tick_idx_x], fontsize=minor_font_size)
                    plt.ylabel("Depth\n[meters]", fontsize=minor_font_size)
                if draw_seabed:
                    plt.plot(np.arange(data.shape[1]), seabed, c=color_seabed['seabed'], lw=lw['seabed'])

            # Labels
            i += 1
            plt.subplot(n_plts, 1, i + 1, sharex = main_ax, sharey = main_ax)
            #plt.imshow(labels != 0, cmap='viridis', aspect='auto')
            plt.imshow(label, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            if show_labels_str:
                plt.title("Annotations", fontsize=major_font_size)
            if draw_seabed:
                plt.plot(np.arange(data.shape[1]), seabed, c=color_seabed['seabed'], lw=lw['seabed'])

            # Hide grid
            if not show_grid:
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=minor_font_size)
                plt.xticks(tick_idx_x, [int(tick_labels_x[j]) for j in tick_idx_x], fontsize=minor_font_size)
                plt.ylabel("Depth\n[meters]", fontsize=minor_font_size)

            # selected mask
            i += 1
            plt.subplot(n_plts, 1, i + 1, sharex = main_ax, sharey = main_ax)
            #plt.imshow(labels != 0, cmap='viridis', aspect='auto')
            plt.imshow(mask, aspect='auto', cmap=cmap_labels, norm=norm_labels)
            if show_labels_str:
                plt.title("Annotations", fontsize=major_font_size)
            if draw_seabed:
                plt.plot(np.arange(data.shape[1]), seabed, c=color_seabed['seabed'], lw=lw['seabed'])

            # Hide grid
            if not show_grid:
                plt.axis('off')
            else:
                plt.yticks(tick_idx_y, [int(tick_labels_y[j]) for j in tick_idx_y], fontsize=minor_font_size)
                plt.xticks(tick_idx_x, [int(tick_labels_x[j]) for j in tick_idx_x], fontsize=minor_font_size)
                plt.ylabel("Depth\n[meters]", fontsize=minor_font_size)

            plt.xlabel("Time [minutes]", fontsize=minor_font_size)
            plt.tight_layout()

            if return_fig:
                plt.savefig(os.path.join(filedir, '%s_%d.png' % (self.name, j)))
                pass
            else:
                plt.show()
              
def get_echograms(years='all', frequencies=None, minimum_shape=224):
    """ Returns all the echograms for a given year that contain the given frequencies"""

    path_to_echograms = data_paths.path_to_echograms()
    raw_eg_names = os.listdir(path_to_echograms)
    eg_names = []
    for name in raw_eg_names:
        try:
            if  '.' not in name:
                if os.path.isfile(os.path.join(path_to_echograms, name, 'labels_heave.dat')):
                    eg_names.append(name)
        except FileNotFoundError:
                    # Handle the case where the file or directory does not exist
            print(f"File not found: {os.path.join(path_to_echograms, name, 'labels_heave.dat')}")

    echograms = [Echogram(os.path.join(path_to_echograms, e)) for e in eg_names]

    # Filter echograms based on multiple conditions
    echograms = [
        e for e in echograms
        if (e.shape[0] > minimum_shape) and
        (e.shape[1] > minimum_shape) and
        (e.shape[1] == e.time_vector.shape[0]) and
        (e.shape[1] == e.heave.shape[0])]
    
    # and not np.isnan(e.data_numpy_transformed()).any()]

    if years != 'all':
        # Ensure years is iterable
        if type(years) not in [list, tuple, np.ndarray]:
            years = [years]

        # Filter on years
        echograms = [e for e in echograms if e.year in years]
    return echograms
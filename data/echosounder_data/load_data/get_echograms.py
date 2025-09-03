# from data.echosounder_data.load_data import data_paths  
# from data.echosounder_data.load_data.echogram import Echogram 
from . import data_paths
from .echogram import Echogram

import os
from collections import defaultdict
import numpy as np

# Get echograms from the data directory: see data_paths
def get_echograms(years, tuple_frequencies, minimum_shape):
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
           (e.shape[1] == e.heave.shape[0]) and
           (tuple(e.frequencies) == tuple_frequencies)
    ]

    if years != 'all':
        # Ensure years is iterable
        if type(years) not in [list, tuple, np.ndarray]:
            years = [years]

        # Filter on years
        echograms = [e for e in echograms if e.year in years]
    return echograms



# Count the number of classes in the echograms
def count_classes_in_echograms(echograms):
    total_class_counts = defaultdict(int)

    for e in echograms:
        labels = e.label_memmap()
        unique_classes, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            total_class_counts[cls] += count

    return total_class_counts

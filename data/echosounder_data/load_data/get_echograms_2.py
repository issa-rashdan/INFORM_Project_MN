import sys
sys.path.append("/INFORM_Project_MN")  # Adjust the path as needed
from data.echosounder_data.load_data import data_paths  
from data.echosounder_data.load_data.echogram import Echogram 


import os
from collections import defaultdict
import numpy as np
import random

# Get echograms from the data directory: see data_paths
def get_echograms(years, tuple_frequencies, minimum_shape):
    """ Returns all the echograms for a given year that contain the given frequencies"""

    path_to_echograms = data_paths.path_to_echograms()
    raw_eg_names = os.listdir(path_to_echograms)
    all_echograms = defaultdict(list)
    eg_names = []
    final_echograms = []
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
    for e in echograms:
        all_echograms[e.year].append(e)

    if years != 'all':
        # Ensure years is iterable
        if not isinstance(years, (list,tuple, np.ndarray)):
            years = [years]
            
        for year in years:
            if year in all_echograms:
                samples = random.sample(all_echograms[year], 
                                    min(10, len(all_echograms[year])))
                print(f"Selected {len(samples)} echograms from year {year}")
                # Extend the final list with the selected echograms from this year      
                final_echograms.extend(samples)
            else:
                print(f"No echograms found for year {year}")
    else:
            #Group echograms by year
            #Select 10 random echograms from each year
            for year in all_echograms:
                year_samples = random.sample(all_echograms[year], 
                                                min(10, len(all_echograms[year])))
                print(f"Selected {len(year_samples)} echograms from year {year}")
                # Extend the final list with the selected echograms from this year
                final_echograms.extend(year_samples)
            
    return final_echograms    


# Count the number of classes in the echograms
def count_classes_in_echograms(echograms):
    total_class_counts = defaultdict(int)

    for e in echograms:
        labels = e.label_memmap()
        unique_classes, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            total_class_counts[cls] += count

    return total_class_counts

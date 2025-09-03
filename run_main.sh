#!/bin/sh
# Unset the malloc stack logging flag so Python won’t attempt to disable it
# unset MallocStackLogging

# Run your main script with faulthandler enabled
python3 -X faulthandler main.py 2> crash.log


"""
After holidays
check autodencoder forward pass. I think we can deal something with channel permutation
# try with a smaller datasamples per year but random.
# Couple with the pretrained ViT.
"""
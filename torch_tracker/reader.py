import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Reader:
    """
    Read quantities from multiple analysis HDF5 files into dictionary
    for easy data manipulation by user.

    Example usage:
        f = Reader(
            files=["inter_MC.h5","out_LC.h5","out_MC.h5"]
            labels=['innMC','outLC','outMC']
        )
        times = f.data['inter_MC']['time']
        sfe = f.data['inter_MC']['sfe']
    """

    def __init__(self, files, labels=None):
        self.files = files
        if labels is None or len(labels) != len(files):
            if labels is not None:
                print('Warning! Labels do not match files shape, defaulting to filenames.')
            labels = [os.path.splitext(os.path.basename(f))[0] for f in files]

        # Load all datasets
        self.data = {}
        for i,f in enumerate(files):
            with h5py.File(f, "r") as h:
                snap = h["snapshots/snapshot"][:]
                time = h["snapshots/time"][:]
                quantities = {q: h[f"quantities/{q}"][:] for q in h["quantities"]}
                self.data[labels[i]] = {"snapshot": snap, "time": time}
                self.data[labels[i]].update(quantities)

import os
import h5py
import numpy as np
from glob import glob
import yt

from .quantities import QUANTITY_REGISTRY, QUANTITY_TYPE, QUANTITY_REQUISITES

class TorchAnalysis:
    """
    Analyze a simulation and store results in an HDF5 file.
    """

    def __init__(self, sim_name="turbsph", data_dir="data", analysis_file="analysis.h5", quantities=None):
        self.sim_name = sim_name
        self.data_dir = data_dir
        self.analysis_file = analysis_file
        self.quantities = quantities or list(QUANTITY_REGISTRY.keys())

        # Open HDF5 file (create if not exists)
        self.h5 = h5py.File(self.analysis_file, "a")
        if "snapshots" not in self.h5:
            self.snap_grp = self.h5.create_group("snapshots")
            self.snap_grp.create_dataset("snapshot", data=[], maxshape=(None,), dtype=int)
            self.snap_grp.create_dataset("time", data=[], maxshape=(None,), dtype=float)
        else:
            self.snap_grp = self.h5["snapshots"]

        if "quantities" not in self.h5:
            self.qgrp = self.h5.create_group("quantities")
            for q in self.quantities:
                if QUANTITY_TYPE[q] == 'scalar':
                    self.qgrp.create_dataset(q, data=[], maxshape=(None,), dtype=float)
                elif QUANTITY_TYPE[q] == 'vector':
                    self.qgrp.create_dataset(q, shape=(0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.float64))
        else:
            self.qgrp = self.h5["quantities"]

        # Meta group to track last snapshot
        if "meta" not in self.h5:
            self.meta = self.h5.create_group("meta")
        else:
            self.meta = self.h5["meta"]

    # -----------------------------
    # Close the underlying HDF5 file
    # -----------------------------
    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -----------------------------
    # Utility: find available snapshots
    # -----------------------------
    def find_snapshots(self):
        pattern = os.path.join(self.data_dir, f"{self.sim_name}_hdf5_plt_cnt_*")
        files = sorted(glob(pattern))
        snaps = [int(os.path.basename(f).split("_")[-1]) for f in files]
        return snaps

    # -----------------------------
    # Clear all data from quantity for recalculation
    # -----------------------------
    def clear(self, quantities):
        if isinstance(quantities, str):
            quantities = [quantities]
        for q in quantities:
            if q in self.quantities:
                print(f"Clearing quantity {q} for recalculation")
                if QUANTITY_TYPE[q] == 'scalar':
                    self.qgrp[q][:] = np.nan
                if QUANTITY_TYPE[q] == 'vector':
                    # must nuke variable length datasets
                    self.qgrp[q][:] = []

    # -----------------------------
    # Update snapshots
    # -----------------------------
    def update(self, start_snapshot=None, last_snapshot=None, step=1):
        if step < 1:
            raise ValueError("step must be >= 1")

        # Gather available snapshots
        available = self.find_snapshots()
        if not available:
            print("No snapshots found!")
            return

        # Determine starting snapshot.
        # With no explicit start, scan *all* available snapshots so a newly added
        # quantity gets back-filled for older snapshots. The per-snapshot
        # missing-quantity check below skips snapshots that are already complete, so
        # only snapshots with missing quantities actually trigger a yt load.
        if start_snapshot is not None:
            start_snap = start_snapshot
        else:
            start_snap = min(available)

        snaps_to_process = [s for s in available if s >= start_snap]
        if last_snapshot is not None:
            snaps_to_process = [s for s in snaps_to_process if s <= last_snapshot]

        # Apply step
        snaps_to_process = snaps_to_process[::step]

        if not snaps_to_process:
            print("No snapshots to process in the specified range/step.")
            return

        # Work out which quantities are missing for each snapshot up front. Each
        # quantity dataset is read exactly once here (rather than re-reading the
        # whole dataset once per snapshot), then the loop below just looks up the
        # result. This is safe because processing one snapshot never changes the
        # missing-status of another: each snapshot occupies its own row.
        existing_snaps = np.array(self.snap_grp["snapshot"][:])
        n_snap = len(existing_snaps)
        snap_to_idx = {int(s): i for i, s in enumerate(existing_snaps)}

        missing_by_snap = {snap: [] for snap in snaps_to_process}
        for q in self.quantities:
            if q not in self.qgrp:
                # Create the dataset, back-filled to n_snap rows so it stays
                # index-aligned with the existing snapshots, then mark it missing
                # for every snapshot we are processing.
                if QUANTITY_TYPE[q] == 'scalar':
                    self.qgrp.create_dataset(q, data=np.full(n_snap, np.nan),
                                             maxshape=(None,), dtype=float)
                elif QUANTITY_TYPE[q] == 'vector':
                    # Each row auto-initialises to an empty array. Creating with
                    # shape=(0,) would raise IndexError when writing an already-
                    # existing snapshot.
                    self.qgrp.create_dataset(q, shape=(n_snap,), maxshape=(None,),
                                             dtype=h5py.vlen_dtype(np.float64))
                for snap in snaps_to_process:
                    missing_by_snap[snap].append(q)
                continue

            qdata = self.qgrp[q][:]  # one read per quantity
            for snap in snaps_to_process:
                idx = snap_to_idx.get(snap)
                if idx is None:
                    # snapshot not yet in the file: every quantity is missing
                    missing_by_snap[snap].append(q)
                elif QUANTITY_TYPE[q] == 'scalar' and np.isnan(qdata[idx]):
                    missing_by_snap[snap].append(q)
                # vector entries are variable-length arrays; an uncomputed/cleared
                # entry is empty (np.isnan is invalid on the stored numeric arrays)
                elif QUANTITY_TYPE[q] == 'vector' and len(qdata[idx]) == 0:
                    missing_by_snap[snap].append(q)

        # Process each snapshot that has missing quantities.
        for snap in sorted(snaps_to_process):
            missing_quantities = missing_by_snap[snap]
            if not missing_quantities:
                continue  # all quantities already exist for this snapshot

            print(f"Processing snapshot {snap:04d} for quantities: {missing_quantities}")
            self._process_snapshot(snap, quantities_to_compute=missing_quantities)
            self.meta.attrs["last_snapshot"] = snap
            self.h5.flush()

    # -----------------------------
    # Process a single snapshot
    # -----------------------------
    def _process_snapshot(self, snap, quantities_to_compute=None):
        quantities_to_compute = quantities_to_compute or self.quantities

        filename = os.path.join(self.data_dir, f"{self.sim_name}_hdf5_plt_cnt_{snap:04d}")
        ds = yt.load(filename)
        current_time = ds.current_time.to("Myr").value

        results = {}
        for q in quantities_to_compute:
            func = QUANTITY_REGISTRY[q]
            prev_values = QUANTITY_REQUISITES.get(q, None)
            try:
                if prev_values:
                    val = func(ds, self._get_prev_values(snap, prev_values))
                else:
                    val = func(ds)
            except Exception as e:
                print(f"Warning: quantity {q} failed at snapshot {snap}: {e}")
                # Use a type-appropriate "missing" sentinel: NaN for scalars,
                # an empty array for vectors (writing scalar NaN into a
                # variable-length row is invalid and crashes h5py).
                val = np.array([]) if QUANTITY_TYPE[q] == 'vector' else np.nan
            results[q] = val

        # Write results (overwrite or append)
        self._write_snapshot(snap, current_time, results)

    
    # -----------------------------
    # Helper function that gets the previous snapshot quantities available in the analysis h5 file, i.e., for computing SFR
    # -----------------------------
    def _get_prev_values(self, snap, prev_values):
        if len(self.snap_grp["snapshot"]) == 0:
            # exit if this is the first snapshot -- no previous values
            return None

        prev_quants = []
        for pv in prev_values:
            if pv == "time" or pv == "snap":
                prev_quants.append(self.snap_grp[pv][-1])
            elif pv in self.quantities:
                prev_quants.append(self.qgrp[pv][-1])
            else:
                print(f"Warning: previous quantity {pv} unavailable at snapshot {snap}")
                prev_quants.append(np.nan)

        return prev_quants

    # -----------------------------
    # Write snapshot to HDF5 (overwrite if exists)
    # -----------------------------
    def _write_snapshot(self, snap, time_val, quantities_dict):
        snap_ds = self.snap_grp["snapshot"]
        time_ds = self.snap_grp["time"]

        existing = np.array(snap_ds[:])
        if snap in existing:
            idx = np.where(existing == snap)[0][0]
            snap_ds[idx] = snap
            time_ds[idx] = time_val
            for q, val in quantities_dict.items():

                self.qgrp[q][idx] = val
        else:
            self._append_snapshot(snap, time_val, quantities_dict)

    # -----------------------------
    # Append snapshot
    # -----------------------------
    def _append_snapshot(self, snap, time_val, quantities_dict):
        for dsname, val in [("snapshot", snap), ("time", time_val)] + list(quantities_dict.items()):
            if dsname == "snapshot":
                ds = self.snap_grp["snapshot"]
            elif dsname == "time":
                ds = self.snap_grp["time"]
            else:
                ds = self.qgrp[dsname]
            ds.resize(ds.shape[0]+1, axis=0)
            ds[-1] = val


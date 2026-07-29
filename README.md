# torch\_analysis

Analysis tool for processing and storing data from Torch star cluster formation simulations. 

## Installation
```
git clone https://github.com/torch-sf/torch-analysis.git
cd torch-analysis
pip install -e .
```

## Usage

Keep a file called ```analysis.py``` in your run directory. Every time you get new simulation
outputs, execute ```python analysis.py```. It will automatically process new snapshots.
It should look something like this:

```
from torch_tracker import TorchAnalysis

tracker = TorchAnalysis(
    data_dir="data",
    sim_name="turbsph",
    analysis_file="M7.h5",
    quantities=[
        "gas_mass",
        "stellar_mass",
        "sfe",
        "bound_gas_mass_fraction",
        "half_mass_radius",
        "sfr"
    ]
)

tracker.update()
```

If you want to add a new quantity later, simply add it to the list. The next update will process all snapshots for that quantity. If you want to add your own quantities to track, add them to ```torch_tracker/quantities.py```. The ```update()``` function also allows you to specify a beginning, end, and step size for processing snapshots with ```start_snapshot,last_snapshot,step```.

If you change a quantity calculation and need to recalculate all values for that quantity in your analysis file, simply call ```tracker.clear(quantities)``` where the input is a list of the quantities you want to reset. Then call update as usual. Just make sure to get rid of this line the next time you call your analysis script! Note that some calculated quantities, like number of feedback stars, need information from your `flash.par` file. For that reason, be sure to activate your torch python environment before running the `torch_tracker` script. 

## Reading data

To read the data into a dictionary for easy plotting:
```
from torch_tracker import Reader
import matplotlib.pyplot as plt

files = [
    "../../innMC/inter_MC.h5",
    "../../outLC/outer_LC.h5",
    "../../outMC/outer_MC.h5"
]
labels = [
    "innMC",
    "outLC",
    "outMC"
]

r = Reader(files=files, labels=labels)

for sim,l in enumerate(labels):
    t = r.data[l]['time']
    # sometimes the hdf5 data can be out of order if you processed
    # files out of order.
    order = np.argsort(t)
    sfe = r.data[l]['sfe_roi']
    plt.plot(t[order],sfe[order],label=l)
plt.legend()
plt.show()
```

## Adding new quantities to track

Every tracked quantity is just a Python function in `torch_tracker/quantities.py` plus a few
lines that register it in the module-level dictionaries at the top of that file:

```python
QUANTITY_REGISTRY   = {}   # name -> function that computes the quantity
QUANTITY_TYPE       = {}   # name -> 'scalar' or 'vector'   (how it is stored in HDF5)
QUANTITY_LABELS     = {}   # name -> LaTeX string for plotting
QUANTITY_REQUISITES = {}   # name -> list of previous-snapshot values the function needs
```

When you run `tracker.update()`, the tracker loads each snapshot with `yt.load(...)`, looks up
your function in `QUANTITY_REGISTRY`, calls it with the yt dataset, and writes the returned value
into the analysis HDF5 file under `quantities/<name>`. Adding a quantity is therefore two steps:
**write the function**, then **register it**.

### Step 1 — Write the calculation

Add a function that takes the yt dataset `ds` as its only argument and returns a single value. You can make helper functions that take other things, but the actual quantity function must only accept a yt dataset. 
Keep the actual physics/reduction inside the function; return a plain Python/NumPy scalar (for a
`scalar` quantity) or a 1-D array of ints (for a `vector` quantity).

```python
def mean_gas_temperature(ds):
    """Mean gas temperature of the whole domain, in Kelvin."""
    ad = ds.all_data()
    temp = ad[("gas", "temperature")].to("K")
    return float(temp.mean().value)
```

A few conventions used throughout the existing quantities that you should follow:

- **Guard against snapshots with no particles.** Early snapshots often have no stars/sinks yet.
  Any quantity that reads particle fields should bail out first:
  ```python
  def my_particle_quantity(ds):
      if not ds.particles_exist:
          return 0.0 # or np.nan depending on desired plotting behavior
      ad = ds.all_data()
      ...
  ```
- **Select stars vs. sinks with the `particle_csgm` marker.** Stars have `particle_csgm == 0`;
  sinks have `particle_csgm != 0`. The helper `particle_mass_container(ds, particle_type=...)`
  already encapsulates this if you only need masses.
- **Attach units and strip them at the end.** yt fields carry units; convert explicitly
  (`.to("Msun")`, `.to("pc")`) and return the bare `.value` / `.v` so the HDF5 file stores a
  clean float.
- **Reuse the `*_container` helpers for domain vs. ROI variants.** Several quantities are computed
  both over the whole box and over the derefinement region of interest (ROI). If you write your
  core logic to take a yt data *container*, you can expose both cheaply:
  ```python
  def my_quantity(ds):
      return _my_quantity_container(ds.all_data())

  def my_quantity_roi(ds):
      return _my_quantity_container(get_roi_region(ds))   # get_roi_region reads flash.par
  ```
- If quantities are repeating calculations, create an internal function (prefixed with _) that 
  the quantity calculations can reuse. 

You do **not** need to handle exceptions yourself. If your function raises, the tracker catches it,
prints a warning, and stores `NaN` for that snapshot (see `_process_snapshot` in `tracker.py`), so
a single bad snapshot never aborts the whole run.

### Step 2 — Register the quantity

Immediately below the function, register it. `QUANTITY_REGISTRY` and `QUANTITY_TYPE` are
**required**; `QUANTITY_LABELS` is optional but strongly recommended (it is the axis label used
when plotting); `QUANTITY_REQUISITES` is only needed for time-derivative-style quantities (see
below).

```python
QUANTITY_REGISTRY["mean_gas_temperature"] = mean_gas_temperature
QUANTITY_TYPE["mean_gas_temperature"]     = 'scalar'
QUANTITY_LABELS["mean_gas_temperature"]   = r"$\langle T_\mathrm{gas}\rangle\,[\mathrm{K}]$"
```

Notes on each field:

- **`QUANTITY_TYPE` must be `'scalar'` or `'vector'`.** This controls the HDF5 dataset that gets
  created. `'scalar'` stores one float per snapshot; `'vector'` stores a variable-length array of
  `int32` per snapshot (used for things like `unbound_star_ids`, where each snapshot yields a
  different-length list of particle IDs). 
- **`QUANTITY_LABELS`** takes a raw LaTeX string (`r"..."`). It is purely for plotting and is safe
  to omit, but tools that read labels will fall back to the raw name without it.

### Optional — Quantities that need the previous snapshot

Some quantities are defined between snapshots (e.g. a star-formation *rate* is `dM/dt`). For these,
declare what earlier values you need in `QUANTITY_REQUISITES`, and give your function a **second
argument** `prev_values`. The tracker will look up the requested values from the most recently
processed snapshot and pass them in; on the very first snapshot it passes `None`, which you must
handle. `sfr` is the reference example:

```python
def sfr(ds, prev_values):
    if prev_values is None:        # first snapshot: no previous value to diff against
        return np.nan
    prev_time, prev_star_mass = prev_values
    sm = stellar_mass(ds)
    dm = sm - prev_star_mass
    dt = (ds.current_time.in_units('Myr').v - prev_time) * 1e6   # Myr -> yr
    return max(dm / dt, 0.0)

QUANTITY_REGISTRY["sfr"]   = sfr
QUANTITY_TYPE["sfr"]       = 'scalar'
QUANTITY_LABELS["sfr"]     = r"$\rm{SFR}\,[\rm{M_\odot~yr^{-1}}]$"
QUANTITY_REQUISITES["sfr"] = ["time", "stellar_mass"]
```

The entries in the requisites list are resolved in order and can be:

- `"time"` or `"snap"` — the previous snapshot's time / snapshot number (read from the `snapshots`
  group), or
- the **name of another tracked quantity** — its previous value is read from the analysis file.

Because requisites are read from the analysis file, **any quantity you list must also be tracked**
(i.e. present in the `quantities=[...]` list in your `analysis.py`) and ideally processed before the
one that depends on it. In the `sfr` example, `stellar_mass` must be in your quantities list.

### Step 3 — Start tracking it

Registering a quantity does not by itself compute anything. Add its name to the `quantities` list
in your `analysis.py` and run the script again:

```python
tracker = TorchAnalysis(
    data_dir="data",
    sim_name="turbsph",
    analysis_file="M7.h5",
    quantities=[
        "gas_mass",
        "stellar_mass",
        "mean_gas_temperature",   # <-- your new quantity
    ]
)
tracker.update()
```

On the next `update()`, the tracker creates the new HDF5 dataset and back-fills it for every
snapshot that is missing a value, so previously processed snapshots are updated automatically —
you do not have to reprocess from scratch.

### Recalculating after you change a formula

If you edit the body of an existing quantity, already-stored values are **not** recomputed
automatically (the tracker only fills in snapshots whose value is `NaN`). To force a recompute,
clear it once and update:

```python
tracker.clear("mean_gas_temperature")   # or a list: tracker.clear(["sfe", "sfr"])
tracker.update()
```

Remember to delete the `clear(...)` line afterward so you don't wipe and recompute on every run.

### Checklist

1. Function in `quantities.py` taking only `ds` (and `prev_values` only if it needs history).
2. `QUANTITY_REGISTRY[name] = func` and `QUANTITY_TYPE[name] = 'scalar'|'vector'`.
3. `QUANTITY_LABELS[name] = r"..."` for nice plot labels (optional).
4. `QUANTITY_REQUISITES[name] = [...]` only if the function takes `prev_values` (optional).
5. Add `name` to the `quantities=[...]` list in `analysis.py` and rerun `python analysis.py`.

> **Environment note:** quantities that read from `flash.par` (via `get_roi_region`) or from Torch
> user parameters (e.g. `number_feedback_stars`, which imports `torch_user`/`amuse`) require your
> Torch Python environment to be active when you run the analysis script.


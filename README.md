# torch\_analysis

Analysis tool for processing and storing data from Torch star cluster formation simulations. 

### Insallation
```
git clone https://github.com/brookepolak/torch_analysis.git
cd torch_analysis
pip install -e .
```

### Example usage

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

If you change a quantity calculation and need to recalculate all values for that quantity in your analysis file, simply call ```tracker.clear(quantities)``` where the input is a list of the quantities you want to reset. Then call update as usual. Just make sure to get rid of this line the next time you call your analysis script!

# Reading data

To read the data into a dictionary for easy plotting:
```
from torch_tracker import Reader
from matplotlib.pyplot import plt

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


To make rudimentary plots of the quantities, do:

```
rom torch_tracker import Plotter
from torch_tracker.quantities import QUANTITY_LABELS
import cmasher as cmr

p = Plotter("M7.h5", quantity_labels=QUANTITY_LABELS)

p.plot_multiple(["gas_mass_roi","stellar_mass_roi"], labels=[r'M$_{\rm gas}$',r'M$_\star$'], ylabel=r'Mass [M$_\odot$]', cmap=cmr.voltage, savename='mass_roi.png')
p.plot("sfe")
p.plot("gas_mass")
p.plot("bound_gas_mass_fraction")
p.plot("half_mass_radius", ylog=False)
p.plot("sfr")
```



from torch_tracker import TorchAnalysis

tracker = TorchAnalysis(
    data_dir="data",
    sim_name="turbsph",
    analysis_file="analysis.h5",
    quantities=[
        "gas_mass",
        "stellar_mass",
        "sfe",
        "gas_mass_roi",
        "sfe_roi",
    ]
)

tracker.update()
tracker.close()


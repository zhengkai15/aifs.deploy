
# 🌦️ AIFS Weather Forecasting Pipeline

> **Hugging Face model:** [ecmwf/aifs-single-1.0](https://huggingface.co/ecmwf/aifs-single-1.0)

This repository provides a **Hugging Face–style inference and visualization pipeline** for the ECMWF **AIFS (Artificial Intelligence Forecasting System)** weather model.
It automates the process of:

* Fetching initialization data from **ECMWF Open Data**
* Preprocessing and regridding with **earthkit**
* Running AI forecasts using the **Anemoi Inference framework**
* Saving outputs to NetCDF files
* (Optional) Visualizing weather fields such as winds and temperature


## 🚀 Features

* ✅ Automatic initialization from ECMWF Open Data
* ✅ Compatible with [`ecmwf/aifs-single-1.0`](https://huggingface.co/ecmwf/aifs-single-1.0) checkpoints
* ✅ Multi-step AI forecast via [Anemoi inference runners](https://github.com/ecmwf/anemoi-inference)
* ✅ Output saving to `.nc` (NetCDF) format
* ✅ Optional map visualization with Cartopy


## 📦 Dependencies

```bash
pip install earthkit-data earthkit-regrid xarray matplotlib cartopy anemoi-inference ecmwf-opendata
```


## ⚙️ Usage

```python
from aifs_pipeline import WeatherConfig, AIFSPipeline, main

# Initialize config (edit paths as needed)
config = WeatherConfig(
    checkpoint_path="/path/to/aifs-single-mse-1.0.ckpt",
    output_nc_base_path="./outputs"
)

# Run full pipeline
main(config)
```

This will:

1. Download the latest ECMWF open data
2. Initialize the model state
3. Run forecasts up to 360 hours ahead
4. Save results as `.nc` files under `./outputs`


## 📊 Example Output

NetCDF files (e.g., `006.nc`, `012.nc`, ...) will be generated in:

```
outputs/
```

Each file contains model variables such as:

* `2t` (2-meter temperature)
* `10u`, `10v` (10-meter winds)
* `msl` (mean sea-level pressure)
* `z_500` (geopotential at 500 hPa), etc.

---

## 🖼️ Visualization Example

```python
from aifs_pipeline import Visualizer, WeatherState

# Assuming 'state' is a WeatherState object
Visualizer(config).plot_state(state, save_path="wind_map.png", field_name="10u")
```


## 📚 Reference

* 🌐 [ECMWF AIFS Model](https://huggingface.co/ecmwf/aifs-single-1.0)
* 📦 [Anemoi Inference Library](https://github.com/ecmwf/anemoi-inference)
* 🌍 [ECMWF Open Data Access](https://www.ecmwf.int/en/forecasts/datasets/open-data)


## 📝 License

This project follows the same license as the [ECMWF AIFS model](https://huggingface.co/ecmwf/aifs-single-1.0).


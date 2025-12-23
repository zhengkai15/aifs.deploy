"""
AIFS Weather Forecasting Pipeline - Hugging Face Style Architecture
"""

import datetime
from loguru import logger
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri

import earthkit.data as ekd
import earthkit.regrid as ekr

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from ecmwf.opendata import Client as OpendataClient




@dataclass
class WeatherConfig:
    """Configuration for AIFS weather forecasting."""
    
    # Surface parameters
    surface_params: List[str] = field(default_factory=lambda: [
        "10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"
    ])
    
    # Soil parameters
    soil_params: List[str] = field(default_factory=lambda: ["vsw", "sot"])
    
    # Pressure level parameters
    pressure_level_params: List[str] = field(default_factory=lambda: ["gh", "t", "u", "v", "w", "q"])
    
    # Pressure levels in hPa
    pressure_levels: List[int] = field(default_factory=lambda: [
        1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
    ])
    
    # Soil levels
    soil_levels: List[int] = field(default_factory=lambda: [1, 2])
    
    # Model checkpoint
    checkpoint_path: str = "/home/sky/.docker/git/aifs.deploy/aifs-single-mse-1.0.ckpt" # /path/to/aifs-single-mse-1.0.ckpt
    
    # Output paths
    output_nc_base_path: str = "/mnt/sdb/aifs" # 
    
    # Forecast settings
    lead_time_hours: int = 360
    time_step_hours: int = 6
    device: str = "cuda"
    
    # Data interpolation
    input_resolution: Tuple[float, float] = (0.25, 0.25)
    output_grid: str = "N320"


@dataclass
class WeatherState:
    """Container for weather data state."""
    
    date: datetime.datetime
    fields: Dict[str, np.ndarray]
    latitudes: Optional[np.ndarray] = None
    longitudes: Optional[np.ndarray] = None
    
    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Safely retrieve a field."""
        logger.info(f"get_field called for: {field_name}")
        return self.fields.get(field_name)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        logger.info("Converting WeatherState to dict.")
        result = {
            "date": self.date,
            "fields": self.fields,
        }
        if self.latitudes is not None:
            result["latitudes"] = self.latitudes
        if self.longitudes is not None:
            result["longitudes"] = self.longitudes
        return result


class DataProvider:
    """Handles data fetching from ECMWF Open Data."""
    
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.client = OpendataClient()
        logger.info("DataProvider initialized.")
    
    def get_latest_date(self) -> datetime.datetime:
        """Fetch the latest available date."""
        logger.info("Fetching latest available date from ECMWF open data.")
        return self.client.latest()
    
    def fetch_parameters(
        self, 
        date: datetime.datetime, 
        param_list: List[str], 
        levelist: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """Fetch weather parameters from open data."""
        logger.info(f"Fetching parameters: {param_list}, levelist: {levelist}, date: {date}")
        fields = defaultdict(list)
        levelist = levelist or []
        
        for fetch_date in [date - datetime.timedelta(hours=6), date]:
            logger.info(f"Fetching {param_list} for {fetch_date}")
            
            data = ekd.from_source(
                "ecmwf-open-data",
                date=fetch_date,
                param=param_list,
                levelist=levelist
            )
            
            for field in data:
                self._process_field(field, fields, levelist)
        
        logger.info(f"Fetched parameters, stacking arrays. Field list: {list(fields.keys())}")
        # Stack fields into single arrays
        return {name: np.stack(values) for name, values in fields.items()}
    
    def _process_field(
        self, 
        field, 
        fields: Dict, 
        levelist: List[int]
    ) -> None:
        """Process and interpolate a single field."""
        logger.info(f"Processing field: {field.metadata('param')}, levelist: {levelist}")
        assert field.to_numpy().shape == (721, 1440), "Unexpected field shape"
        
        # Shift from -180,180 to 0,360
        logger.info("Rolling field array from -180:180 to 0:360")
        values = np.roll(field.to_numpy(), -field.shape[1] // 2, axis=1)
        
        # Interpolate to target grid
        logger.info("Interpolating field to target grid")
        values = ekr.interpolate(
            values,
            {"grid": self.config.input_resolution},
            {"grid": self.config.output_grid}
        )
        
        param_name = field.metadata("param")
        if levelist:
            param_name = f"{param_name}_{field.metadata('levelist')}"
        
        logger.info(f"Adding field {param_name} to fields dictionary.")
        fields[param_name].append(values)


class DataPreprocessor:
    """Handles data preprocessing and transformation."""
    
    def __init__(self, config: WeatherConfig):
        self.config = config
        # Soil parameter mapping
        self.soil_mapping = {
            'sot_1': 'stl1', 'sot_2': 'stl2',
            'vsw_1': 'swvl1', 'vsw_2': 'swvl2'
        }
        logger.info("DataPreprocessor initialized.")
    
    def process_surface_fields(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process surface-level fields."""
        logger.info(f"Processing surface fields: {list(fields.keys())}")
        return fields
    
    def process_soil_fields(
        self, 
        soil_data: Dict[str, np.ndarray], 
        fields: Dict[str, np.ndarray]
    ) -> None:
        """Transform and integrate soil fields."""
        logger.info(f"Processing soil fields: {list(soil_data.keys())}")
        for original_name, standard_name in self.soil_mapping.items():
            if original_name in soil_data:
                logger.info(f"Renaming {original_name} to {standard_name}")
                fields[standard_name] = soil_data.pop(original_name)
    
    def process_pressure_level_fields(self, fields: Dict[str, np.ndarray]) -> None:
        """Transform pressure level fields (e.g., GH to Z)."""
        logger.info("Processing pressure level fields for conversion GH -> Z.")
        for level in self.config.pressure_levels:
            gh_key = f"gh_{level}"
            if gh_key in fields:
                logger.info(f"Converting field {gh_key} to z_{level} by multiplying by 9.80665")
                fields[f"z_{level}"] = fields.pop(gh_key) * 9.80665


class Visualizer:
    """Handles visualization and plotting of weather data."""
    
    def __init__(self, config: WeatherConfig):
        self.config = config
        logger.info("Visualizer initialized.")
    
    @staticmethod
    def fix_longitudes(lons: np.ndarray) -> np.ndarray:
        """Shift longitudes from 0-360 to -180-180."""
        logger.info("Shifting longitudes from 0-360 to -180-180.")
        return np.where(lons > 180, lons - 360, lons)
    
    def plot_state(
        self, 
        state: WeatherState, 
        save_path: Optional[str] = None,
        field_name: Optional[str] = None
    ) -> None:
        """Plot and optionally save weather state visualization."""
        logger.info(f"Plotting weather state for {state.date}. Save path: {save_path}, Field: {field_name}")
        # Determine which field to plot
        if field_name and field_name in state.fields:
            values = state.fields[field_name]
            title_param = f"{field_name}"
        elif "100u" in state.fields:
            values = state.fields["100u"]
            title_param = "100m winds (100u)"
        elif "10u" in state.fields:
            values = state.fields["10u"]
            title_param = "10m winds (10u)"
        else:
            first_key = list(state.fields.keys())[0]
            values = state.fields[first_key]
            title_param = first_key
        
        fig, ax = plt.subplots(
            figsize=(11, 6),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        
        # Create triangulation
        triangulation = tri.Triangulation(
            self.fix_longitudes(state.longitudes),
            state.latitudes
        )
        
        # Plot contours
        contour = ax.tricontourf(
            triangulation,
            values,
            levels=20,
            transform=ccrs.PlateCarree(),
            cmap="RdBu"
        )
        
        cbar = fig.colorbar(
            contour,
            ax=ax,
            orientation="vertical",
            shrink=0.7,
            label=title_param
        )
        
        plt.title(f"{title_param} at {state.date}")
        
        if save_path:
            logger.info(f"Saving plot to {save_path}")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.close()
        logger.info("Plotting complete.")


class NetCDFWriter:
    """Handles writing weather states to NetCDF format."""
    
    @staticmethod
    def save_state(
        state: WeatherState,
        save_path: str,
        forecast_hour: int,
        reference_date: Optional[datetime.datetime] = None
    ) -> None:
        """Save weather state to NetCDF file."""
        logger.info(f"Saving state to NetCDF file: {save_path}, forecast_hour: {forecast_hour}, reference_date: {reference_date}")
        # 根据要求分开存储 ds_sfc 和 ds_pl
        ds_sfc, ds_pl = pickle2netcdf(state)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        sfc_path = str(save_path)
        if sfc_path.endswith('.nc'):
            sfc_path = sfc_path.replace('.nc', '_sfc.nc')
            pl_path = sfc_path.replace('.nc', '_pl.nc')
        else:
            sfc_path = sfc_path + "_sfc.nc"
            pl_path = sfc_path + "_pl.nc"

        ds_sfc.to_netcdf(sfc_path)
        logger.info(f"Surface state saved to NetCDF: {sfc_path}")

        ds_pl.to_netcdf(pl_path)
        logger.info(f"Pressure level state saved to NetCDF: {pl_path}")
            
def pickle2netcdf(ds):
    #print(ds['fields'].keys())
    
    def fix(lons):
        # Shift the longitudes from 0-360 to -180-180
        return np.where(lons > 180, lons - 360, lons)


    #surface_vars = ['sp', 'msl', '10u', '10v', '100u', '100v', '2t', '2d', 'skt', 'tcw', 'cp', 'tp', 'sf', 'tcc', 'hcc', 'lcc', 'mcc']

    latitudes = np.arange(90,-90.25, -0.25)
    longitudes = fix(np.arange(0, 360, 0.25))
    lons, lats = np.meshgrid(longitudes, latitudes)


    data = xr.Dataset({'2t':(['lat', 'lon'], ekr.interpolate(ds['fields']['2t'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    '2d':(['lat', 'lon'], ekr.interpolate(ds['fields']['2d'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'sp':(['lat', 'lon'], ekr.interpolate(ds['fields']['sp'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'msl':(['lat', 'lon'], ekr.interpolate(ds['fields']['msl'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    '10u':(['lat', 'lon'], ekr.interpolate(ds['fields']['10u'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    '10v':(['lat', 'lon'], ekr.interpolate(ds['fields']['10v'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    '100u':(['lat', 'lon'], ekr.interpolate(ds['fields']['100u'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    '100v':(['lat', 'lon'], ekr.interpolate(ds['fields']['100v'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'skt':(['lat', 'lon'], ekr.interpolate(ds['fields']['skt'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'tcw':(['lat', 'lon'], ekr.interpolate(ds['fields']['tcw'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'cp':(['lat', 'lon'], ekr.interpolate(ds['fields']['cp'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'tp':(['lat', 'lon'], ekr.interpolate(ds['fields']['tp'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'sf':(['lat', 'lon'], ekr.interpolate(ds['fields']['sf'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'tcc':(['lat', 'lon'], ekr.interpolate(ds['fields']['tcc'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'hcc':(['lat', 'lon'], ekr.interpolate(ds['fields']['hcc'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'lcc':(['lat', 'lon'], ekr.interpolate(ds['fields']['lcc'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'mcc':(['lat', 'lon'], ekr.interpolate(ds['fields']['mcc'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'ssrd':(['lat', 'lon'], ekr.interpolate(ds['fields']['ssrd'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'strd':(['lat', 'lon'], ekr.interpolate(ds['fields']['strd'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'stl1':(['lat', 'lon'], ekr.interpolate(ds['fields']['stl1'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'stl2':(['lat', 'lon'], ekr.interpolate(ds['fields']['stl2'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'swvl1':(['lat', 'lon'], ekr.interpolate(ds['fields']['swvl1'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    'swvl2':(['lat', 'lon'], ekr.interpolate(ds['fields']['swvl2'], {"grid": "N320"}, {"grid": (0.25, 0.25)})),
                    }, 
                    coords={'lon':np.arange(0, 360, 0.25), 
                            'lat':np.arange(90,-90.25, -0.25)})


    # Set variable attributes
    data['lon'].attrs['units'] = "degrees_east"
    data['lat'].attrs['units'] = "degrees_north"
    data['lon'].attrs['long_name'] = "longitude"
    data['lat'].attrs['long_name'] = "latitude"
    data['lon'].attrs['standard_name'] = "longitude"
    data['lat'].attrs['standard_name'] = "latitude"

    data['sp'].attrs['units'] = "Pa"
    data['msl'].attrs['units'] = "Pa"
    data['2t'].attrs['units'] = "K"
    data['2d'].attrs['units'] = "K"
    data['skt'].attrs['units'] = "K"
    data['10u'].attrs['units'] = "m/s"
    data['10v'].attrs['units'] = "m/s"
    data['100u'].attrs['units'] = "m/s"
    data['100v'].attrs['units'] = "m/s"
    data['tcw'].attrs['units'] = "kg m**-2"
    data['cp'].attrs['units'] = "m"
    data['tp'].attrs['units'] = "m"
    data['sf'].attrs['units'] = "m"
    data['ssrd'].attrs['units'] = "J m**-2"
    data['strd'].attrs['units'] = "J m**-2"
    data['stl1'].attrs['units'] = "K"
    data['stl2'].attrs['units'] = "K"
    data['swvl1'].attrs['units'] = "m**3 m**-3"
    data['swvl2'].attrs['units'] = "m**3 m**-3"

    data['sp'].attrs['long_name'] = "Surface pressure"
    data['msl'].attrs['long_name'] = "Mean sea level pressure"
    data['2t'].attrs['long_name'] = "2 metre temperature"
    data['2d'].attrs['long_name'] = "2 metre dewpoint temperature"
    data['skt'].attrs['long_name'] = "Skin temperature"
    data['10u'].attrs['long_name'] = "10 metre U wind component"
    data['10v'].attrs['long_name'] = "10 metre V wind component"
    data['100u'].attrs['long_name'] = "100 metre U wind component"
    data['100v'].attrs['long_name'] = "100 metre V wind component"
    data['tcw'].attrs['long_name'] = "Total column water"
    data['cp'].attrs['long_name'] = "Convective precipitation"
    data['tp'].attrs['long_name'] = "Total precipitation"
    data['sf'].attrs['long_name'] = "Snowfall"
    data['tcc'].attrs['long_name'] = "Total cloud cover"
    data['hcc'].attrs['long_name'] = "High cloud cover"
    data['lcc'].attrs['long_name'] = "Low cloud cover"
    data['mcc'].attrs['long_name'] = "Medium cloud cover"
    data['ssrd'].attrs['long_name'] = "Surface short-wave radiation downwards"
    data['strd'].attrs['long_name'] = "Surface long-wave radiation downwards"
    data['stl1'].attrs['long_name'] = "Soil temperature level 1"
    data['stl2'].attrs['long_name'] = "Soil temperature level 2"
    data['swvl1'].attrs['long_name'] = "Volumetric soil water layer 1"
    data['swvl2'].attrs['long_name'] = "Volumetric soil water layer 2"

    # Add Time dimension
    data.coords['time'] = ds['date']
    data_sfc = data.expand_dims('time')

    #pressure_vars = ['q', 't', 'u', 'v', 'z']

    pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

    latitudes = np.arange(90,-90.25, -0.25)
    longitudes = fix(np.arange(0, 360, 0.25))
    lons, lats = np.meshgrid(longitudes, latitudes)

    q_1000 = ekr.interpolate(ds['fields']['q_1000'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_925 = ekr.interpolate(ds['fields']['q_925'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_850 = ekr.interpolate(ds['fields']['q_850'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_700 = ekr.interpolate(ds['fields']['q_700'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_600 = ekr.interpolate(ds['fields']['q_600'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_500 = ekr.interpolate(ds['fields']['q_500'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_400 = ekr.interpolate(ds['fields']['q_400'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_300 = ekr.interpolate(ds['fields']['q_300'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_250 = ekr.interpolate(ds['fields']['q_250'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_200 = ekr.interpolate(ds['fields']['q_200'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_150 = ekr.interpolate(ds['fields']['q_150'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_100 = ekr.interpolate(ds['fields']['q_100'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    q_50 = ekr.interpolate(ds['fields']['q_50'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]

    t_1000 = ekr.interpolate(ds['fields']['t_1000'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_925 = ekr.interpolate(ds['fields']['t_925'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_850 = ekr.interpolate(ds['fields']['t_850'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_700 = ekr.interpolate(ds['fields']['t_700'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_600 = ekr.interpolate(ds['fields']['t_600'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_500 = ekr.interpolate(ds['fields']['t_500'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_400 = ekr.interpolate(ds['fields']['t_400'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_300 = ekr.interpolate(ds['fields']['t_300'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_250 = ekr.interpolate(ds['fields']['t_250'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_200 = ekr.interpolate(ds['fields']['t_200'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_150 = ekr.interpolate(ds['fields']['t_150'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_100 = ekr.interpolate(ds['fields']['t_100'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    t_50 = ekr.interpolate(ds['fields']['t_50'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]

    u_1000 = ekr.interpolate(ds['fields']['u_1000'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_925 = ekr.interpolate(ds['fields']['u_925'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_850 = ekr.interpolate(ds['fields']['u_850'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_700 = ekr.interpolate(ds['fields']['u_700'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_600 = ekr.interpolate(ds['fields']['u_600'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_500 = ekr.interpolate(ds['fields']['u_500'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_400 = ekr.interpolate(ds['fields']['u_400'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_300 = ekr.interpolate(ds['fields']['u_300'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_250 = ekr.interpolate(ds['fields']['u_250'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_200 = ekr.interpolate(ds['fields']['u_200'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_150 = ekr.interpolate(ds['fields']['u_150'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_100 = ekr.interpolate(ds['fields']['u_100'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    u_50 = ekr.interpolate(ds['fields']['u_50'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]


    v_1000 = ekr.interpolate(ds['fields']['v_1000'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_925 = ekr.interpolate(ds['fields']['v_925'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_850 = ekr.interpolate(ds['fields']['v_850'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_700 = ekr.interpolate(ds['fields']['v_700'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_600 = ekr.interpolate(ds['fields']['v_600'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_500 = ekr.interpolate(ds['fields']['v_500'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_400 = ekr.interpolate(ds['fields']['v_400'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_300 = ekr.interpolate(ds['fields']['v_300'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_250 = ekr.interpolate(ds['fields']['v_250'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_200 = ekr.interpolate(ds['fields']['v_200'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_150 = ekr.interpolate(ds['fields']['v_150'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_100 = ekr.interpolate(ds['fields']['v_100'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    v_50 = ekr.interpolate(ds['fields']['v_50'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]


    z_1000 = ekr.interpolate(ds['fields']['z_1000'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_925 = ekr.interpolate(ds['fields']['z_925'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_850 = ekr.interpolate(ds['fields']['z_850'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_700 = ekr.interpolate(ds['fields']['z_700'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_600 = ekr.interpolate(ds['fields']['z_600'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_500 = ekr.interpolate(ds['fields']['z_500'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_400 = ekr.interpolate(ds['fields']['z_400'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_300 = ekr.interpolate(ds['fields']['z_300'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_250 = ekr.interpolate(ds['fields']['z_250'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_200 = ekr.interpolate(ds['fields']['z_200'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_150 = ekr.interpolate(ds['fields']['z_150'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_100 = ekr.interpolate(ds['fields']['z_100'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]
    z_50 = ekr.interpolate(ds['fields']['z_50'], {"grid": "N320"}, {"grid": (0.25, 0.25)})[np.newaxis,:,:]

    data = xr.Dataset({'q':(['level','lat', 'lon'], np.concatenate([q_1000, q_925, q_850, q_700, q_600, q_500, q_400, q_300, 
                                                                    q_250, q_200, q_150, q_100, q_50], axis=0)),
                    't':(['level','lat', 'lon'], np.concatenate([t_1000, t_925, t_850, t_700, t_600, t_500, t_400, t_300, 
                                                                                    t_250, t_200, t_150, t_100, t_50], axis=0)),
                    'u':(['level','lat', 'lon'], np.concatenate([u_1000, u_925, u_850, u_700, u_600, u_500, u_400, u_300, 
                                                                                    u_250, u_200, u_150, u_100, u_50], axis=0)),
                    'v':(['level','lat', 'lon'], np.concatenate([v_1000, v_925, v_850, v_700, v_600, v_500, v_400, v_300, 
                                                                                    v_250, v_200, v_150, v_100, v_50], axis=0)),
                    'z':(['level','lat', 'lon'], np.concatenate([z_1000, z_925, z_850, z_700, z_600, z_500, z_400, z_300, 
                                                                                    z_250, z_200, z_150, z_100, z_50], axis=0)),
                    }, 
                    coords={'level':np.array(pressure_levels),
                            'lon':np.arange(0, 360, 0.25), 
                            'lat':np.arange(90,-90.25, -0.25)})

    # Set variable attributes
    data['level'].attrs['units'] = "hPa"
    data['level'].attrs['standard_name'] = "air_pressure"
    data['level'].attrs['long_name'] = "pressure"

    data['lon'].attrs['units'] = "degrees_east"
    data['lat'].attrs['units'] = "degrees_north"
    data['lon'].attrs['long_name'] = "longitude"
    data['lat'].attrs['long_name'] = "latitude"
    data['lon'].attrs['standard_name'] = "longitude"
    data['lat'].attrs['standard_name'] = "latitude"

    data['z'].attrs['units'] = "m2/s2"
    data['q'].attrs['units'] = "kg/kg"
    data['u'].attrs['units'] = "m/s"
    data['v'].attrs['units'] = "m/s"
    data['t'].attrs['units'] = "K"

    data['z'].attrs['long_name'] = "m2/s2"
    data['q'].attrs['long_name'] = "kg/kg"
    data['u'].attrs['long_name'] = "m/s"
    data['v'].attrs['long_name'] = "m/s"
    data['t'].attrs['long_name'] = "K"

    data['z'].attrs['long_name'] = "Geopotential"
    data['q'].attrs['long_name'] = "Specific humidity"
    data['u'].attrs['long_name'] = "U component of wind"
    data['v'].attrs['long_name'] = "V component of wind"
    data['t'].attrs['long_name'] = "Temperature"

    data['z'].attrs['standard_name'] = "geopotential"
    data['q'].attrs['standard_name'] = "specific humidity"
    data['u'].attrs['standard_name'] = "eastward_wind"
    data['v'].attrs['standard_name'] = "northward_wind"
    data['t'].attrs['standard_name'] = "air_temperature"

    # Add Time dimension
    data.coords['time'] = ds['date']
    data_pl = data.expand_dims('time')
    return data_sfc, data_pl


class AIFSPipeline:
    """Main AIFS forecasting pipeline."""
    
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.data_provider = DataProvider(config)
        self.preprocessor = DataPreprocessor(config)
        self.visualizer = Visualizer(config)
        
        logger.info("AIFS Pipeline initialized")
    
    def load_initial_state(self, date: datetime.datetime) -> WeatherState:
        """Load initial weather state from data sources."""
        logger.info(f"Loading initial state for {date}")
        
        fields = {}
        
        # Fetch surface parameters
        logger.info("Fetching surface parameters.")
        sfc_data = self.data_provider.fetch_parameters(
            date,
            self.config.surface_params
        )
        fields.update(sfc_data)
        logger.info("Surface parameters fetched.")

        # Fetch and process soil parameters
        logger.info("Fetching soil parameters.")
        soil_data = self.data_provider.fetch_parameters(
            date,
            self.config.soil_params,
            self.config.soil_levels
        )
        logger.info("Soil parameters fetched; processing soil fields.")
        self.preprocessor.process_soil_fields(soil_data, fields)
        
        # Fetch and process pressure level parameters
        logger.info("Fetching pressure level parameters.")
        pl_data = self.data_provider.fetch_parameters(
            date,
            self.config.pressure_level_params,
            self.config.pressure_levels
        )
        logger.info("Pressure level parameters fetched; processing pressure level fields.")
        fields.update(pl_data)
        self.preprocessor.process_pressure_level_fields(fields)
        
        logger.info(f"Loaded {len(fields)} fields")
        for name, field_data in fields.items():
            logger.info(f"  {name}: {field_data.shape}")
        
        logger.info(f"Returning WeatherState for {date}.")
        return WeatherState(date=date, fields=fields)
    
    def run_forecast(self, initial_state: WeatherState, output_dir: Optional[str] = None) -> None:
        """Run the full forecasting pipeline."""
        logger.info("Starting forecast pipeline execution.")
        output_dir = output_dir or self._get_output_dir(initial_state.date)
        complete_file = Path(output_dir) / f"{initial_state.date.strftime('%Y%m%d%H')}.complete"
        
        if complete_file.exists():
            logger.info(f"Forecast already completed: {complete_file}")
            return
        
        # Validate checkpoint
        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from: {checkpoint_path}")
        runner = SimpleRunner(str(checkpoint_path), device=self.config.device)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run forecast
        logger.info("Running model forecast.")
        for i, state in enumerate(runner.run(
            input_state=initial_state.to_dict(),
            lead_time=self.config.lead_time_hours
        )):
            forecast_hour = (i + 1) * self.config.time_step_hours
            
            logger.info(f"Processing forecast hour {forecast_hour}")
            print_state(state)
            
            # Save to NetCDF
            nc_filename = f"{forecast_hour:03d}.nc"
            nc_path = str(Path(output_dir) / nc_filename)
            logger.info(f"Saving NetCDF data for hour {forecast_hour} to {nc_path}")
            NetCDFWriter.save_state(
                WeatherState(**state),
                nc_path,
                forecast_hour,
                initial_state.date
            )
        
        # Mark completion
        logger.info(f"Writing completion marker to {complete_file}")
        complete_file.write_text(
            f"Completed at {datetime.datetime.now().isoformat()}"
        )
        logger.info(f"Forecast completed: {complete_file}")
    
    def _get_output_dir(self, date: datetime.datetime) -> str:
        """Generate output directory path."""
        date_str = date.strftime("%Y%m%d%H")
        output_dir = str(Path(self.config.output_nc_base_path) / f"{date_str}")
        logger.info(f"Computed output directory: {output_dir}")
        return output_dir


def main(config: Optional[WeatherConfig] = None) -> None:
    """Main entry point."""
    logger.info("Starting AIFS main function...")
    
    config = config or WeatherConfig()
    logger.info(f"Using config: {config}")

    pipeline = AIFSPipeline(config)
    
    # Get latest date
    logger.info("Getting latest initial date...")
    date = pipeline.data_provider.get_latest_date()
    logger.info(f"Initial date: {date}")
    
    # Load initial state
    logger.info("Loading initial state...")
    initial_state = pipeline.load_initial_state(date)
    
    # Run forecast
    logger.info("Running forecast...")
    pipeline.run_forecast(initial_state)
    logger.info("AIFS forecast finished.")


if __name__ == "__main__":
    logger.info("AIFS script execution started as main module.")
    main()
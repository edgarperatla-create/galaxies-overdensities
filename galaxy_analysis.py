#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced galaxy catalog analysis with YAML configuration
"""


# %% Librerías
import time
import os
import re
import logging
import yaml
from typing import Optional
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import simpson as simp
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


## Gráficas
import matplotlib.pyplot as plt
import seaborn as sns
style = 'darkgrid'
sns.set(style=style)


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



# %% Parámetros iniciales
M_ce = 10.0
M_ve = 8.75
S_ce = 1.0
S_ve = 0.5

r_sec = 120 # radio de los círculos sd en arcosegundos
r_deg = r_sec/3600
r_Mpc = 10   # radio de los círculos sd en Mpc


# %% Clases
class PathFinder:
    """Buscador de ruta con validación extendida"""
    @staticmethod
    def get_script_directory() -> str:
        """Regresa la ruta absoluta al directorio del script con 100% de fiabilidad"""
        # Se prueban todos los métodos posibles con su respectiva validación
        attempts = [
            lambda: os.path.dirname(os.path.abspath(__file__)),  # Script estándar
            lambda: os.getcwd(),                                  # Interactivo/Libreta
            lambda: os.path.abspath(''),                          # Directorio actual
            lambda: os.path.expanduser('~')                       # Directorio base
        ]
        
        for attempt in attempts:
            try:
                path = attempt()
                if isinstance(path, str) and os.path.isdir(path):
                    return path
            except:
                continue
                
        raise RuntimeError("No se pudo determinar la ubicación del script después de todos los intentos")

    @staticmethod
    def find_config_file(filename: Optional[str] = None) -> str:
        """Encuentra el archivo config en su ruta absoluta y con validación completa"""
        filename = filename or "config.yaml"
        
        if not isinstance(filename, (str, bytes, os.PathLike)):
            raise TypeError(f"El nombre del archivo debe ser de tipo string o tipo Path, no {type(filename)}")
            
        script_dir = PathFinder.get_script_directory()
        logger.debug(f"Directorio del script resuelto a: {script_dir}")
        
        # Check in script directory first
        config_path = os.path.join(script_dir, filename)
        if os.path.isfile(config_path):
            return os.path.abspath(config_path)
            
        # Check one level up if not found
        parent_dir = os.path.dirname(script_dir)
        parent_config = os.path.join(parent_dir, filename)
        if os.path.isfile(parent_config):
            return os.path.abspath(parent_config)
            
        # Final check in current working directory
        cwd_config = os.path.join(os.getcwd(), filename)
        if os.path.isfile(cwd_config):
            return os.path.abspath(cwd_config)
            
        raise FileNotFoundError(
            f"Could not find {filename} in:\n"
            f"- {script_dir}\n"
            f"- {parent_dir}\n"
            f"- {os.getcwd()}"
        )

class ConfigManager:
    def __init__(self):
        """Always finds config.yaml in the same directory as this script"""
        # Get the directory containing THIS script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(script_dir, "config.yaml")
        
        # Verify config exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"config.yaml not found in script directory: {script_dir}\n"
                f"Please ensure it exists at: {self.config_path}"
            )
            
        # Load config
        try:
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f) or {}
                self.config.setdefault('paths', {})
                self.config.setdefault('files', {})
        except Exception as e:
            raise RuntimeError(f"Failed to load config.yaml: {str(e)}")

    def get_full_path(self, catalog_type, catalog_name):
        """Build absolute path for any catalog"""
        try:
            # Construct path relative to home directory
            rel_path = self.config['paths'][catalog_type][catalog_name]
            home = os.path.expanduser("~")
            return os.path.join(home, rel_path)
        except KeyError as e:
            available = list(self.config['paths'].get(catalog_type, {}).keys())
            raise KeyError(
                f"Missing path for {catalog_type}/{catalog_name}. "
                f"Available {catalog_type} catalogs: {available}"
            ) from e

    def get_completeness_path(self, name):
        """Get full path to completeness data file."""
        try:
            base_path = os.path.expanduser(self.config['paths']['completeness_data'][name])
            filename = self.config['files']['completeness_data'][name] + '.csv'
            return os.path.join(base_path, filename)
        except KeyError as e:
            raise KeyError(f"Missing completeness data config for {name}. Check config.yaml") from e

class CosmologyUtils:
    """Handles cosmological calculations"""
    
    def __init__(self, cosmology=cosmo):
        self.cosmo = cosmology
        
    @property
    def hubble_distance(self):
        """Hubble distance (c/H0) in Mpc"""
        return self.cosmo.hubble_distance.value
        
    def efunc(self, z):
        """E(z) = H(z)/H0"""
        return self.cosmo.efunc(z)
        
    def comoving_distance(self, z: float) -> float:
        """Calculate comoving distance in Mpc"""
        return self.cosmo.comoving_distance(z).value
        
    def angular_scale(self, z: float, physical_size: float) -> float:
        """
        Convert physical size to angular scale
        Args:
            z: redshift
            physical_size: size in Mpc
        Returns:
            angular size in degrees
        """
        dc = self.comoving_distance(z)
        return np.degrees(physical_size / dc)
        
    def redshift_to_comoving(self, ra: float, dec: float, z: float) -> np.ndarray:
        """
        Convert (RA, Dec, z) to comoving coordinates (x, y, z) in Mpc
        """
        # Convert to radians
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        # Comoving distance
        dc = self.comoving_distance(z)
        
        # Cartesian coordinates
        x = dc * np.cos(dec_rad) * np.cos(ra_rad)
        y = dc * np.cos(dec_rad) * np.sin(ra_rad)
        z_coord = dc * np.sin(dec_rad)
        
        return np.array([x, y, z_coord])
    
    def create_redshift_bins(self, z_min: float, z_max: float, n_bins: int = 10, log: bool = True) -> np.ndarray:
        """
        Create redshift bins, either linear or logarithmic
        """
        if log:
            return np.logspace(np.log10(z_min), np.log10(z_max), n_bins+1)
        return np.linspace(z_min, z_max, n_bins+1)

class CampGeom:
    """Handles complex survey geometry calculations"""
    
    def __init__(self, lower_boundary: np.ndarray, upper_boundary: np.ndarray):
        """
        Args:
            lower_boundary: Array of (RA, Dec) points for lower boundary
            upper_boundary: Array of (RA, Dec) points for upper boundary
        """
        self.lower = lower_boundary
        self.upper = upper_boundary
        self._init_boundary_functions()
        
    def _init_boundary_functions(self):
        """Pre-compute boundary functions"""
        # Lower boundary
        self.m_low = (self.lower[2][1]-self.lower[1][1])/(self.lower[2][0]-self.lower[1][0])
        self.b_low = ((self.lower[1][1]+self.lower[2][1])/2) - self.m_low*((self.lower[1][0]+self.lower[2][0])/2)
        
        # Upper boundary
        self.m_up = (self.upper[1][1]-self.upper[0][1])/(self.upper[1][0]-self.upper[0][0])
        self.b_up = ((self.upper[0][1]+self.upper[1][1])/2) - self.m_up*((self.upper[0][0]+self.upper[1][0])/2)
    
    def lower_bound(self, ra: float) -> float:
        """Get Dec lower bound for given RA"""
        if self.lower[0][0] <= ra < self.lower[1][0]:
            return self.lower[0][1]
        elif self.lower[1][0] < ra <= self.lower[2][0]:
            return self.m_low*ra + self.b_low
        return np.nan
    
    def upper_bound(self, ra: float) -> float:
        """Get Dec upper bound for given RA"""
        if self.upper[0][0] <= ra < self.upper[1][0]:
            return self.m_up*ra + self.b_up
        elif self.upper[1][0] < ra <= self.upper[2][0]:
            return self.upper[2][1]
        return np.nan
    
    def calculate_area(self, n_points: int = 100) -> float:
        """Calculate field area using numerical integration"""
        ra_values = np.linspace(self.lower[0][0], self.lower[2][0], n_points)
        lower_dec = np.array([self.lower_bound(ra) for ra in ra_values])
        upper_dec = np.array([self.upper_bound(ra) for ra in ra_values])
        return simp(upper_dec - lower_dec, x=ra_values)
    
    def contains(self, ra: float, dec: float) -> bool:
        """Check if point is within field boundaries"""
        lower = self.lower_bound(ra)
        upper = self.upper_bound(ra)
        return not np.isnan(lower) and not np.isnan(upper) and lower <= dec <= upper

class Cat:
    """Clase base para todos los catálogos"""
    def __init__(self, config=None, catalog_name=None, force_log_mass=True, is_simulation=True):
        """
        Args:
            config: Instancia de ConfigManager
            catalog_name: Nombre del catálogo
            force_log_mass: Si es necesario convertir masas a log10
            is_simulation: Si es un catálogo de una simulación
        """
        self.config = config
        self.catalog_name = catalog_name
        self.force_log_mass = force_log_mass
        self.is_simulation = is_simulation
        self.metadata = {}
        self.columns = {}
        self.stats = {}
        self.data = None

        if config:
            if is_simulation:
                self._initialize_sim_paths()
                self._parse_config()
                self._read_data()
                self._add_unique_ids()
                self._standardize_column_names()  # This should come after _read_data
            
            if force_log_mass and self.data is not None:
                self._auto_convert_masses()
                
            # Rebuild stats after potential column renaming
            self.compute_stats()

    def _add_unique_ids(self):
        """Only add IDs to simulation catalogs"""
        if self.is_simulation and 'ID' not in self.data.columns:
            prefix = self.catalog_name[:3].upper()  # e.g., 'AR_' for araceli
            self.data['ID'] = [f"{prefix}_{i:05d}" for i in range(len(self.data))]
            self.columns['ID'] = {'units': None, 'description': 'Unique identifier'}

    def _standardize_column_names(self):
        """Standardize column names across different simulations"""
        # Create mapping of alternate names to preferred names
        column_mapping = {
            'z_obs': 'z',       # Aldo simulation uses 'z_obs'
            'redshift': 'z',    # Other possible variants
            'Z': 'z',
            'z_best': 'z'
        }
        
        # Rename columns if needed
        renamed = False
        for old_name, new_name in column_mapping.items():
            if old_name in self.data.columns and new_name not in self.data.columns:
                self.data.rename(columns={old_name: new_name}, inplace=True)
                renamed = True
                
                # Update metadata if exists
                if old_name in self.columns:
                    self.columns[new_name] = self.columns.pop(old_name)
                if old_name in self.stats:
                    self.stats[new_name] = self.stats.pop(old_name)
        
        if renamed:
            print(f"Renamed columns in {self.catalog_name} catalog: {column_mapping}")

    def _initialize_sim_paths(self):
        """Simulation-specific path setup"""
        if not self.is_simulation:
            return
            
        base_path = self.config.get_full_path('simulations', self.catalog_name)
        file_info = self.config.config['files'][self.catalog_name]
        
        self.data_file = os.path.join(base_path, file_info['table'] + '.txt')
        self.config_file = os.path.join(base_path, file_info['metadata'] + '.txt')

    def _auto_convert_masses(self):
        """Automatically detect and convert linear masses to log10"""
        mass_keywords = ['M_vir_halo', 'M_star', 'mass', 'm_', 'mstar', 'mvir', 'm_halo']
        
        for col in self.data.columns:
            if any(kw in col.lower() for kw in mass_keywords):
                col_data = self.data[col]
                
                # Detection heuristic (linear masses typically > 1e8)
                if self._is_linear_mass(col_data):
                    print(f"Converting {col} from linear to log10")
                    self.data[col] = np.log10(col_data)
                    if col in self.columns:
                        self.columns[col]['units'] = f"log({self.columns[col].get('units', 'M_sun')})"
                else:
                    print(f"{col} appears to already be logarithmic (no conversion needed)")

    def _is_linear_mass(self, col_data):
        """Heuristic to detect linear mass columns"""
        median = np.nanmedian(col_data)
        return median > 1e8  # Typical linear mass threshold


    def _parse_config(self):
        """Robust parser for the complex metadata file format"""
        with open(self.config_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]  # Remove empty lines
    
        # --- Parse Header ---
        for line in lines:
            if line.startswith("Table name:"):
                self.metadata["table_name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Number of galaxies:"):
                self.metadata["n_galaxies"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Area:"):
                self.metadata["area"] = line.split(":", 1)[1].strip()
            elif line.startswith("+-----"):
                break
    
        # --- Parse Statistics Table ---
        table_start = next(i for i, line in enumerate(lines) if line.startswith("+-----"))
        # headers = [h.strip() for h in re.split(r'\s*\|\s*', lines[table_start+2])[1:-1]]
    
        for line in lines[table_start+3:]:
            if line.startswith("+-----"):
                break
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in re.split(r'\s*\|\s*', line)[1:-1]]
            self.stats[parts[0]] = {
                "mean": float(parts[1]),
                "sd": float(parts[2]),
                "min": float(parts[3]),
                "max": float(parts[4]),
            }
    
        # --- Parse Column Descriptions ---
        desc_start = next((i for i, line in enumerate(lines) 
                         if line.startswith("Column") and "Units" in line), None)
        
        if desc_start:
            for line in lines[desc_start+1:]:
                if not line.strip() or line.startswith("Notes:"):
                    break
                # Updated parsing for "2: RA" format
                match = re.match(r'(\d+):\s+(\w+)\s+(\S+.*)', line.strip())
                if match:
                    # col_id = int(match.group(1))
                    col_name = match.group(2)
                    rest = match.group(3).split(maxsplit=1)
                    units = rest[0] if len(rest) > 0 else None
                    desc = rest[1] if len(rest) > 1 else None
                    self.columns[col_name] = {
                        "units": units if units != "None" else None,
                        "description": desc
                    }

    def _read_data(self):
        """Read data file correctly handling all column/index issues"""
        # Read the data file with these key parameters:
        self.data = pd.read_csv(
            self.data_file,
            sep=r'\s+',          # Whitespace separator
            header=None,         # No header row
            # skiprows=1,          # Skip first row (with # and column numbers)
            comment='#',         # Skip any comment lines
            names=list(self.stats.keys()),  # Use metadata column names
            dtype=float
        )
        
        # Reset any automatic index pandas created
        self.data.reset_index(drop=True, inplace=True)
        
        # Verify column alignment
        if len(self.data.columns) != len(self.stats):
            raise ValueError(
                f"Data has {len(self.data.columns)} columns but "
                f"metadata expects {len(self.stats)}"
            )

    def mean(self, column_name):
        return self.data[column_name].mean()
    
    def std(self, column_name):
        return self.data[column_name].std()
    
    def plot_histogram(self, column_name, nbins=20, logx=False, logy=False, density=False, med=False, mn=False, std=False):
        """Plot histogram with optional log scales and density normalization.
        
        Args:
            column_name (str): Column to plot.
            nbins (int): Number of bins.
            logx (bool): Use log scale on x-axis.
            logy (bool): Use log scale on y-axis.
            density (bool): Normalize to probability density (area under histogram = 1).
        """
        
        plt.figure()
        
        # Set bins based on log/linear scale
        if logx:
            bins = np.logspace(
                np.log10(self.data[column_name].min()),
                np.log10(self.data[column_name].max()),
                nbins
            )
            plt.xscale('log')
        else:
            bins = np.linspace(
                self.data[column_name].min(),
                self.data[column_name].max(),
                nbins
            )
        
        # Plot histogram with density option
        plt.hist(
            self.data[column_name],
            bins=bins,
            edgecolor='k',
            linewidth=0.5,
            alpha=0.7,
            density=density  # <-- Key addition
        )
        
        # Adjust y-axis scale
        if logy:
            plt.yscale('log')
        
        # Añadir medianas, medias o desviación estándar
        if med:
            plt.axvline(x=np.median(self.data[column_name]), label=r'$med\,=\,$'+str(round(np.median(self.data[column_name]),2)))
            plt.legend(loc='best')
        if mn:
            plt.axvline(x=np.mean(self.data[column_name]), label=r'$mn\,=\,$'+str(round(np.mean(self.data[column_name]),2)))
            plt.legend(loc='best')
        if std:
            plt.axvline(x=np.mean(self.data[column_name])-np.std(self.data[column_name]), label=r'$\sigma\,=\,\pm$'+str(round(np.std(self.data[column_name]),2)),linestyle='--')
            plt.axvline(x=np.mean(self.data[column_name])+np.std(self.data[column_name]),linestyle='--')
            plt.legend(loc='best')

        
        # Labels and title
        plt.xlabel(f"{column_name} [{self.columns[column_name].get('units', '')}]")
        plt.ylabel("Probability Density" if density else "Count")
        plt.title(f"Distribution of {column_name} ({'Density' if density else 'Count'})")
        plt.show()
    
    def plot_scatter(self, x_col, y_col, logx=False, logy=False, s=10, equal_axis=False):
        """Scatter plot with optional log scales"""
        
        plt.figure()
        plt.scatter(self.data[x_col], self.data[y_col], alpha=0.5, s=s)
        
        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')

        # Equalize axis scaling (square plot)
        if equal_axis:
            plt.gca().set_aspect('equal', adjustable='datalim')
            plt.axis('equal')  # Alternative method (redundant but ensures robustness)
        
        plt.xlabel(f"{x_col} [{self.columns[x_col]['units']}]")
        plt.ylabel(f"{y_col} [{self.columns[y_col]['units']}]")
        plt.title(f"{y_col} vs {x_col}")
        # plt.grid(True, which="both", ls="--")
        plt.show()
    
    def apply_cuts(self, cuts):
        """
        Returns a new Catalog/ObservationCatalog instance with filtered data,
        preserving all attributes and methods.
        """
        # Create mask
        combined_mask = np.ones(len(self.data), dtype=bool)
        for col, min_val, max_val in cuts:
            col_mask = (self.data[col] >= min_val) & (self.data[col] <= max_val)
            combined_mask &= col_mask
    
        # Create new instance (works for both parent and child classes)
        filtered_catalog = self.__class__.__new__(self.__class__)
        
        # Copy all attributes from original catalog
        filtered_catalog.__dict__ = {k: v.copy() if hasattr(v, 'copy') else v 
                                   for k, v in self.__dict__.items()}
        
        # Apply the mask to data
        filtered_catalog.data = self.data[combined_mask].copy()
        
        return filtered_catalog

    def compute_stats(self, columns=None):
        """
        Compute statistics only for numeric columns, skipping strings and IDs.
        Automatically excludes columns containing 'ID' in their name.
        """
        if columns is None:
            # Auto-detect numeric columns and exclude IDs
            numeric_cols = [col for col in self.data.columns 
                           if pd.api.types.is_numeric_dtype(self.data[col]) 
                           and ('ID' not in col)]
        else:
            # Use specified columns but verify they're numeric
            numeric_cols = []
            for col in columns:
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    print(f"Warning: Skipping non-numeric column '{col}'")
                    continue
                numeric_cols.append(col)
        
        # Compute stats for selected columns
        for col in numeric_cols:
            self.stats[col] = {
                "mean": float(self.data[col].mean()),
                "sd": float(self.data[col].std()),
                "min": float(self.data[col].min()),
                "max": float(self.data[col].max())
            }




    def match_multi_catalogs(self, other_catalogs, id_mappings, columns_to_keep=None):
        """
        Improved version that:
        1. Only keeps one ID column (from base catalog)
        2. Properly includes all requested columns
        3. Handles column name conflicts better
        """
        matched_catalog = self.__class__.__new__(self.__class__)
        matched_catalog.__dict__ = {k: v for k, v in self.__dict__.items()}
        
        # Get base ID column name (from first wavelength)
        base_id_col = list(id_mappings.values())[0]['self']
        
        # Start with base catalog data including the ID and requested columns
        if columns_to_keep and 'self' in columns_to_keep:
            keep_cols = [base_id_col] + [c for c in columns_to_keep['self'] if c in self.data.columns]
        else:
            keep_cols = [base_id_col]
        
        result_df = self.data[keep_cols].copy()
        
        # Merge other catalogs
        for catalog, catalog_name in other_catalogs:
            if catalog_name in columns_to_keep:
                # Find the ID column mapping for this catalog
                their_id_col = None
                for mapping in id_mappings.values():
                    if catalog_name in mapping:
                        their_id_col = mapping[catalog_name]
                        break
                
                if their_id_col and their_id_col in catalog.data.columns:
                    # Select only desired columns from this catalog
                    cols_to_merge = [c for c in columns_to_keep[catalog_name] 
                                   if c in catalog.data.columns]
                    
                    # Skip if no columns to merge
                    if not cols_to_merge:
                        continue
                    
                    # Perform the merge
                    result_df = pd.merge(
                        result_df,
                        catalog.data[[their_id_col] + cols_to_merge],
                        left_on=base_id_col,
                        right_on=their_id_col,
                        how='left',
                        suffixes=('', f'_{catalog_name}')
                    )
                    
                    # Drop the redundant ID column from the merged catalog
                    if their_id_col in result_df.columns and their_id_col != base_id_col:
                        result_df.drop(columns=[their_id_col], inplace=True)
        
        matched_catalog.data = result_df
        matched_catalog._rebuild_columns_metadata(other_catalogs)
        return matched_catalog
    
    def _perform_selective_matching(self, other_catalogs, id_mappings, id_columns, columns_to_keep):
        """Merge only specified columns while preserving ID matching"""
        result_df = self.data[[id_columns['self']]].copy()
        
        # Add base catalog columns
        if columns_to_keep and 'self' in columns_to_keep:
            for col in columns_to_keep['self']:
                if col in self.data.columns:
                    result_df[col] = self.data[col]
                else:
                    logger.warning(f"Column {col} not found in base catalog")
        
        # Merge other catalogs
        for catalog, catalog_name in other_catalogs:
            if catalog_name in columns_to_keep:
                # Get ID column for this catalog
                their_id_col = None
                for mapping in id_mappings.values():
                    if catalog_name in mapping:
                        their_id_col = mapping[catalog_name]
                        break
                
                if their_id_col:
                    # Select only desired columns
                    cols_to_merge = [their_id_col] + columns_to_keep[catalog_name]
                    cols_to_merge = [c for c in cols_to_merge if c in catalog.data.columns]
                    
                    result_df = pd.merge(
                        result_df,
                        catalog.data[cols_to_merge],
                        left_on=id_columns['self'],
                        right_on=their_id_col,
                        how='left',
                        suffixes=('', f'_{catalog_name}')
                    )
        
        return result_df

    def _rebuild_columns_metadata(self, other_catalogs):
        """Combine columns metadata from all matched catalogs"""
        self.columns = {}
        
        # Add our original columns
        for col in self.data.columns:
            if col in self.__dict__.get('columns', {}):
                self.columns[col] = self.__dict__['columns'][col].copy()
            else:
                # Check other catalogs
                for catalog, _ in other_catalogs:
                    if col in catalog.columns:
                        self.columns[col] = catalog.columns[col].copy()
                        break
                else:
                    # Default metadata
                    self.columns[col] = {
                        'units': None,
                        'description': f"Merged column: {col}"
                    }

    def __repr__(self):
        """Show all columns in representation"""
        cols = list(self.data.columns) if hasattr(self, 'data') else []
        return (f"Catalog: {self.metadata.get('table_name', 'Unnamed')}\n"
                f"Columns: {cols[:10]}{'...' if len(cols) > 10 else ''}")

class Cat_Obs(Cat):
    def __init__(self, config, catalog_name, usecols=None, col_defs=None, has_header=False):
            # Initialize parent without calling compute_stats yet
            self.config = config
            self.catalog_name = catalog_name
            self.force_log_mass = False
            self.is_simulation = False
            self.metadata = {}
            self.columns = {}
            self.stats = {}
            self.data = None
            
            self.usecols = usecols
            self.col_defs = col_defs
            self.has_header = has_header
            
            # Now load the data
            self._initialize_obs_paths()
            self._read_obs_data()
            self._init_obs_columns()
            
            # Only now compute stats after data is loaded
            self.compute_stats()

    def _add_unique_ids(self):
        """Skip ID creation for observational catalogs"""
        pass  # Preserve original IDs

    def _initialize_obs_paths(self):
        """Observation-specific path setup"""
        base_path = self.config.get_full_path('observations', self.catalog_name)
        file_config = self.config.config['files'][self.catalog_name]
        
        ext = '.csv' if self.catalog_name == 'kevin' else '.txt'
        self.data_file = os.path.join(base_path, file_config['table'] + ext)
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

    def _detect_delimiter(self, file_path, sample_lines=5):
        """Detect if the file is comma or whitespace-separated."""
        delimiters = [',', r'\s+']  # Prioritize comma first
        with open(file_path, 'r') as f:
            lines = [f.readline() for _ in range(sample_lines)]
        for delim in delimiters:
            # Check if delimiter splits lines into >1 column
            if all(len(re.split(delim, line.strip())) > 1 for line in lines if line.strip()):
                return delim
        return r'\s+'  # Default fallback

    def _read_obs_data(self):
        """Read data with better error handling"""
        try:
            delimiter = self._detect_delimiter(self.data_file)
            self.data = pd.read_csv(
                self.data_file,
                sep=delimiter,
                engine='python',
                usecols=self.usecols,
                header=0 if self.has_header else None,
                comment='#',
                names=[col[0] for col in self.col_defs] if self.col_defs else None
            )
            if self.data.empty:
                raise ValueError("Loaded empty DataFrame")
        except Exception as e:
            print(f"\nERROR loading {self.catalog_name}:")
            print(f"File: {self.data_file}")
            print(f"Error: {str(e)}")
            print("\nFile contents preview:")
            with open(self.data_file, 'r') as f:
                print(f.read(500))  # Show first 500 chars
            raise
            
    def _init_obs_columns(self):
        """Initialize observation metadata"""
        if self.col_defs:
            self.columns = {
                name: {"units": unit, "description": desc}
                for name, unit, desc in self.col_defs
            }
        else:
            self.columns = {
                col: {"units": None, "description": None}
                for col in self.data.columns
            }

# %% Datos

# %%% Observaciones

#### Lectura de datos
# Initialize config
config = ConfigManager()

# %%%% Stefanon (CANDELS)
##### Stefanon (vecinas):
cols_St = [0, 1, 2, 5, 7, 8, 9, 11, 12, 13, 17, 18, 27]  # Columnas utilizadas
cols_def_St = [
    ('IDnum', None, None),
    ('RA', 'deg', 'RA'),
    ('Dec', 'deg', 'Dec'),
    ('Class', '[0=gal]', 'Classification (0=galaxy)'),
    ('z_phot', None, None),
    ('z_spec', None, None),
    ('z_spec_q', None, None),
    ('z', None, 'Redshift used'),
    ('z_best_low68', None, None),
    ('z_best_high68', None, None),
    ('MS_log', 'M_sun', 'Stellar mass (log)'),
    ('e_MS_log', None, None),
    ('M_star', 'M_sun', 'Stellar mass used') ]

cat_St = Cat_Obs(
    config=config,
    catalog_name='stefanon',
    usecols=cols_St,
    col_defs=cols_def_St,
    has_header=False )

cuts_St = [
    ('z', 0.0, 6.0),     # rango de corrimientos
    ('Class', 0.0, 0.9),         # clase = galaxias (de acuerdo al crit. de Kev)
    ('M_star', M_ve, 15.0) ]            # rango de masas

cat_St_cuts = cat_St.apply_cuts(cuts_St)
St_cts_z_mn = cat_St_cuts.data['z'].mean()
St_cts_z_med = cat_St_cuts.data['z'].median()
St_cts_z_std = cat_St_cuts.data['z'].std()

# %%%% Lu-Zavala (SCUBA)
##### Kevin (centrales):
cols_Ke = [0,15,16]
cols_def_Ke = [('ID_SCUBA',None,None),('sd_120sec',None,'sobredensidad (radio=120sec)'),('e_sd120s',None,'error de la sobredensidad')]
cat_Ke = Cat_Obs(
    config=config,
    catalog_name='kevin',
    usecols=cols_Ke,
    has_header=True )

##### Lu (centrales):
cols_Lu = [0,1,3,6,7,8]
cols_def_Lu = [('ID_SCUBA_850',None,None),('ID_SCUBA_450',None,None),('M_barro','M_sun','Masa estelar (Barro)'),('RA','deg','Asencion Recta'),('Dec','deg','Declinacion'),('z','None','z_best')]
cat_Lu = Cat_Obs(
    config=config,
    catalog_name='luisa',
    usecols=cols_Lu,
    col_defs=cols_def_Lu,
    has_header=True )

##### Zavala (centrales):
cols_Za = [5,11,13,19]
cols_def_Za = [('ID_SCUBA_850',None,None),('S850','mJy/beam','flujo a 850'),('ID_SCUBA_450',None,None),('S450','mJy/beam','flujo a 450')]
cat_Za = Cat_Obs(
    config=config,
    catalog_name='zavala',
    usecols=cols_Za,
    col_defs=cols_def_Za,
    has_header=True )


##### Match de fuentes KeLuZa (centrales):
id_mappings = {
    '850': {'self': 'ID_SCUBA',
            'Lu': 'ID_SCUBA_850',
            'Za': 'ID_SCUBA_850'},
    '450': {'self': 'ID_SCUBA',          # columna ID de Kevin a 450µm
            'Lu': 'ID_SCUBA_450',        # columna ID de Luisa a 450µm
            'Za': 'ID_SCUBA_450'} }      # columna ID de Zavala a 450µm

columns_to_keep = {
    'self': ['sd_120', 'e_sd120'],  # columnas que usaremos de Kevin
    'Za': ['S850', 'S450'],          # flujos de Zavala
    'Lu': ['M_barro', 'z', 'RA', 'Dec'] } # masa, corrimiento y coordenadas de Lu

# Nos aseguramos que cada ID se lea como str para poder hacer el match
cat_Ke.data['ID_SCUBA'] = cat_Ke.data['ID_SCUBA'].astype(str)
cat_Lu.data['ID_SCUBA_850'] = cat_Lu.data['ID_SCUBA_850'].astype(str)
cat_Za.data['ID_SCUBA_850'] = cat_Za.data['ID_SCUBA_850'].astype(str)

# Ejecución del matching a partir de cat_Ke y junto con cat_Lu y cat_Za
cat_KeLuZa = cat_Ke.match_multi_catalogs(
    other_catalogs=[(cat_Lu, 'Lu'), (cat_Za, 'Za')],
    id_mappings=id_mappings,
    columns_to_keep=columns_to_keep )

# %%%% Verificaciones:

# %%%%% Verificación de las columnas del catálogo match
print("Final columns:", cat_KeLuZa.data.columns.tolist())
print("Number of rows:", len(cat_KeLuZa.data))  # Should still be 61
print("\n Last 5 rows:")
print(cat_KeLuZa.data.tail())


# Verificación rápida:
# Pick a few IDs from the merged catalog
sample_ids = cat_KeLuZa.data['ID_SCUBA'].head(3).tolist()

# Verificación/comparación de valores en los distintos catálogos
for id in sample_ids:
    print(f"\n--- ID: {id} ---")
    
    # Kevin
    print("Kevin (sd_120):", 
          cat_Ke.data.loc[cat_Ke.data['ID_SCUBA'] == id, 'sd_120'].values)
    
    # Luisa
    print("Luisa (M_barro, z):", 
          cat_Lu.data.loc[cat_Lu.data['ID_SCUBA_850'] == id, ['M_barro', 'z']].values)
    
    # Zavala
    print("Zavala (S850):", 
          cat_Za.data.loc[cat_Za.data['ID_SCUBA_850'] == id, 'S850'].values)

    # Catálogo fusionado
    print("Merged (M_barro, S850):", 
          cat_KeLuZa.data.loc[cat_KeLuZa.data['ID_SCUBA'] == id, ['M_barro', 'S850']].values)

# %%%%% Verificación de las fuentes a 450 μm

def integrate_450um_sources_from_kevin(cat_Ke, cat_Lu, cat_Za):
    """
    Integración del catálogo KeLuZa a partir del match de los IDs de Kevin
    con los de Luisa y Zavala, y contemplando los IDs en 850 y 450 μm
    """
    # Iniciamos con el catálogo original de Kevin
    integrated_df = cat_Ke.data.copy()
    
    # Creamos un marco de datos con las columnas de nuestro interés
    for col in ['M_barro', 'z', 'RA', 'Dec', 'S850', 'S450']:
        if col not in integrated_df.columns:
            integrated_df[col] = np.nan
    
    print(f"Iniciamos con {len(integrated_df)} fuentes del catálogo de Kevin")
    
    # Primer paso: intentamos corresponder las fuentes a 850μm (código original)
    for idx, row in integrated_df.iterrows():
        kevin_id = str(row['ID_SCUBA'])
        
        # Correspondencia con el catálogo de Luisa
        lu_match_850 = cat_Lu.data[cat_Lu.data['ID_SCUBA_850'].astype(str) == kevin_id]
        if len(lu_match_850) == 1:
            integrated_df.loc[idx, 'M_barro'] = lu_match_850['M_barro'].iloc[0]
            integrated_df.loc[idx, 'z'] = lu_match_850['z'].iloc[0]
            integrated_df.loc[idx, 'RA'] = lu_match_850['RA'].iloc[0]
            integrated_df.loc[idx, 'Dec'] = lu_match_850['Dec'].iloc[0]
        
        # Correspondencia con el catálogo de Zavala
        za_match_850 = cat_Za.data[cat_Za.data['ID_SCUBA_850'].astype(str) == kevin_id]
        if len(za_match_850) == 1:
            integrated_df.loc[idx, 'S850'] = za_match_850['S850'].iloc[0]
    
    # Segundo paso: intentamos correspondencia de fuentes a 450μm con las fuentes faltantes
    missing_data_mask = (
        pd.isna(integrated_df['M_barro']) | 
        pd.isna(integrated_df['z']) |
        pd.isna(integrated_df['RA']) | 
        pd.isna(integrated_df['Dec']) |
        pd.isna(integrated_df['S850'])
    )
    
    print(f"Se encontraron {missing_data_mask.sum()} fuentes para corresponder con 450μm")
    
    for idx in integrated_df[missing_data_mask].index:
        kevin_id = str(integrated_df.loc[idx, 'ID_SCUBA'])
        
        # Correspondencia con el catálogo de Luisa
        lu_match_450 = cat_Lu.data[cat_Lu.data['ID_SCUBA_450'].astype(str) == kevin_id]
        if len(lu_match_450) == 1:
            integrated_df.loc[idx, 'M_barro'] = lu_match_450['M_barro'].iloc[0]
            integrated_df.loc[idx, 'z'] = lu_match_450['z'].iloc[0]
            integrated_df.loc[idx, 'RA'] = lu_match_450['RA'].iloc[0]
            integrated_df.loc[idx, 'Dec'] = lu_match_450['Dec'].iloc[0]
        
        # Correspondencia con el catálogo de Zavala
        za_match_450 = cat_Za.data[cat_Za.data['ID_SCUBA_450'].astype(str) == kevin_id]
        if len(za_match_450) == 1:
            integrated_df.loc[idx, 'S450'] = za_match_450['S450'].iloc[0]
    
    return integrated_df

def create_KeLuZa_catalog(cat_Ke, integrated_df):
    """
    Create final catalog based on Kevin's original structure
    """
    # Create new instance preserving Kevin's catalog attributes
    final_cat = cat_Ke.__class__.__new__(cat_Ke.__class__)
    final_cat.__dict__ = {k: v for k, v in cat_Ke.__dict__.items()}
    
    # Use the integrated data but keep Kevin's original columns plus the new ones
    final_cat.data = integrated_df
    
    # Update metadata for the new columns
    new_columns_metadata = {
        'M_barro': {'units': 'M_sun', 'description': 'Stellar mass from Barro et al.'},
        'z': {'units': None, 'description': 'Redshift'},
        'RA': {'units': 'deg', 'description': 'Right Ascension'},
        'Dec': {'units': 'deg', 'description': 'Declination'},
        'S850': {'units': 'mJy/beam', 'description': 'Flux density at 850μm'},
        'S450': {'units': 'mJy/beam', 'description': 'Flux density at 450μm'}
    }
    
    for col, meta in new_columns_metadata.items():
        if col not in final_cat.columns:
            final_cat.columns[col] = meta
    
    return final_cat

# Usage - Start from cat_Ke instead of cat_KeLuZa
integrated_df = integrate_450um_sources_from_kevin(cat_Ke, cat_Lu, cat_Za)
cat_KeLuZa = create_KeLuZa_catalog(cat_Ke, integrated_df)


def verify_kevin_integration(cat_Ke, cat_KeLuZa):
    """
    Verificación del catálogo integrado KeLuZa
    """
    print("=== Verificación de la integración basada en cat_Ke ===")
    print(f"Tamaño original de cat_Ke: {len(cat_Ke.data)}")
    print(f"Tamaño del catálogo final integrado: {len(cat_KeLuZa.data)}")
    
    # Chequeo de lo que se añadió
    print("\n Columnas originales de Kevin:", list(cat_Ke.data.columns))
    print("Columnas finales del catálogo:", list(cat_KeLuZa.data.columns))
    print("Final catalog columns:", list(cat_KeLuZa.data.columns))
    
    # Check data completeness
    print("\n Completitud en los datos:")
    for col in ['M_barro', 'z', 'RA', 'Dec', 'S850', 'S450']:
        if col in cat_KeLuZa.data.columns:
            complete = cat_KeLuZa.data[col].notna().sum()
            total = len(cat_KeLuZa.data)
            print(f"{col}: {complete}/{total} ({complete/total*100:.1f}%)")
    
    # Analyze wavelength distribution
    has_s850 = pd.notna(cat_KeLuZa.data.get('S850', pd.Series()))
    has_s450 = pd.notna(cat_KeLuZa.data.get('S450', pd.Series()))
    
    print(f"\n fuentes a 850μm: {has_s850.sum()}")
    print(f"fuentes a 450μm: {has_s450.sum()}")
    print(f"fuentes a ambas μm: {(has_s850 & has_s450).sum()}")
    print(f"fuentes en ninguna μm: {(~has_s850 & ~has_s450).sum()}")

# Verificación
verify_kevin_integration(cat_Ke, cat_KeLuZa)

# Estructura final
print("\n=== FINAL CATALOG SAMPLE ===")
print(cat_KeLuZa.data.head()[['ID_SCUBA', 'sd_120', 'e_sd120', 'M_barro', 'z', 'S850', 'S450']])

def convert_450_to_850_flux_and_rename(catalog, alpha=3.5):
    """
    Convert 450μm flux to equivalent 850μm flux for sources missing 850μm data,
    and rename M_barro to M_star.
    
    Parameters:
    -----------
    catalog : Cat_Obs
        Input catalog with S450 and/or S850 columns
    alpha : float
        Spectral index for conversion: S_850 = S_450 * (450/850)^(α-3)
    
    Returns:
    --------
    catalog_new : Cat_Obs
        New catalog with renamed column and converted fluxes
    """
    print(f"=== CONVERTING 450μm → 850μm FLUXES (α={alpha}) ===")
    
    # ✅ Step 1: Copy the data DataFrame only
    data = catalog.data.copy()
    
    # ✅ Step 2: Rename M_barro to M_star
    if 'M_barro' in data.columns:
        data = data.rename(columns={'M_barro': 'M_star'})
        print("Renamed column: M_barro → M_star")
    
    # ✅ Step 3: Identify sources with S450 but missing S850
    has_s450 = pd.notna(data['S450']) if 'S450' in data.columns else False
    missing_s850 = pd.isna(data['S850']) if 'S850' in data.columns else True
    
    if isinstance(has_s450, bool):
        has_s450 = pd.Series([has_s450] * len(data), index=data.index)
    if isinstance(missing_s850, bool):
        missing_s850 = pd.Series([missing_s850] * len(data), index=data.index)
    
    convert_mask = has_s450 & missing_s850
    n_converted = convert_mask.sum()
    
    print(f"Converting {n_converted} sources from S450 → S850")
    
    # ✅ Step 4: Apply conversion
    if n_converted > 0:
        conversion_factor = (450 / 850) ** (alpha - 3)
        print(f"Using conversion factor: {conversion_factor:.3f} (α = {alpha})")
        
        # Only convert where needed
        data.loc[convert_mask, 'S850'] = data.loc[convert_mask, 'S450'] * conversion_factor
        
        # Add source flag
        if 'S850_source' not in data.columns:
            data['S850_source'] = 'direct_measurement'
        data.loc[convert_mask, 'S850_source'] = f'converted_from_450_α{alpha}'
    
    # ✅ Step 5: Create new Cat_Obs instance (do NOT use .copy() on object)
    cat_new = catalog.__class__.__new__(catalog.__class__)
    cat_new.__dict__ = {k: v.copy() if hasattr(v, 'copy') else v 
                        for k, v in catalog.__dict__.items()}
    
    # Replace data with modified version
    cat_new.data = data
    
    # ✅ Step 6: Update metadata for renamed column
    if 'M_star' in data.columns and 'M_barro' in cat_new.columns:
        cat_new.columns['M_star'] = cat_new.columns.pop('M_barro')
        cat_new.columns['M_star']['description'] = 'Stellar mass (from Barro et al.)'
    
    # Add metadata for new S850_source column
    cat_new.columns['S850_source'] = {
        'units': None,
        'description': f'Source of S850: direct or converted from S450 using α={alpha}'
    }
    
    # Recompute stats after modification
    cat_new.compute_stats()
    
    print("✅ Conversion and renaming completed.")
    return cat_new


# Run the conversion
cat_KeLuZa_conv = convert_450_to_850_flux_and_rename(cat_KeLuZa, alpha=3.5)


# %%%% Sobredensidades 2D

def comp_sd_mc_2D(cat_ce, cat_camp, rad_deg, geometry=None, ra_range=None, dec_range=None, n_mc=1000):
    """
    Args:
        cat_ce: Catálogo de fuentes centrales (debe contener columnas 'RA', 'Dec')
        cat_camp: Catálogo de campo para densidad de fondo
        rad_deg: Radio de búsqueda en grados
        geometry: Objeto FieldGeometry (para campos no rectangulares)
        ra_range: Tuple (min, max) para campos rectangulares
        dec_range: Tuple (min, max) para campos rectangulares
        n_mc: Número de iteraciones Monte Carlo para corrección de bordes
    """
    # Validación de parámetros
    if geometry is None and (ra_range is None or dec_range is None):
        raise ValueError("Debe proporcionar geometry o ra_range/dec_range")

    # Coordenadas
    centros = cat_ce.data[['RA', 'Dec']].values
    camp = cat_camp.data[['RA', 'Dec']].values

    # Asegurar que exista columna 'ID' en cat_ce
    if 'ID' not in cat_ce.data.columns:
        cat_ce.data['ID'] = [f'SRC_{i:04d}' for i in range(len(cat_ce.data))]

    # KD-Tree para búsqueda eficiente
    tree = cKDTree(camp)

    # Calcular densidad de fondo UNA VEZ
    total_area = geometry.calculate_area() if geometry else (ra_range[1] - ra_range[0]) * (dec_range[1] - dec_range[0])
    n_fondo = len(camp) / total_area  # galaxias/deg²

    resultados = []
    for idx, (ra, dec) in enumerate(centros):
        src_id = cat_ce.data.iloc[idx]['ID']  # Conservar ID existente o usar el nuevo

        # Detección de bordes (optimizado para geometrías rectangulares/complejas)
        if geometry is not None:
            dec_min = geometry.lower_bound(ra)
            dec_max = geometry.upper_bound(ra)
            ra_min = min(geometry.lower[:, 0])
            ra_max = max(geometry.lower[:, 0])
            near_edge = (ra - rad_deg < ra_min or ra + rad_deg > ra_max or
                         dec - rad_deg < dec_min or dec + rad_deg > dec_max)
        else:
            near_edge = (ra - rad_deg < ra_range[0] or ra + rad_deg > ra_range[1] or
                         dec - rad_deg < dec_range[0] or dec + rad_deg > dec_range[1])

        # Cálculo de área efectiva y conteo
        if not near_edge:
            count = tree.query_ball_point([ra, dec], r=rad_deg, return_length=True)
            area_eff = np.pi * rad_deg**2
        else:
            n_inside = 0
            for _ in range(n_mc):
                theta = 2 * np.pi * np.random.rand()
                r_rand = rad_deg * np.sqrt(np.random.rand())
                dra = r_rand * np.cos(theta)
                ddec = r_rand * np.sin(theta)
                ra_rand = ra + dra / np.cos(np.radians(dec))
                dec_rand = dec + ddec

                # Ajuste para coordenadas en el límite RA=0/360
                if ra_rand < 0: ra_rand += 360
                if ra_rand > 360: ra_rand -= 360

                if geometry:
                    in_field = geometry.contains(ra_rand, dec_rand)
                else:
                    in_field = (ra_range[0] <= ra_rand <= ra_range[1] and 
                               dec_range[0] <= dec_rand <= dec_range[1])

                n_inside += 1 if in_field else 0

            area_eff = (n_inside / n_mc) * np.pi * rad_deg**2
            count = tree.query_ball_point([ra, dec], r=rad_deg, return_length=True)

        n_expected = n_fondo * area_eff
        overdensity = (count - n_expected) / n_expected if n_expected > 0 else 0

        resultados.append({
            'ID': src_id,  # Identificador único
            'RA': ra,
            'Dec': dec,
            'N_obs': count,
            'N_esp': n_expected,
            'sd': overdensity,
            'Area_eff': area_eff
        })

    # Crear nuevo catálogo con resultados
    new_cat = cat_ce.__class__.__new__(cat_ce.__class__)
    new_cat.__dict__ = {k: v for k, v in cat_ce.__dict__.items()}
    
    # Combinar datos originales + resultados (usando 'ID' como clave)
    results_df = pd.DataFrame(resultados)
    new_cat.data = pd.merge(
        cat_ce.data,
        results_df.drop(['RA', 'Dec'], axis=1),
        on='ID',
        how='left'
    )

    new_cat.columns.update({
    'N_obs': {'units': None, 'descripción': 'Núm. obs. de galaxias dentro del radio'},
    'N_esp': {'units': None, 'descripción': 'Núm. esp de galaxias según densidad de fondo'},
    'sd': {'units': None, 'descripción': 'Sobredensidad (N_obs - N_esp)/N_esp'},
    'Area_eff': {'units': 'deg²', 'descripción': 'Área efectiva corregida por bordes'}
})


    return new_cat


cat_sd2D_AA = comp_sd_mc_2D(cat_Ar_zmtch, cat_Al_comp_St, r_deg, ra_range=ra_range, dec_range=dec_range, n_mc=1000)
# cat_sd2D_AA = comp_sd_mc_2D(cat_Ar_cuts, cat_Al_cuts, r_deg, ra_range=ra_range, dec_range=dec_range, n_mc=1000)



# %%%%% 

# %%% Simulaciones
# Siempre correr esta variable antes de obtener las rutas a los distintos archivos
config = ConfigManager()

ra_range, dec_range = (268.85, 271.15), (-1.15, 1.15)
ra_span = ra_range[1] - ra_range[0]
dec_span = dec_range[1] - dec_range[0]
dec_mid = 0.0

# %%%% Simulación: Araceli 5.3deg2
# Carga de la simulación de Araceli
cat_Ar = Cat(
    config=config,
    catalog_name='araceli',      # Key under 'simulations' in config.yaml
    force_log_mass=True,         # Convert masses to log10 if needed
    is_simulation=True )         # Critical for simulation data paths

# Primeros cortes: masa, flujo y corrimientos
cuts_Ar = [
    ('M_star', M_ce, 11.9),     # rango de masas (log10)
    ('mu_Flux_850', S_ce, 8.5), # rango de flujos (mJy)
    ('z', St_cts_z_mn-2*St_cts_z_std, St_cts_z_mn+2*St_cts_z_std) ] # rango de z
# Araceli_simu 1er corte
cat_Ar_cuts = cat_Ar.apply_cuts(cuts_Ar)

#### Criterio de completitud (Zavala):
# Creación de la función de interpolación (extrapole con precaución)
def load_comp_func_Za(config, name='zavala_850'):
    """Carga de la curva de completez y regresando la función de interpolación"""
    comp_path = config.get_completeness_path(name)
    comp_data = pd.read_csv(comp_path, header=None)
    return interp1d(comp_data[0], comp_data[1], 
                   bounds_error=False, 
                   fill_value=(0.0, 1.0))  # Clips to [0,1]
def ap_comp_Za(catalog, comp_func, flux_column='mu_Flux_850', rng_seed=None):
    """Aplicación de la función de completez al catálogo"""
    if rng_seed is not None:
        np.random.seed(rng_seed)
    
    probs = comp_func(catalog.data[flux_column]).clip(0, 1)
    mask = np.random.rand(len(probs)) <= probs
    
    filtered_catalog = catalog.__class__.__new__(catalog.__class__)
    filtered_catalog.__dict__ = {k: v for k, v in catalog.__dict__.items()}
    filtered_catalog.data = catalog.data[mask].copy()
    return filtered_catalog
comp_func_Za = load_comp_func_Za(config, 'zavala_850')
cat_Ar_comp_Za = ap_comp_Za(cat_Ar_cuts, comp_func_Za)

#### Mimetización de la distribución de corrimientos (Luisa):
def ap_z_dist(catalog, reference_catalog, z_column='z', n_bins=20, rng_seed=None):
    """
    Filtra el catálogo para que siga la distribución de corrimientos de referencia.
    
    Args:
        catalog (Cat): Catálogo a filtrar (ej. cat_Ar_cuts)
        reference_catalog (Cat): Catálogo de referencia (ej. cat_KeLuZa)
        z_column (str): Nombre de la columna de redshift
        n_bins (int): Número de bins para el histograma
        rng_seed (int): Semilla para reproducibilidad
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)
    
    # 1. Calcular histograma normalizado de la referencia
    counts, bins = np.histogram(reference_catalog.data[z_column], bins=n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts_norm = counts / counts.max()  # Normalizar a [0,1]
    
    # 2. Crear función de interpolación
    z_func = interp1d(bin_centers, counts_norm, 
                     bounds_error=False, 
                     fill_value=(0, 1))  # Extrapolar conservadoramente
    
    # 3. Calcular probabilidades para cada fuente
    probs = z_func(catalog.data[z_column]).clip(0, 1)
    
    # 4. Muestreo aleatorio basado en las probabilidades
    mask = np.random.rand(len(probs)) <= probs
    
    # 5. Crear nuevo catálogo filtrado (manteniendo metadatos)
    filtered_catalog = catalog.__class__.__new__(catalog.__class__)
    filtered_catalog.__dict__ = {k: v for k, v in catalog.__dict__.items()}
    filtered_catalog.data = catalog.data[mask].copy()
    
    return filtered_catalog

# Aplicar distribución de corrimientos de cat_Lu a cat_Ar
cat_Ar_zmtch = ap_z_dist(
    catalog=cat_Ar_comp_Za,      # Catálogo ya filtrado por completitud
    reference_catalog=cat_Lu,      # Catálogo de referencia (Luisa)
    z_column='z',                  # Columna con redshift
    n_bins=10,                     # Número de bins para el histograma
    rng_seed=42  )                 # Para reproducibilidad


# %%%% Simulación: Aldo 5.3deg2
# Carga de la simulación de Aldo
cat_Al = Cat(
    config=config,
    catalog_name='aldo',
    force_log_mass=True,
    is_simulation=True)

# Primeros cortes: masa, flujo y corrimientos
cuts_Al = [
    ('M_star', M_ve, 11.9),     # rango de masas (log10)
    ('z', St_cts_z_med-2*St_cts_z_std, St_cts_z_med+2*St_cts_z_std) ] # rango de z
# Aldo_simu 1er corte
cat_Al_cuts = cat_Al.apply_cuts(cuts_Al)

#### Criterio de completitud (Stefanon):
# Creación de la función de interpolación
def load_2d_comp_St(config, name_50='stefanon_50comp', name_90='stefanon_90comp'):
    """Carga de las curvas de compltetitud al 50 y 90 % y creación de las curvas de interpolación"""
    # Obtención de las rutas de los archivos
    base_path = os.path.expanduser(config.config['paths']['completeness_data']['stefanon_2d'])
    file_50 = os.path.join(base_path, config.config['files']['completeness_data'][name_50] + '.csv')
    file_90 = os.path.join(base_path, config.config['files']['completeness_data'][name_90] + '.csv')
    
    # Carga de datos
    data_50 = pd.read_csv(file_50, header=None)
    data_90 = pd.read_csv(file_90, header=None)
    
    # Creación de las funciones de interpolación
    M50_func = interp1d(data_50[0], data_50[1], bounds_error=False, fill_value='extrapolate')
    M90_func = interp1d(data_90[0], data_90[1], bounds_error=False, fill_value='extrapolate')
    
    return M50_func, M90_func

def calc_2d_prob_St(z, M, M50_func, M90_func, z_min, z_max, dz=0.05):
    """Cálculo de la probabilidad de completitud para cada fuente"""
    # Discretización de los corrimientos
    z_bin = np.clip(((z - z_min) / dz).astype(int), 0, int((z_max-z_min)/dz)-1)
    
    # Obtención del umbral de las masas para el bin de corrimientos seleccionado
    M50 = M50_func(z_min + z_bin*dz)
    M90 = M90_func(z_min + z_bin*dz)
    
    # Interpolación lineal entre el 50 y 90 % de la completitud
    prob = np.where(
        M <= M50, 
        0.5 + 0.4*(M - M50)/(M90 - M50),  # Linear between thresholds
        np.where(M <= M90, 0.9, 1.0)      # Above 90% threshold
    )
    return np.clip(prob, 0, 1)

### Aplicación al catálogo:

# Carga de las funciones de completitud
M50_func, M90_func = load_2d_comp_St(config)
# Obtención de los rangos de corrimientos a partir de las curvas de completitud
z_min = min(M50_func.x.min(), M90_func.x.min())
z_max = max(M50_func.x.max(), M90_func.x.max())

def ap_2d_comp_St(catalog, M50_func, M90_func, z_min, z_max, 
                         z_column='z', M_column='M_star', rng_seed=None):
    """Apply 2D completeness selection"""
    if rng_seed is not None:
        np.random.seed(rng_seed)
    
    # Calculate probabilities
    probs = calc_2d_prob_St(
        catalog.data[z_column].values,
        catalog.data[M_column].values,
        M50_func, M90_func, z_min, z_max
    )
    
    # Random selection
    mask = np.random.rand(len(probs)) <= probs
    
    # Create filtered catalog
    filtered_catalog = catalog.__class__.__new__(catalog.__class__)
    filtered_catalog.__dict__ = {k: v for k, v in catalog.__dict__.items()}
    filtered_catalog.data = catalog.data[mask].copy()
    
    return filtered_catalog

# Aplicación al catálogo de Aldo
cat_Al_comp_St = ap_2d_comp_St(
    cat_Al_cuts, 
    M50_func, M90_func, 
    z_min, z_max
)


# %%%% Sobredensidades 2D

def comp_sd_mc_2D(cat_ce, cat_camp, rad_deg, geometry=None, ra_range=None, dec_range=None, n_mc=1000):
    """
    Args:
        cat_ce: Catálogo de fuentes centrales (debe contener columnas 'RA', 'Dec')
        cat_camp: Catálogo de campo para densidad de fondo
        rad_deg: Radio de búsqueda en grados
        geometry: Objeto FieldGeometry (para campos no rectangulares)
        ra_range: Tuple (min, max) para campos rectangulares
        dec_range: Tuple (min, max) para campos rectangulares
        n_mc: Número de iteraciones Monte Carlo para corrección de bordes
    """
    # Validación de parámetros
    if geometry is None and (ra_range is None or dec_range is None):
        raise ValueError("Debe proporcionar geometry o ra_range/dec_range")

    # Coordenadas
    centros = cat_ce.data[['RA', 'Dec']].values
    camp = cat_camp.data[['RA', 'Dec']].values

    # Asegurar que exista columna 'ID' en cat_ce
    if 'ID' not in cat_ce.data.columns:
        cat_ce.data['ID'] = [f'SRC_{i:04d}' for i in range(len(cat_ce.data))]

    # KD-Tree para búsqueda eficiente
    tree = cKDTree(camp)

    # Calcular densidad de fondo UNA VEZ
    total_area = geometry.calculate_area() if geometry else (ra_range[1] - ra_range[0]) * (dec_range[1] - dec_range[0])
    n_fondo = len(camp) / total_area  # galaxias/deg²

    resultados = []
    for idx, (ra, dec) in enumerate(centros):
        src_id = cat_ce.data.iloc[idx]['ID']  # Conservar ID existente o usar el nuevo

        # Detección de bordes (optimizado para geometrías rectangulares/complejas)
        if geometry is not None:
            dec_min = geometry.lower_bound(ra)
            dec_max = geometry.upper_bound(ra)
            ra_min = min(geometry.lower[:, 0])
            ra_max = max(geometry.lower[:, 0])
            near_edge = (ra - rad_deg < ra_min or ra + rad_deg > ra_max or
                         dec - rad_deg < dec_min or dec + rad_deg > dec_max)
        else:
            near_edge = (ra - rad_deg < ra_range[0] or ra + rad_deg > ra_range[1] or
                         dec - rad_deg < dec_range[0] or dec + rad_deg > dec_range[1])

        # Cálculo de área efectiva y conteo
        if not near_edge:
            count = tree.query_ball_point([ra, dec], r=rad_deg, return_length=True)
            area_eff = np.pi * rad_deg**2
        else:
            n_inside = 0
            for _ in range(n_mc):
                theta = 2 * np.pi * np.random.rand()
                r_rand = rad_deg * np.sqrt(np.random.rand())
                dra = r_rand * np.cos(theta)
                ddec = r_rand * np.sin(theta)
                ra_rand = ra + dra / np.cos(np.radians(dec))
                dec_rand = dec + ddec

                # Ajuste para coordenadas en el límite RA=0/360
                if ra_rand < 0: ra_rand += 360
                if ra_rand > 360: ra_rand -= 360

                if geometry:
                    in_field = geometry.contains(ra_rand, dec_rand)
                else:
                    in_field = (ra_range[0] <= ra_rand <= ra_range[1] and 
                               dec_range[0] <= dec_rand <= dec_range[1])

                n_inside += 1 if in_field else 0

            area_eff = (n_inside / n_mc) * np.pi * rad_deg**2
            count = tree.query_ball_point([ra, dec], r=rad_deg, return_length=True)

        n_expected = n_fondo * area_eff
        overdensity = (count - n_expected) / n_expected if n_expected > 0 else 0

        resultados.append({
            'ID': src_id,  # Identificador único
            'RA': ra,
            'Dec': dec,
            'N_obs': count,
            'N_esp': n_expected,
            'sd': overdensity,
            'Area_eff': area_eff
        })

    # Crear nuevo catálogo con resultados
    new_cat = cat_ce.__class__.__new__(cat_ce.__class__)
    new_cat.__dict__ = {k: v for k, v in cat_ce.__dict__.items()}
    
    # Combinar datos originales + resultados (usando 'ID' como clave)
    results_df = pd.DataFrame(resultados)
    new_cat.data = pd.merge(
        cat_ce.data,
        results_df.drop(['RA', 'Dec'], axis=1),
        on='ID',
        how='left'
    )

    new_cat.columns.update({
    'N_obs': {'units': None, 'descripción': 'Núm. obs. de galaxias dentro del radio'},
    'N_esp': {'units': None, 'descripción': 'Núm. esp de galaxias según densidad de fondo'},
    'sd': {'units': None, 'descripción': 'Sobredensidad (N_obs - N_esp)/N_esp'},
    'Area_eff': {'units': 'deg²', 'descripción': 'Área efectiva corregida por bordes'}
})


    return new_cat


cat_sd2D_AA = comp_sd_mc_2D(cat_Ar_zmtch, cat_Al_comp_St, r_deg, ra_range=ra_range, dec_range=dec_range, n_mc=1000)
# cat_sd2D_AA = comp_sd_mc_2D(cat_Ar_cuts, cat_Al_cuts, r_deg, ra_range=ra_range, dec_range=dec_range, n_mc=1000)

# %%%% Sobredensidades 3D

st = time.time()

cosmo_utils = CosmologyUtils()
H_dist = cosmo_utils.hubble_distance

def DC(z):
    return cosmo_utils.comoving_distance(z)

def exact_comoving_volume(z1, z2, ra_span, dec_span, cosmo, cat_camp):
    """
    Calculate exact comoving volume of a pyramid slice
    For a comoving volume differential element:
        
        dV = DC^2 · dDC/dz dz · dΩ
    
    where dΩ = ∫∫ cos(Dec) dRA dDec = RA_f - RA_i · (sin(Dec_f) - sin(Dec_i))
    """
    # Convert angular spans to radians
    Δra = np.radians(ra_span)
    Δsin_dec = np.sin(np.radians(dec_range[1])) - np.sin(np.radians(dec_range[0]))
    ΔΩ = Δra * Δsin_dec
    
    # dec_mid = 0.5 * (cat_camp.data['Dec'].min() + cat_camp.data['Dec'].max())
    
    # Integrand: dV/dz = (dc/dz) * dc² * Δra * cos(dec)
    def integrand(z):
        # dc = cosmo.comoving_distance(z)  # Already in Mpc
        dcdz = H_dist / cosmo.efunc(z)  # dc/dz = c/(H(z))
        return dcdz * DC(z)**2 * ΔΩ
    
    # Integrate from z1 to z2
    volume, _ = quad(integrand, z1, z2)
    return volume


def comp_sd_mc_3d(cat_ce, cat_camp, r_Mpc, cosmo, ra_range=None, dec_range=None, n_bins=10, log_bins=True, n_mc=1000):
    """
    Compute 3D overdensities in comoving space with exact volume calculation
    """
    
    # 1. Create redshift bins
    z_min = min(cat_ce.data['z'].min(), cat_camp.data['z'].min())
    z_max = max(cat_ce.data['z'].max(), cat_camp.data['z'].max())
    
    if log_bins:
        z_bins = np.logspace(np.log10(z_min), np.log10(z_max), n_bins+1)
    else:
        z_bins = np.linspace(z_min, z_max, n_bins+1)

    # 2. Calculate survey geometry
    if ra_range and dec_range:
        ra_span = ra_range[1] - ra_range[0]
        dec_span = dec_range[1] - dec_range[0]
    else:
        ra_span = cat_camp.data['RA'].max() - cat_camp.data['RA'].min()
        dec_span = cat_camp.data['Dec'].max() - cat_camp.data['Dec'].min()
    
    # 3. Precompute exact comoving volumes for each bin
    bin_volumes = []
    for i in range(len(z_bins)-1):
        z1, z2 = z_bins[i], z_bins[i+1]
        vol = exact_comoving_volume(z1, z2, ra_span, dec_span, cosmo, cat_camp)
        bin_volumes.append(vol)
    
    # 4. Calculate mean density in each bin
    bin_densities = []
    for i in range(len(z_bins)-1):
        mask = (cat_camp.data['z'] >= z_bins[i]) & (cat_camp.data['z'] < z_bins[i+1])
        n_gal = mask.sum()
        bin_densities.append(n_gal / bin_volumes[i] if bin_volumes[i] > 0 else 0)

    # 5. Convert all galaxies to comoving coordinates
    coords_ce = np.array([cosmo.redshift_to_comoving(row['RA'], row['Dec'], row['z']) 
                         for _, row in cat_ce.data.iterrows()])
    coords_camp = np.array([cosmo.redshift_to_comoving(row['RA'], row['Dec'], row['z']) 
                           for _, row in cat_camp.data.iterrows()])
    
    # 6. Build KD-tree for efficient neighbor searches and sphere volume
    tree = cKDTree(coords_camp)
    sphere_volume = (4/3) * np.pi * (r_Mpc**3)

    # 7. Compute overdensities for each central galaxy
    resultados = []
    zero_esp_count = 0
    
    for idx in range(len(cat_ce.data)):
        row = cat_ce.data.iloc[idx]
        ra, dec, z = row['RA'], row['Dec'], row['z']
        
        # Find appropriate redshift bin
        z_bin = np.searchsorted(z_bins, z) - 1
        z_bin = max(0, min(z_bin, len(bin_densities)-1))
        mean_density = bin_densities[z_bin]
        
        # Get 3D position
        coord = coords_ce[idx]
        
        # Count neighbors within sphere
        count = tree.query_ball_point(coord, r=r_Mpc, return_length=True)
        
        # Calculate expected number
        sphere_volume = (4/3) * np.pi * (r_Mpc**3)
        expected = mean_density * sphere_volume
        
        # --- Fix for N_esp = 0 cases ---
        if expected == 0:
            overdensity = np.nan
            zero_esp_count += 1
        else:
            overdensity = (count - expected) / expected
        
        # Store results (matching 2D function format)
        resultados.append({
            'ID': row['ID'],
            'RA': ra,
            'Dec': dec,
            'N_obs': count,
            'N_esp': expected,
            'sd': overdensity,
            'Volume_eff': sphere_volume,
            'z_bin': z_bin,
            'bin_vol': bin_volumes[z_bin] if z_bin < len(bin_volumes) else np.nan
            # 'bin_vol': bin_volumes[i]
        })

    print(f"Found {zero_esp_count} sources with N_esp = 0 (δ set to NaN)")

    # 8. Create output catalog
    new_cat = cat_ce.__class__.__new__(cat_ce.__class__)
    new_cat.__dict__ = {k: v for k, v in cat_ce.__dict__.items()}
    
    # Merge results
    results_df = pd.DataFrame(resultados)
    new_cat.data = pd.merge(
        cat_ce.data,
        results_df.drop(['RA', 'Dec'], axis=1),
        on='ID',
        how='left'
    )
    
    # Update metadata
    new_cat.columns.update({
        'N_obs': {'units': None, 'description': 'Number of neighbors within r_Mpc'},
        'N_esp': {'units': None, 'description': 'Expected number from mean density'},
        'sd': {'units': None, 'description': 'Overdensity (N_obs - N_esp)/N_esp (Nan if N_esp=0)'},
        'Volume_eff': {'units': 'Mpc³', 'description': 'Search sphere volume'},
        'z_bin': {'units': None, 'description': 'Redshift bin index'},
        'bin_vol': {'units': 'Mpc3', 'description': 'Volumen de la rebanada en z'}
    })
    
    
    
    return new_cat, bin_volumes




cat_sd3D_AA_log, volume_bins_log = comp_sd_mc_3d(
    cat_ce=cat_Ar_zmtch,
    cat_camp=cat_Al_comp_St, 
    r_Mpc=r_Mpc,
    cosmo=cosmo_utils
)

cat_sd3D_AA_lin, volume_bins = comp_sd_mc_3d(
    cat_ce=cat_Ar_zmtch,
    cat_camp=cat_Al_comp_St,
    r_Mpc=r_Mpc,
    cosmo=cosmo_utils,
    n_bins=int((DC(cat_Al_comp_St.data['z'].max())-DC(cat_Al_comp_St.data['z'].min()))/(r_Mpc*5)),
    log_bins=False)

# Sin cortes
# cat_sd3D_AA_log, volume_bins_log = comp_sd_mc_3d(
#     cat_ce=cat_Ar_cuts,
#     cat_camp=cat_Al_cuts, 
#     r_Mpc=r_Mpc,
#     cosmo=cosmo_utils
# )

# cat_sd3D_AA_lin, volume_bins = comp_sd_mc_3d(
#     cat_ce=cat_Ar_cuts,
#     cat_camp=cat_Al_cuts,
#     r_Mpc=r_Mpc,
#     cosmo=cosmo_utils,
#     n_bins=int((DC(cat_Al_comp_St.data['z'].max())-DC(cat_Al_comp_St.data['z'].min()))/(r_Mpc*5)),
#     log_bins=False)


print('t = '+str(round(time.time()-st),))

# %%%% Verificaciones de valores nulos y negativos

#### Valores 0

def verify_fix(catalog, name=""):
    """
    Verify that the δ=0 artifact has been fixed
    """
    data = catalog.data
    zero_mask = data['sd'] == 0
    nan_mask = data['sd'].isna()
    esp_zero_mask = data['N_esp'] == 0
    
    print(f"\n=== {name} ===")
    print(f"Total sources: {len(data)}")
    print(f"Sources with δ = 0: {zero_mask.sum()} ({zero_mask.sum()/len(data)*100:.1f}%)")
    print(f"Sources with δ = NaN: {nan_mask.sum()} ({nan_mask.sum()/len(data)*100:.1f}%)")
    print(f"Sources with N_esp = 0: {esp_zero_mask.sum()} ({esp_zero_mask.sum()/len(data)*100:.1f}%)")
    
    # Check if the fix worked: all N_esp=0 should have δ=NaN
    if esp_zero_mask.sum() > 0:
        correctly_fixed = nan_mask[esp_zero_mask].sum()
        print(f"Correctly fixed: {correctly_fixed}/{esp_zero_mask.sum()} "
              f"({correctly_fixed/esp_zero_mask.sum()*100:.1f}%) of N_esp=0 have δ=NaN")

verify_fix(cat_sd3D_AA_lin, "3D Overdensity (After Fix)")

#### Valores -1
def diagnose_negative_overdensities(cat_ce, cat_camp, sd_catalog, r_Mpc=3.0):
    """
    Verify why some overdensities are exactly -1
    """
    data = sd_catalog.data
    neg_mask = data['sd'] == -1
    
    print(f"Found {neg_mask.sum()} sources with δ = -1")
    
    if neg_mask.sum() > 0:
        neg_sources = data[neg_mask]
        
        for idx, row in neg_sources.iterrows():
            source_id = row['ID']
            
            # Check if central source exists in field catalog
            in_ce = source_id in cat_ce.data['ID'].values
            in_camp = source_id in cat_camp.data['ID'].values
            
            print(f"\nSource {source_id}:")
            print(f"  In central catalog (Araceli): {in_ce}")
            print(f"  In field catalog (Aldo): {in_camp}")
            print(f"  N_obs = {row['N_obs']}, N_esp = {row['N_esp']:.4f}")
            
            if not in_camp:
                print("  → CONFIRMED: Central source missing from field catalog!")
                print("  → This causes N_obs = 0 but N_esp > 0 → δ = -1")
            
            # Check a few examples only
            if idx > 2:  # Limit output
                print("  ... (showing first 3 examples only)")
                break

diagnose_negative_overdensities(cat_Ar_zmtch, cat_Al_comp_St, cat_sd3D_AA_lin)


# %%%% Valores extremos

st = time.time()

def redshift_to_comoving(ra, dec, z, cosmo):
    """Convert (RA, Dec, z) to comoving coordinates (Mpc)"""
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    dc = cosmo.comoving_distance(z)  # Mpc
    
    x = dc * np.cos(dec_rad) * np.cos(ra_rad)
    y = dc * np.cos(dec_rad) * np.sin(ra_rad)
    z_coord = dc * np.sin(dec_rad)
    return np.array([x, y, z_coord])

def calculate_bin_densities(cat_camp, cosmo, z_bins):
    """Calculate mean galaxy density for each redshift bin"""
    bin_densities = []
    bin_volumes = []
    
    ra_span = cat_camp.data['RA'].max() - cat_camp.data['RA'].min()
    dec_span = cat_camp.data['Dec'].max() - cat_camp.data['Dec'].min()
    
    for i in range(len(z_bins)-1):
        z1, z2 = z_bins[i], z_bins[i+1]
        
        # Count galaxies in this redshift bin
        mask = (cat_camp.data['z'] >= z1) & (cat_camp.data['z'] < z2)
        n_gal = mask.sum()
        
        # Calculate exact volume
        vol = exact_comoving_volume(z1, z2, ra_span, dec_span, cosmo, cat_camp)
        
        bin_volumes.append(vol)
        bin_densities.append(n_gal / vol if vol > 0 else 0)
    
    return bin_densities, bin_volumes

cat_ce = cat_Ar_zmtch
cat_camp = cat_Al_comp_St
def nbins_lin(n_bins=10, log_bins=True):
    z_min = min(cat_ce.data['z'].min(), cat_camp.data['z'].min())
    z_max = max(cat_ce.data['z'].max(), cat_camp.data['z'].max())
    if log_bins:
        return np.logspace(np.log10(z_min), np.log10(z_max), n_bins+1)
    else:
        return np.linspace(z_min, z_max, n_bins+1)

def mpc_to_delta_z(r_mpc, z_reference, cosmo):
    """
    Convert physical distance (Mpc) to redshift interval around a reference z
    
    Args:
        r_mpc: Distance in Mpc
        z_reference: Central redshift
        cosmo: Cosmology instance
    """
    H_z = cosmo.H(z_reference).value  # Hubble parameter at z (km/s/Mpc)
    c = 3e5  # Speed of light (km/s)
    
    # Δz ≈ Δr * H(z) / c
    delta_z = r_mpc * H_z / c
    return delta_z


# Get extreme OD from LIN-binned catalog  

z_bins_lin = nbins_lin(n_bins = int((DC(cat_camp.data['z'].max())-DC(cat_camp.data['z'].min()))/r_Mpc), log_bins=False)
bin_densities_lin, bin_volumes_lin = calculate_bin_densities(cat_Al_comp_St,cosmo_utils,z_bins_lin)

extreme_row_lin = cat_sd3D_AA_lin.data.loc[cat_sd3D_AA_lin.data['sd'].idxmax()]
extreme_id_lin = extreme_row_lin['ID']
extreme_z_lin = extreme_row_lin['z']

# Get the redshift bin boundaries
z_bin_idx_lin = extreme_row_lin['z_bin']
z_min_bin_lin = z_bins_lin[z_bin_idx_lin]
z_max_bin_lin = z_bins_lin[z_bin_idx_lin + 1]

print(f"Extreme OD in lin bin {z_bin_idx_lin}: z={extreme_z_lin:.3f} (bin: {z_min_bin_lin:.3f}-{z_max_bin_lin:.3f})")

delta_z_lin = mpc_to_delta_z(r_Mpc, extreme_z_lin, cosmo)
print(f"±{r_Mpc} Mpc corresponds to ±{delta_z_lin:.4f} in redshift around z={extreme_z_lin:.3f}")

# Filter to this bin only
mask_slice_lin = (cat_Al_comp_St.data['z'] >= z_min_bin_lin - delta_z_lin) & \
                 (cat_Al_comp_St.data['z'] < z_max_bin_lin + delta_z_lin)
coords_slice_lin = np.array([
    redshift_to_comoving(row['RA'], row['Dec'], row['z'], cosmo_utils)
    for _, row in cat_Al_comp_St.data[mask_slice_lin].iterrows()
])

extreme_coord_lin = redshift_to_comoving(extreme_row_lin['RA'], extreme_row_lin['Dec'], extreme_row_lin['z'], cosmo_utils)


# Get extreme OD from LOG-binned catalog

z_bins_log = np.logspace(np.log10(cat_Al_comp_St.data['z'].min()), np.log10(cat_Al_comp_St.data['z'].max()), 11)
bin_densities_log, bin_volumes_log = calculate_bin_densities(cat_Al_comp_St,cosmo_utils,z_bins_log)

extreme_row_log = cat_sd3D_AA_log.data.loc[cat_sd3D_AA_log.data['sd'].idxmax()]
extreme_id_log = extreme_row_log['ID']
extreme_z_log = extreme_row_log['z']

# Get the redshift bin boundaries for this extreme OD
z_bin_idx_log = extreme_row_log['z_bin']  # This should be the bin index from your catalog
z_min_bin_log = z_bins_log[z_bin_idx_log]
z_max_bin_log = z_bins_log[z_bin_idx_log + 1]

print(f"Extreme OD in log bin {z_bin_idx_log}: z={extreme_z_log:.3f} (bin: {z_min_bin_log:.3f}-{z_max_bin_log:.3f})")

delta_z_log = mpc_to_delta_z(r_Mpc, extreme_z_log, cosmo)
print(f"±{r_Mpc} Mpc corresponds to ±{delta_z_log:.4f} in redshift around z={extreme_z_log:.3f}")

# Filter field galaxies to ONLY this redshift bin
mask_slice_log = (cat_Al_comp_St.data['z'] >= z_min_bin_log - delta_z_log) & \
                 (cat_Al_comp_St.data['z'] < z_max_bin_log + delta_z_log)
coords_slice_log = np.array([
    redshift_to_comoving(row['RA'], row['Dec'], row['z'], cosmo_utils)
    for _, row in cat_Al_comp_St.data[mask_slice_log].iterrows()
])

# Get coordinates for extreme OD galaxy
extreme_coord_log = redshift_to_comoving(extreme_row_log['RA'], extreme_row_log['Dec'], extreme_row_log['z'], cosmo_utils)

# Initialize cosmology
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'  # Opens plot in default browser

# Create the figure
fig = go.Figure()

# 1. Plot ONLY galaxies in this redshift slice
coords_slice = coords_slice_lin
z_min_bin = z_min_bin_lin
z_max_bin = z_max_bin_lin
extreme_coord = extreme_coord_lin
extreme_row = extreme_row_lin
extreme_z = extreme_z_lin
z_bin_idx = z_bin_idx_lin

fig.add_trace(go.Scatter3d(
    x=coords_slice[:, 0],
    y=coords_slice[:, 1], 
    z=coords_slice[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color='blue',
        opacity=0.1,
        line=dict(width=0)
    ),
    name=f'Slice galaxies (z={z_min_bin:.2f}-{z_max_bin:.2f}, N={len(coords_slice)})'
))

# 2. Extreme overdensity galaxy
fig.add_trace(go.Scatter3d(
    x=[extreme_coord[0]],
    y=[extreme_coord[1]],
    z=[extreme_coord[2]],
    mode='markers',
    marker=dict(size=6, color='red', symbol='x'),
    name=f'Extreme OD: δ={extreme_row["sd"]:.1f}, z={extreme_z:.3f}'
))

# 3. Search sphere (3 Mpc radius)
r = 3.0
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = extreme_coord[0] + r*np.cos(u)*np.sin(v)
y_sphere = extreme_coord[1] + r*np.sin(u)*np.sin(v)
z_sphere = extreme_coord[2] + r*np.cos(v)

fig.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    opacity=0.3,
    colorscale='Reds',
    showscale=False,
    name='3 Mpc sphere'
))

# 4. Add pyramid edges for THIS SLICE only
ra_min, ra_max = 269, 271
dec_min, dec_max = -1.15, 1.15
vertices = []
for ra in [ra_min, ra_max]:
    for dec in [dec_min, dec_max]:
        for z in [z_min_bin, z_max_bin]:  # Use slice boundaries, not full range
            dc = cosmo.comoving_distance(z).value
            x = dc * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
            y = dc * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
            z_coord = dc * np.sin(np.radians(dec))
            vertices.append([x, y, z_coord])

# Plot pyramid edges (12 lines)
edges = [[0,1], [2,3], [4,5], [6,7], [0,2], [1,3], [4,6], [5,7], [0,4], [1,5], [2,6], [3,7]]
for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[vertices[edge[0]][0], vertices[edge[1]][0]],
        y=[vertices[edge[0]][1], vertices[edge[1]][1]],
        z=[vertices[edge[0]][2], vertices[edge[1]][2]],
        mode='lines',
        line=dict(color='gray', width=2, dash='dot'),
        showlegend=False
    ))

# 5. Update layout
fig.update_layout(
    title=f'Redshift Slice with Extreme Overdensity (Bin {z_bin_idx})<br>z={z_min_bin:.2f}-{z_max_bin:.2f}, δ={extreme_row["sd"]:.1f}',
    scene=dict(
        aspectmode='data',
        xaxis_title='X (Mpc)',
        yaxis_title='Y (Mpc)', 
        zaxis_title='Z (Mpc)'
    ),
    height=800
)

fig.show()


print('t = '+str(round(time.time()-st),))


# %%%% Prueba de volumenes

def calculate_volumes(z_bins, ra_span, dec_span, dec_mid):
    """Calculate volumes using both methods with correct Astropy syntax"""
    results = []
    
    for z1, z2 in zip(z_bins[:-1], z_bins[1:]):
        # Method 1: Pyramid approximation (your approach)
        dc1 = cosmo.comoving_distance(z1).value
        dc2 = cosmo.comoving_distance(z2).value
        A1 = (dc1**2) * np.radians(ra_span) * np.radians(dec_span) * np.cos(dec_mid)
        A2 = (dc2**2) * np.radians(ra_span) * np.radians(dec_span) * np.cos(dec_mid)
        pyramid_vol = 0.5 * (A1 + A2) * (dc2 - dc1)
        
        # Method 2: Proper cosmological volume
        # Correct way to get differential comoving volume in Astropy
        def integrand(z):
            return cosmo.differential_comoving_volume(z).value  # in Mpc³/sr
        
        vol_per_sr, _ = quad(integrand, z1, z2)
        cosmo_vol = vol_per_sr * (np.radians(ra_span) * np.radians(dec_span) * np.cos(dec_mid))
        
        results.append({
            'z_center': 0.5*(z1 + z2),
            'z_min': z1,
            'z_max': z2,
            'pyramid_volume': pyramid_vol,
            'cosmo_volume': cosmo_vol,
            'ratio': pyramid_vol / cosmo_vol
        })
    
    return pd.DataFrame(results)

ra_span = cat_Al.data['RA'].max() - cat_Al.data['RA'].min()
dec_span = cat_Al.data['Dec'].max() - cat_Al.data['Dec'].min()
dec_mid = 0.5 * (cat_Al.data['Dec'].min() + cat_Al.data['Dec'].max())
z_min = min(cat_Ar.data['z'].min(), cat_Al.data['z'].min())
z_max = max(cat_Ar.data['z'].max(), cat_Al.data['z'].max())
z_bins = np.logspace(np.log10(z_min), np.log10(z_max), 10)


Vols = calculate_volumes(z_bins,ra_span,dec_span,dec_mid)







# %% Correlaciones

def nan_corrcoef(x, y):
    """
    Calculate correlation coefficient ignoring NaN values
    """
    # Remove pairs where either x or y is NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:  # Need at least 2 points for correlation
        return np.nan
    
    return np.corrcoef(x_clean, y_clean)[0, 1]

# %%% Gráfica multivarible para 2D o para 3D

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

# Estándares globales para comparación cruzada de gráficas
REDSHIFT_RANGE = [0, 6]
MASS_RANGE = [9, 12]
SIZE_RANGE = [20, 300]

def create_standard_colormap(z_min=0, z_max=6):
    """Creación de mapa de colores (azul-rojo) estandarizado para los corrimientos """
    colors_blue_red = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                       '#f7f7f7',  # White center
                       '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    return mcolors.LinearSegmentedColormap.from_list('standard_blue_red', colors_blue_red)

def mass_to_size(mass_values, mass_range=MASS_RANGE, size_range=SIZE_RANGE):
    """Convert mass values to standardized sizes"""
    mass_min, mass_max = mass_range
    size_min, size_max = size_range
    # Linear scaling: mass -> size
    sizes = size_min + (size_max - size_min) * (mass_values - mass_min) / (mass_max - mass_min)
    return np.clip(sizes, size_min, size_max)

def NormalizeData(data):
    """Normalize data to [0, 1] range"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot_multivariate_correlation(catalog, flux_col='mu_Flux_850', sd_col='sd', 
                                 mass_col='M_star', z_col='z', size_scale=100,
                                 figsize=(12, 8), alpha=0.7, n_points=None,
                                 z_min=0.0, z_max=4.0):
    """
    Gráfica de burbujas multivariable con visualización de masas (tamaño) y corrimientos (colores)
    Usa colormap 'rainbow' con límites fijos de redshift y normalización personalizada de masas
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    data = catalog.data.copy()
    
    # Randomize order to prevent overplotting bias
    if n_points is not None and len(data) > n_points:
        # Subsample if too many points
        data = data.sample(n=n_points, random_state=42)
    else:
        # Still randomize order for better visibility
        data = data.sample(frac=1, random_state=42)  # Shuffle all points
    
    fluxes = data[flux_col]
    overdensities = data[sd_col]
    masses = data[mass_col]
    redshifts = data[z_col]
    
    # Apply your mass normalization and scaling
    masses_normalized = NormalizeData(masses)
    sizes = ((masses_normalized + 1) * 1.5) ** 6 * size_scale / 100  # Your scaling formula
    
    # Create scatter plot with randomized order using rainbow colormap
    scatter = ax.scatter(
        fluxes,
        overdensities,
        s=sizes,  # Use your custom mass scaling
        c=redshifts,  # Color by redshift
        cmap='rainbow',  # Use rainbow colormap as requested
        alpha=alpha,
        edgecolors='black',
        linewidth=0.3,
        zorder=3  # Ensure points are above grid
    )
    
    # Set fixed color limits for redshift
    scatter.set_clim(z_min, z_max)
    
    # Labels and title
    # ax.set_xlabel(f'{flux_col} [mJy]', fontsize=14, fontweight='bold')
    ax.set_xlabel(r'$S_{850}$'+u' [mJy]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\delta(\rho) = (\rho_x - \bar{\rho})/\bar{\rho}$', fontsize=14, fontweight='bold')
    ax.set_title(u'Sobredensidad vs densidad de flujo sin cortes\n'+
                 '(tamaño: masa, color: corrimiento) \n'+
                 f'M_c > {M_ce}, M_v > {M_ve}, r_c[Mpc] = {r_Mpc}', fontsize=16, fontweight='bold')
    
    # Log scales if appropriate
    if fluxes.max() / fluxes.min() > 100:
        ax.set_xscale('log')
        ax.set_xlabel(f'{flux_col} [mJy] (log scale)', fontsize=14, fontweight='bold')
    
    if overdensities.max() / overdensities.min() > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Overdensity (δ) (log scale)', fontsize=14, fontweight='bold')
    
    # Add colorbar for redshift with fixed ticks
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('Corrimiento (z)', fontsize=12, fontweight='bold')
    cbar.set_ticks(np.arange(z_min, z_max + 1))  # Integer ticks from z_min to z_max
    
    # Add mass legend using your mass values
    mass_legend_values = [9, 9.5, 10, 10.5, 11]  # Example mass values, adjust as needed
    mass_legend_handles = []
    
    for mass_val in mass_legend_values:
        # Calculate size using your formula
        mass_norm = (mass_val - masses.min()) / (masses.max() - masses.min())
        size_val = ((mass_norm + 1) * 1.5) ** 6 * size_scale / 100
        
        # Create legend handle
        handle = ax.scatter([], [], s=size_val, c='gray', alpha=0.8, 
                          edgecolors='black', label=f'M={mass_val:.1f}')
        mass_legend_handles.append(handle)
    
    # Add mass legend to plot
    ax.legend(handles=mass_legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1),
              title='Masa Estelar', frameon=True)
    
    # Add grid with lower zorder
    ax.grid(True, alpha=0.2, linestyle='--', zorder=1)
    
    # Add correlation coefficient
    corr_coef = nan_corrcoef(fluxes, overdensities)
    ax.text(0.05, 0.95, f'ρ = {corr_coef:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=12, fontweight='bold')
    
    # Add sample size info
    ax.text(0.05, 0.05, f'N = {len(data)}', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    return fig, ax, scatter  # Return scatter object for additional customization if needed


# Usage for both 2D and 3D overdensities
fig_2d, ax_2d, scatter_2d = plot_multivariate_correlation(
    cat_sd2D_AA, 
    flux_col='mu_Flux_850', 
    sd_col='sd',
    mass_col='M_star',
    z_col='z',
    size_scale=300,  # Adjust point size scaling
    alpha=0.6,
    n_points=2000,
    z_min=0.0,      # Fixed redshift limits
    z_max=4.0       # Fixed redshift limits
)

fig_3d, ax_3d, scatter_3d = plot_multivariate_correlation(
    cat_sd3D_AA_lin,  # or cat_sd3D_AA_log
    flux_col='mu_Flux_850',
    sd_col='sd', 
    mass_col='M_star',
    z_col='z',
    size_scale=300,
    alpha=0.6,
    n_points=2000,
    z_min=0.0,      # Same fixed limits for comparison
    z_max=4.0       # Same fixed limits for comparison
)







# %%% Gráfica interactiva

import plotly.express as px
import plotly.io as pio

# Set the default renderer to browser
pio.renderers.default = "browser"

def interactive_multivariate_plot(catalog, flux_col='mu_Flux_850', sd_col='sd',
                                mass_col='M_star', z_col='z', max_points=2000):
    """
    Interactive version with Plotly for better exploration
    """
    # First, clean the data - remove NaN values
    data = catalog.data.dropna(subset=[sd_col, flux_col, mass_col, z_col]).copy()
    
    # Subsample if needed
    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42)
    
    print(f"Plotting {len(data)} points...")
    
    # Create size array with proper scaling (Plotly needs explicit sizes)
    mass_min, mass_max = data[mass_col].min(), data[mass_col].max()
    data['point_size'] = 5 + 15 * (data[mass_col] - mass_min) / (mass_max - mass_min)
    
    fig = px.scatter(
        data,
        x=flux_col,
        y=sd_col,
        size='point_size',  # Use the computed size column
        color=z_col,
        size_max=20,
        color_continuous_scale='RdBu_r',
        title=f'Overdensity vs {flux_col} (size: mass, color: redshift)',
        hover_data=['ID', z_col, mass_col, sd_col, flux_col],
        opacity=0.7
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=f'{flux_col} [mJy]',
        yaxis_title='Overdensity (δ)',
        coloraxis_colorbar=dict(title='Redshift'),
        template='plotly_white',
        width=1000,
        height=700
    )
    
    # Add log scales if needed
    if data[flux_col].max() / data[flux_col].min() > 100:
        fig.update_xaxes(type='log', title=f'{flux_col} [mJy] (log scale)')
    if data[sd_col].max() / data[sd_col].min() > 100:
        fig.update_yaxes(type='log', title='Overdensity (δ) (log scale)')
    
    return fig

# Usage with explicit show()
fig_interactive_2D = interactive_multivariate_plot(cat_sd2D_AA, max_points=3000)
fig_interactive_2D.show(renderer="browser")  # Force browser renderer

# For the second plot, you might need a slight delay
import time
time.sleep(2)  # Wait 2 seconds between plots

fig_interactive_3D = interactive_multivariate_plot(cat_sd3D_AA_lin, max_points=3000)
fig_interactive_3D.show(renderer="browser")


# %%% Evolución de sd con el corrimiento

def plot_redshift_evolution(catalog, sd_col='sd', z_col='z', mass_col='M_star'):
    """
    Plot how overdensity evolves with redshift, colored by mass
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sc = ax.scatter(
        catalog.data[z_col],
        catalog.data[sd_col],
        c=catalog.data[mass_col],
        cmap='plasma',
        alpha=0.7,
        s=50
    )
    
    ax.set_xlabel('Redshift (z)', fontsize=14)
    ax.set_ylabel('Overdensity (δ)', fontsize=14)
    ax.set_title('Overdensity Evolution with Redshift\n(color: stellar mass)', fontsize=16)
    
    if catalog.data[sd_col].max() / catalog.data[sd_col].min() > 100:
        ax.set_yscale('log')
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Stellar Mass (log M☉)', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    return fig, ax

fig_2D, ax_2D = plot_redshift_evolution(cat_sd2D_AA)
fig_3D, ax_3D = plot_redshift_evolution(cat_sd3D_AA_lin)
plt.show()



# %%% Comparación sd 2D vs 3D

def plot_2d_vs_3d_comparison(cat_2d, cat_3d, mass_col='M_star', z_col='z', 
                           figsize=(10, 8), alpha=0.7, n_points=None):
    """
    Compare 2D vs 3D overdensities for the same sources
    """
    # Merge the catalogs on ID to ensure same sources
    merged_data = pd.merge(
        cat_2d.data[['ID', 'sd', mass_col, z_col]].rename(columns={'sd': 'sd_2d'}),
        cat_3d.data[['ID', 'sd', mass_col, z_col]].rename(columns={'sd': 'sd_3d'}),
        on='ID',
        suffixes=('', '_3d')
    )
    
    # Randomize order to prevent overplotting bias
    if n_points is not None and len(merged_data) > n_points:
        # Subsample if too many points
        merged_data = merged_data.sample(n=n_points, random_state=42)
    else:
        # Still randomize order for better visibility
        merged_data = merged_data.sample(frac=1, random_state=42)  # Shuffle all points
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(
        merged_data['sd_2d'],
        merged_data['sd_3d'],
        s=50 * (merged_data[mass_col] - merged_data[mass_col].min()) / 
           (merged_data[mass_col].max() - merged_data[mass_col].min()) + 20,
        c=merged_data[z_col],
        cmap='rainbow',
        alpha=alpha,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Set color limits
    scatter.set_clim(0, 4)
    
    # Add 1:1 line for reference
    max_val = max(merged_data['sd_2d'].max(), merged_data['sd_3d'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
    
    # Labels and title
    ax.set_xlabel(r'$\delta_{2D}$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\delta_{3D}$', fontsize=14, fontweight='bold')
    ax.set_title('Comparación de sobredensidades (2D vs 3D) sin cortes \n'+
                 ' (tamaño: masa, color: corrimiento) \n'+
                 f'M_c > {M_ce}, M_v > {M_ve}, r_c[Mpc] = {r_Mpc}', 
                 fontsize=16, fontweight='bold')
    
    # Log scale if needed
    if max_val > 100:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\delta_{2D}$'+u' - log scale')
        ax.set_ylabel(r'$\delta_{3D}$'+u' - log scale')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Corrimiento (z)', fontweight='bold')
    
    # Add correlation coefficient
    corr_coef = nan_corrcoef(merged_data['sd_2d'], merged_data['sd_3d'])
    ax.text(0.05, 0.95, f'corr_coef = {corr_coef:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=12, fontweight='bold')
    
    # Add sample info
    ax.text(0.05, 0.05, f'N = {len(merged_data)} fuentes conjuntas', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=10)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim((merged_data['sd_2d'].min(),merged_data['sd_2d'].max()))
    plt.tight_layout()
    
    return fig, ax, merged_data

# Usage
fig_compare, ax_compare, matched_data = plot_2d_vs_3d_comparison(
    cat_sd2D_AA, 
    cat_sd3D_AA_lin,  # or cat_sd3D_AA_log
    n_points = None
)

plt.show()





# %%% Relación Masa-sd para distintos bines de corrimiento

def plot_mass_sd_by_z(catalog, sd_col='sd', mass_col='M_star', z_col='z', n_z_bins=4):
    """
    Plot mass vs overdensity in different redshift bins
    """
    z_bins = np.quantile(catalog.data[z_col], np.linspace(0, 1, n_z_bins + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(n_z_bins):
        mask = (catalog.data[z_col] >= z_bins[i]) & (catalog.data[z_col] < z_bins[i+1])
        subset = catalog.data[mask]
        
        axes[i].scatter(
            subset[mass_col],
            subset[sd_col],
            c=subset[z_col],
            cmap='rainbow',
            alpha=0.7,
            s=50
        )
        
        axes[i].set_xlabel('Stellar Mass (log M☉)', fontsize=12)
        axes[i].set_ylabel('Overdensity (δ)', fontsize=12)
        axes[i].set_title(f'z = {z_bins[i]:.2f} - {z_bins[i+1]:.2f}', fontsize=14)
        
        if subset[sd_col].max() / subset[sd_col].min() > 100:
            axes[i].set_yscale('log')
    
    plt.tight_layout()
    return fig, axes

fig_2D, ax_2D = plot_mass_sd_by_z(cat_sd2D_AA)
fig_3D, ax_3D = plot_mass_sd_by_z(cat_sd3D_AA_lin)
plt.show()










# %%% Correlación entre distintas variables

# %%%% Relación real entre las variables y delta

feature_vars = ['mu_Flux_850', 'M_star', 'z', 'SFR', 'M_vir_halo']  # Define ONCE
feature_vars_u = ['mJy', 'log(M/M_sun)', 'z','M_sun/yr','log(M/M_sun)']
i = 2
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(cat_sd2D_AA.data[feature_vars[i]], cat_sd2D_AA.data['sd'], 
                    alpha=0.5, s=20)
ax.set_xlabel(f'{feature_vars[i]} [{feature_vars_u[i]}]')
ax.set_ylabel('Actual Overdensity (δ)')
ax.set_title(f'Actual Relationship: {feature_vars[i]} vs Overdensity')
plt.show()

# %%%% Matríz de correlaciones

def plot_correlation_matrix(catalog, variables=None, sd=None):
    """
    Plot correlation matrix for all numerical variables
    """
    if variables is None:
        # Select only numerical columns
        numerical_cols = catalog.data.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID columns and other non-physical variables
        numerical_cols = [col for col in numerical_cols if 'ID' not in col and 'bin' not in col]
    else:
        numerical_cols = variables
    
    data = catalog.data[numerical_cols].dropna()
    
    # Calculate correlation matrix
    corr_matrix = data.corr(method='spearman')  # Use Spearman for non-linear relationships
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax)
    if sd == None:
        print('Especificar caso: 2D o 3D')
    else:
        ax.set_title('Matríz de correlación Spearman '+sd, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax, corr_matrix

# Usage
variables_to_analyze = ['mu_Flux_850', 'sd', 'M_star', 'z', 'SFR', 'M_vir_halo']
fig_corr_2D, ax_corr_2D, corr_matrix_2D = plot_correlation_matrix(cat_sd2D_AA, variables_to_analyze, '2D')
fig_corr_3D, ax_corr_3D, corr_matrix_3D = plot_correlation_matrix(cat_sd3D_AA_lin, variables_to_analyze, '3D')






# %%%% Correlación con ML Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def analyze_feature_importance(catalog, target_var='sd', n_estimators=100):
    """
    Use Random Forest to determine which features most influence overdensity
    """
    # Select features and target
    feature_vars = ['mu_Flux_850', 'M_star', 'z', 'SFR', 'M_vir_halo']  # Adjust as needed
    data = catalog.data[feature_vars + [target_var]].dropna()
    
    X = data[feature_vars]
    y = data[target_var]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_vars,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax)
    ax.set_title(f'Feature Importance for Predicting {target_var}', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    
    return fig, ax, feature_importance_df, rf

# Usage
fig_importance_2D, ax_importance_2D, importance_df_2D, rf_model_2D = analyze_feature_importance(cat_sd2D_AA)
fig_importance_3D, ax_importance_3D, importance_df_3D, rf_model_3D = analyze_feature_importance(cat_sd3D_AA_lin)


# %%%% Dependencia parcial (cuando las otras variables se dejan constantes)
from sklearn.inspection import PartialDependenceDisplay

def plot_partial_dependence(rf_model, catalog, features=None, sd=None):
    """
    Show how each feature affects overdensity when others are held constant
    """
    if features is None:
        features = ['mu_Flux_850', 'M_star', 'z']
    
    data = catalog.data[features].dropna()
    feature_vars = data.columns.tolist()
    
    # Scale features (same as during training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Create partial dependence plots
    fig, ax = plt.subplots(figsize=(15, 5))
    PartialDependenceDisplay.from_estimator(
        rf_model, X_scaled, features=range(len(features)),
        feature_names=feature_vars, ax=ax
    )
    if sd == None:
        print('Especificar caso: 2D o 3D')
    else:
        ax.set_title('Dependencia parcial con las otras variables fijas '+sd, 
                 fontsize=16, fontweight='bold')
    
    return fig, ax

# Usage (requires the rf_model from previous function)
fig_pdp_2D, ax_pdp_2D = plot_partial_dependence(rf_model_2D, cat_sd2D_AA, sd='2D')
fig_pdp_3D, ax_pdp_3D = plot_partial_dependence(rf_model_3D, cat_sd3D_AA_lin, sd='3D')

# %%%% Dependencia parcial con una mejor interpretación

def analyze_feature_importance(catalog, target_var='sd', feature_vars=None, n_estimators=100):
    """
    Use Random Forest to determine which features most influence overdensity
    """
    if feature_vars is None:
        feature_vars = ['mu_Flux_850', 'M_star', 'z', 'SFR', 'M_vir_halo']  # Default features
    
    # Select features and target
    data = catalog.data[feature_vars + [target_var]].dropna()
    
    X = data[feature_vars]
    y = data[target_var]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_vars,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax)
    ax.set_title(f'Feature Importance for Predicting {target_var}', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    
    # Return the scaler and feature list for consistent future use
    return fig, ax, feature_importance_df, rf, scaler, feature_vars

def plot_partial_dependence_physical(rf_model, scaler, feature_vars, catalog, features_to_plot=None):
    """
    Partial dependence plots with physical units - FIXED feature consistency
    """
    if features_to_plot is None:
        features_to_plot = feature_vars  # Use all features by default
    
    data = catalog.data[feature_vars + ['sd']].dropna()
    
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(5*len(features_to_plot), 5))
    
    for i, feature in enumerate(features_to_plot):
        # Create grid of physical values for THIS feature
        feature_range = np.linspace(data[feature].min(), data[feature].max(), 50)
        
        # Calculate partial dependence
        pdp_values = []
        
        for val in feature_range:
            # Create test data with ALL features at their mean values
            test_data = data[feature_vars].mean().to_frame().T
            
            # Modify only the current feature
            test_data[feature] = val
            
            # Scale using the SAME scaler from training
            test_data_scaled = scaler.transform(test_data)
            
            # Predict
            pdp_values.append(rf_model.predict(test_data_scaled)[0])
        
        # Plot with physical units
        if len(features_to_plot) > 1:
            ax = axes[i]
        else:
            ax = axes
            
        ax.plot(feature_range, pdp_values, 'b-', linewidth=3)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Predicted Overdensity (δ)', fontsize=12)
        ax.set_title(f'Partial Dependence: {feature}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

# Usage with consistent features
feature_vars = ['mu_Flux_850', 'M_star', 'z', 'SFR', 'M_vir_halo']  # Define ONCE

# 1. Train the model
fig_importance_2D, ax_importance_2D, importance_df_2D, rf_model_2D, scaler_2D, feature_vars_2D = analyze_feature_importance(
    cat_sd2D_AA, 
    feature_vars=feature_vars  # Pass the same feature list
)

fig_importance_3D, ax_importance_3D, importance_df_3D, rf_model_3D, scaler_3D, feature_vars_3D = analyze_feature_importance(
    cat_sd3D_AA_lin, 
    feature_vars=feature_vars  # Pass the same feature list
)

# 2. Plot partial dependence for top features
top_features_2D = importance_df_2D['feature'].head(3).tolist()  # Get top 3 features
fig_pdp_2D, axes_pdp_2D = plot_partial_dependence_physical(
    rf_model_2D, 
    scaler_2D, 
    feature_vars,  # Pass ALL features used in training
    cat_sd2D_AA,
    features_to_plot=top_features_2D  # Plot only these 3
)

top_features_3D = importance_df_3D['feature'].head(3).tolist()  # Get top 3 features
fig_pdp_3D, axes_pdp_3D = plot_partial_dependence_physical(
    rf_model_3D, 
    scaler_3D, 
    feature_vars,  # Pass ALL features used in training
    cat_sd3D_AA_lin,
    features_to_plot=top_features_3D  # Plot only these 3
)

# %%%% Análisis de agrupamiento (ML)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster_analysis(catalog, n_clusters=3, sd=None):
    """
    Identify natural clusters in your data
    """
    features = ['mu_Flux_850', 'sd', 'M_star', 'z']
    data = catalog.data[features].dropna()
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    if sd == None:
        print('Especificar caso: 2D o 3D')
    else:
        ax.set_title(f'K-means Clustering (k={n_clusters}) '+sd, fontsize=16, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    
    # Analyze cluster characteristics
    data['cluster'] = clusters
    cluster_stats = data.groupby('cluster').mean()
    
    return fig, ax, cluster_stats, kmeans

# Usage
fig_cluster_2D, ax_cluster_2D, cluster_stats_2D, kmeans_model_2D = cluster_analysis(cat_sd2D_AA, sd='2D')
fig_cluster_3D, ax_cluster_3D, cluster_stats_3D, kmeans_model_3D = cluster_analysis(cat_sd3D_AA_lin, sd='3D')




# %%%% Agrupamiento con mejor visualización

def better_cluster_analysis(catalog, n_clusters=3):
    """
    Improved clustering visualization with proper categorical coloring
    and integration of cluster labels into the catalog
    """
    features = ['mu_Flux_850', 'sd', 'M_star', 'z']
    data = catalog.data[features].dropna().copy()  # Use copy to avoid modifying original
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Add cluster labels to the temporary data
    data['cluster'] = clusters
    
    # Create proper categorical coloring
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use distinct colors for each cluster
    colors = ['blue', 'green', 'red', 'purple', 'orange'][:n_clusters]
    
    for cluster_id in range(n_clusters):
        mask = data['cluster'] == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=colors[cluster_id], 
                  label=f'Cluster {cluster_id}',
                  alpha=0.7, s=30)
    
    ax.set_xlabel('PCA Component 1 (Most important variation)', fontsize=12)
    ax.set_ylabel('PCA Component 2 (Second most important variation)', fontsize=12)
    ax.set_title(f'K-means Clustering (k={n_clusters})', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Analyze what each cluster represents
    cluster_stats = data.groupby('cluster').mean()
    cluster_sizes = data.groupby('cluster').size()
    
    print("=== Cluster Characteristics ===")
    for cluster_id in range(n_clusters):
        print(f"\nCluster {cluster_id} (N={cluster_sizes[cluster_id]}):")
        print(cluster_stats.loc[cluster_id])
    
    # Now add cluster labels to the ORIGINAL catalog for future use
    # We need to handle the fact that some rows had NaN and were dropped
    catalog.data = catalog.data.copy()  # Ensure we're not modifying a view
    
    # Initialize cluster column with NaN
    catalog.data['cluster'] = np.nan
    
    # Find the indices of the data we actually clustered (after dropping NaN)
    clustered_indices = data.index  # These are the rows that were used
    
    # Assign cluster labels to the appropriate rows
    catalog.data.loc[clustered_indices, 'cluster'] = clusters
    
    # Update catalog metadata
    catalog.columns['cluster'] = {
        'units': None,
        'description': f'K-means cluster assignment (k={n_clusters})'
    }
    
    return fig, ax, cluster_stats, kmeans, pca, data

# Usage
fig_cluster_2D, ax_cluster_2D, cluster_stats_2D, kmeans_model_2D, pca_2D, clustered_data_2D = better_cluster_analysis(cat_sd2D_AA)
fig_cluster_3D, ax_cluster_3D, cluster_stats_3D, kmeans_model_3D, pca_3D, clustered_data_3D = better_cluster_analysis(cat_sd3D_AA_lin)

# 1. Check the cluster distribution
print("Cluster distribution in full catalog:")
print(cat_sd2D_AA.data['cluster'].value_counts(dropna=False))

# 2. Examine the extreme cluster in detail
extreme_cluster_2D = cat_sd2D_AA.data[cat_sd2D_AA.data['cluster'] == 2]  # Right cluster
print(f"Extreme cluster has {len(extreme_cluster_2D)} galaxies")
print("Properties:", extreme_cluster_2D[['mu_Flux_850', 'sd', 'M_star', 'z']].mean())

# 3. See where your extreme overdensity sources fall
extreme_sources_2D = cat_sd2D_AA.data[cat_sd2D_AA.data['sd'] > 10]  # Or your threshold
print(f"Extreme overdensity sources are in clusters: {extreme_sources_2D['cluster'].value_counts()}")

# 4. Check cluster properties
for cluster_id in [0, 1, 2]:
    cluster_data_2D = cat_sd2D_AA.data[cat_sd2D_AA.data['cluster'] == cluster_id]
    print(f"\n=== Cluster {cluster_id} ===")
    print(f"Size: {len(cluster_data_2D)} galaxies")
    print(f"Mean redshift: {cluster_data_2D['z'].mean():.2f}")
    print(f"Mean mass: {cluster_data_2D['M_star'].mean():.2f}")
    print(f"Mean flux: {cluster_data_2D['mu_Flux_850'].mean():.2f}")
    print(f"Mean overdensity: {cluster_data_2D['sd'].mean():.2f}")


# 1. Check the cluster distribution
print("Cluster distribution in full catalog:")
print(cat_sd3D_AA_lin.data['cluster'].value_counts(dropna=False))

# 2. Examine the extreme cluster in detail
extreme_cluster_3D = cat_sd3D_AA_lin.data[cat_sd3D_AA_lin.data['cluster'] == 2]  # Right cluster
print(f"Extreme cluster has {len(extreme_cluster_3D)} galaxies")
print("Properties:", extreme_cluster_3D[['mu_Flux_850', 'sd', 'M_star', 'z']].mean())

# 3. See where your extreme overdensity sources fall
extreme_sources_3D = cat_sd3D_AA_lin.data[cat_sd3D_AA_lin.data['sd'] > 10]  # Or your threshold
print(f"Extreme overdensity sources are in clusters: {extreme_sources_3D['cluster'].value_counts()}")

# 4. Check cluster properties
for cluster_id in [0, 1, 2]:
    cluster_data_3D = cat_sd3D_AA_lin.data[cat_sd3D_AA_lin.data['cluster'] == cluster_id]
    print(f"\n=== Cluster {cluster_id} ===")
    print(f"Size: {len(cluster_data_3D)} galaxies")
    print(f"Mean redshift: {cluster_data_3D['z'].mean():.2f}")
    print(f"Mean mass: {cluster_data_3D['M_star'].mean():.2f}")
    print(f"Mean flux: {cluster_data_3D['mu_Flux_850'].mean():.2f}")
    print(f"Mean overdensity: {cluster_data_3D['sd'].mean():.2f}")

# 3. Check if clusters match known categories
# Do the clusters correspond to different redshift ranges? Mass ranges? etc.
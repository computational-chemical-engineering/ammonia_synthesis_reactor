"""JSON-backed database of gas-mixture property data.

Provides :class:`MixturePropertyDatabase`, which loads property tables
into pandas DataFrames, supports adding or updating entries, and
retrieves properties for a given list of species.
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Module-level logger
logger = logging.getLogger(__name__)


class MixturePropertyDatabase:
    """
    A class to manage and retrieve properties of gas mixtures from a database.

    This class provides methods to load data from a JSON file, save data to a JSON file,
    add or update entries, and retrieve properties for a given list of species.

    Attributes:
    -----------
    dataframes : dict
        A dictionary to store Pandas DataFrames for different property categories.

    Methods:
    --------
    __init__(self, filename=None):
        Initialize the database. If a JSON file is provided, load the data.

    _standardize_label(self, species_list):
        Standardize multi-species labels to a unique ordering.

    load_json(self, filename):
        Load data from a JSON file and create Pandas DataFrames dynamically.

    save_json(self, filename):
        Save the current database to a JSON file.

    add_entry(self, df_name, species, properties):
        Add or update an entry in a specific DataFrame.

    get_species_properties(self, df_name, species_list):
        Retrieve all available properties for a given list of species.
    """

    def __init__(self, filename=None):
        """
        Initialize the database. If a JSON file is provided, load the data.

        Parameters:
        -----------
        filename : str, optional
            Path to the JSON file containing the database.
        """
        self.dataframes = {}
        if filename:
            self.load_json(self._resolve_filename(filename))

    @staticmethod
    def _resolve_filename(filename):
        """Resolve database path with package-relative fallbacks for robustness."""
        path = Path(filename)
        if path.exists():
            return path
        if path.is_absolute():
            return path

        package_dir = Path(__file__).resolve().parent
        candidate_paths = [
            package_dir / path,
            package_dir / "data" / path.name,
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate
        return path

    def _standardize_label(self, species_list):
        """
        Standardize multi-species labels to a unique ordering.

        Parameters:
        -----------
        species_list : list
            List of species names.

        Returns:
        --------
        str
            Standardized label for the species.
        """
        return "/".join(sorted(species_list))

    def load_json(self, filename):
        """
        Load data from a JSON file and create Pandas DataFrames dynamically.

        Parameters:
        -----------
        filename : str
            Path to the JSON file containing the database.
        """
        with open(filename, "r") as f:
            data = json.load(f)

        for df_name, df_info in data.items():
            columns = df_info["columns"]
            records = df_info["data"]

            # Convert dictionary into DataFrame
            df = pd.DataFrame.from_dict(records, orient="index", columns=columns)

            # Store DataFrame and metadata
            self.dataframes[df_name] = df

    def save_json(self, filename):
        """
        Save the current database to a JSON file.

        Parameters:
        -----------
        filename : str
            Path to the JSON file to save the database.
        """
        data = {}

        for df_name, df in self.dataframes.items():
            columns = df.columns.tolist()
            # Convert DataFrame back to dictionary
            records = df.to_dict(orient="index")

            # Store metadata and data
            data[df_name] = {"columns": columns, "data": records}

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def add_entry(self, df_name, species, properties):
        """
        Add or update an entry in a specific DataFrame.

        Parameters:
        -----------
        df_name : str
            Name of the DataFrame to add or update the entry.
        species : str or list
            Species name or list of species names.
        properties : dict
            Dictionary of properties to add or update.
        """
        if df_name not in self.dataframes:
            raise ValueError(f"DataFrame '{df_name}' does not exist in the database.")

        df = self.dataframes[df_name]
        expected_columns = df.columns.tolist()

        if isinstance(species, (list, tuple)):
            species = "/".join(species)  # Keep order

        # Convert properties to DataFrame row
        new_entry = pd.DataFrame([properties], index=[species])

        # Find missing columns
        missing_cols = [col for col in expected_columns if col not in properties]
        if missing_cols:
            logger.warning(
                "Missing properties %s for %s in %s", missing_cols, species, df_name
            )

        # Add or update the row
        self.dataframes[df_name] = df.combine_first(new_entry)

    def get_species_properties(self, df_name, species_list):
        """
        Retrieve all available properties for a given list of species.

        Parameters:
        -----------
        df_name : str
            Name of the DataFrame to retrieve properties from.
        species_list : list
            List of species names.

        Returns:
        --------
        dict
            Dictionary of properties with property names as keys and numpy arrays as values.
        """
        if df_name not in self.dataframes:
            raise ValueError(f"DataFrame '{df_name}' does not exist in the database.")

        df = self.dataframes[df_name]
        num_species = len(species_list)
        properties_dict = {}

        # Determine which properties are symmetric
        symmetric_properties = {prop: True for prop in df.columns}
        count_species = {prop: 1 for prop in df.columns}

        for entry in df.index:
            species_entry = entry.split("/")
            if len(species_entry) > 1:
                reversed_label = "/".join(reversed(species_entry))
                found_reversed_label = reversed_label in df.index
                for prop in df.columns:
                    if not pd.isna(df.at[entry, prop]):
                        count_species[prop] = max(
                            count_species[prop], len(species_entry)
                        )
                        if found_reversed_label and not pd.isna(
                            df.at[reversed_label, prop]
                        ):
                            symmetric_properties[prop] = (
                                False  # Found asymmetric property
                            )

        # Initialize arrays for properties
        for prop in df.columns:
            shape = (num_species,) * count_species[prop]
            properties_dict[prop] = np.full(shape, np.nan)

        # Process each row in the DataFrame
        for entry, row in df.iterrows():
            species_entry = entry.split("/")
            species_indices = [
                species_list.index(s) for s in species_entry if s in species_list
            ]

            if len(species_indices) != len(species_entry):
                continue  # Skip if some species are missing

            indices = tuple(species_indices)
            reversed_label = "/".join(reversed(species_entry))

            for prop in df.columns:
                value = row[prop]
                if pd.isna(value):
                    continue

                if symmetric_properties[prop]:
                    # Symmetric case: Fill both [i, j] and [j, i]
                    properties_dict[prop][indices] = value
                    properties_dict[prop][tuple(reversed(indices))] = value
                else:
                    # Asymmetric case: Only store in original order
                    properties_dict[prop][indices] = value

        return properties_dict

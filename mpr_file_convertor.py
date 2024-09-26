from biologic_reader import BiologicReader
from pathlib import Path
import logging


def convert_file_to_cloud(mpr_file_path):
    """
    This function converts an mpr file at the provided path to the NV cloud format DataFrame
    using the BiologicReader. If an error occurs while reading or converting the file,
    the function will return None.

    Args:
        mpr_file_path (str): The file path to process.

    Returns:
        DataFrame: The Biologic data converted to cloud format, or None if an error occurred.
    """
    try:
        file_path = Path(mpr_file_path)
        logging.debug(f"mpr_file_convertor. Converting {file_path.stem} to cloud format")
        # create an instance of the BiologicReader class
        reader = BiologicReader()
        biologic_data = reader.read(file_path)
        cloud_data = reader.convert_to_cloud(biologic_data)
        #logging.debug(f"{file_path.stem} converted successfully to cloud format")
        return cloud_data
    except Exception as e:
        print(f"mpr_file_convertor. An error occurred while processing file {mpr_file_path}: {e}")
        return None
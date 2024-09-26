import re
import logging

                                                                                            
def extract_doe_cell(filename):
    """
    Extracts the DOE and Cell numbers from the given filename.
    Parameters:
        filename (str): The filename to parse.
    Returns:
        tuple: A tuple containing the DOE and Cell numbers.
    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    logging.debug(f"FILENAME PROCESSING FUNCTIONS. Extracting DOE and cell number for {filename}.")

    # Define the regex pattern to match keywords and cell number
    pattern = re.compile(
        r'(Formation-Capacity-Check|FC-DCIR-Rate|Cycle-Life)-CC-(?P<cell>\d+)_0\.\d+.*',
        re.IGNORECASE
    )

    # Attempt to match the pattern
    match = pattern.search(filename)
    if match:
        # Split the filename by '-' to get the parts
        parts = filename.split('-')

        # List of keywords to search for in the filename
        keywords = ['Formation', 'FC', 'Cycle']

        # Find the index of the first keyword in the split parts
        keyword_index = None
        for i, part in enumerate(parts):
            if part.lower() == 'rerun':
                continue  # Skip the 'rerun' part
            if part in keywords:
                keyword_index = i
                break

        # Ensure that we found a keyword, otherwise raise an error
        if keyword_index is None:
            raise ValueError(f"Filename '{filename}' does not contain a recognized keyword.")

        # DOE is the part right before the keyword
        doe = parts[keyword_index - 1]
        if doe.lower() == 'rerun':
            doe = parts[keyword_index - 2]  # Go back one more if DOE is 'rerun'

        # Extract cell number from the regex match
        cell = match.group('cell')

        logging.debug(f"FILENAME PROCESSING FUNCTIONS. DOE and cell number for {filename} extracted successfully.")
        return doe, cell
    else:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")



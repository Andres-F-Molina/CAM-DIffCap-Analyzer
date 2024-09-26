import pandas as pd
import logging
import BioLogic
from pathlib import Path

class BiologicReader:
    """
    Reads Biologic data from local file system.
    The Biologic files should be in the .mpr format.
    """

    def __init__(self):
        self.extension = ".mpr"

    def read(self, path: Path) -> pd.DataFrame:
        """
        Read Biologic data from local file system.

        Parameters
        ----------
        path : pathlib.Path
            Path to the Biologic (.mpr) data file.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe containing the Biologic data in
            Biologic format.
        """
        try:
            logging.debug(f"BIOLOGIC-READER. Reading Biologic data from {Path(path).stem}...")
            mpr_file = BioLogic.MPRfile(path.as_posix())
            df = pd.DataFrame(mpr_file.data)
            return df
        except Exception as e:
            logging.error(f"BIOLOGIC-READER. An error occurred while reading file: {e}")
            raise

    def convert_to_cloud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Biologic data to cloud format.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the Arbin data in
            Arbin format.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe containing the Arbin data in
            cloud format.
        """

        try:
            logging.debug(f"BIOLOGIC-READER. Converting Biologic data to cloud format...")
            total_time_millis = df["time/s"] * 1e3  # convert to milliseconds
            total_time_millis -= total_time_millis.min()  # set the first time point to zero

            step_number = df["Ns"]

            # find the cycle number by adding 1 to the step number every time it reduces
            cycle = (step_number.astype(int).diff() < 0).cumsum() + 1

            step_amp_hours = df["Q charge/discharge/mA.h"].abs() * 1e-3  # convert to Ah

            # create an intermediate df with the cycle number and step numberq
            int_df = pd.DataFrame(
                {
                    "cycle": cycle,
                    "step_number": step_number,
                    "total_time_millis": total_time_millis,
                }
            )

            # find the discharge and charge capacities by step
            initial_step_time = int_df.groupby(["cycle", "step_number"])[
                "total_time_millis"
            ].first()

            # map onto the whole dataframe by creating a multi-index
            mapping_index = pd.Series(list(zip(int_df["cycle"], int_df["step_number"])))

            step_time_init = mapping_index.map(initial_step_time)

            step_time_millis = int_df["total_time_millis"] - step_time_init

            dQ = df["dQ/mA.h"] * 1e-3  # convert to Ah
            dt = step_time_millis.diff() * 1e-3 / 3600  # convert to hours

            current = dQ / dt

            # fill back the first value with the second value
            current.iloc[0] = current.iloc[1]

            df = pd.DataFrame(
                {
                    "total_time_millis": total_time_millis,
                    "step_time_millis": step_time_millis,
                    "step_number": step_number,
                    "current": current,
                    "voltage": df["Ewe/V"].to_numpy(),
                    "step_amp_hours": step_amp_hours,
                    "cycle": cycle,
                    "label": "CYCLING",
                }
            )

            df["step_type"] = "REST"
            # when current > 0 it's a CHARGE step
            df.loc[df["current"] > 0, "step_type"] = "CHARGE"
            # when current < 0 it's a DISCHARGE step
            df.loc[df["current"] < 0, "step_type"] = "DISCHARGE"
            logging.debug(f"BIOLOGIC-READER. Biologic data successfully converted to cloud format...")
            return df

        except Exception as e:
            logging.error(f"BIOLOGIC-READER. An error occurred while converting data to cloud format: {e}")
            raise
import logging
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import dotenv_values

from .cache import Cache


class EntsoePyWrapper:
    """
    A wrapper class for querying the `EntsoePandasClient` with caching functionality.
    This class provides a method to query the `EntsoePandasClient` and cache the results
    to avoid redundant API calls. It uses a caching mechanism to store and retrieve
    query results.
    """

    base_entries = dict(
        time_periods=[
            dict(
                t0=pd.Timestamp(year=2024, month=1, day=1, hour=0, minute=0, tz="Europe/Vienna"),
                t1=pd.Timestamp(year=2024, month=12, day=31, hour=23, minute=0, tz="Europe/Vienna"),
            )
        ],
        countries=["AT", "DE_LU"],
        country_queries=[
            "day_ahead_prices",
            "net_position",
            "load",
            "load_forecast",
            "generation",
            "generation_forecast",
            "wind_and_solar_forecast",
        ],
        links=[("AT", "DE")],
        link_queries=[dict(query="crossborder_flows", both=True)],
    )

    def __init__(self):
        self._client = EntsoePandasClient(api_key=dotenv_values().get("ENTSOE_API_KEY"))
        self._cache = Cache()
        self._logger = logging.getLogger("eem")

        self._build_base_cache()

    def query(self, method: str, *args, check_only: bool = False, **kwargs):
        """
        Queries the client method with the provided arguments and caches the result.
        This method checks if the result of the query is already cached. If it is,
        it loads the result from the cache. Otherwise, it performs the query, caches
        the result, and then returns it.

        Args:
            method (str): The name of the client method to query.
            *args: Variable length argument list to pass to the client method.
            check_only (bool, optional): If True, only checks if the result is cached and does not perform the query, or return anything. Defaults to False.
            **kwargs: Arbitrary keyword arguments to pass to the client method.
        Returns:
            The result of the client method query, either loaded from cache or freshly queried.
        """
        # TODO: if time is within something that is contained in the base cache, re-construct from there
        #       this should almost always be possible (if the base case covers enough time), but should be done carefully to not introduce errors there

        cache_exist, cache_file = self._cache.has_get("entsoe", self._cache.mkid(method, *args, **kwargs))

        if check_only and cache_exist:
            return None

        if not cache_exist:
            result = getattr(self._client, method)(*args, **kwargs)
            result.to_pickle(cache_file)
            self._logger.debug(
                f"Caching {method}({', '.join([*map(repr, args), *[f'{k}={repr(v)}' for k, v in kwargs.items()]])}) to: {cache_file}"
            )

            if check_only:
                return None
        else:
            self._logger.debug(
                f"Loading {method}({', '.join([*map(repr, args), *[f'{k}={repr(v)}' for k, v in kwargs.items()]])}) from: {cache_file}"
            )
            result = pd.read_pickle(cache_file)

        assert "end" in kwargs, "The 'end' timestamp must be provided."
        return self._fix(result, ts_end=kwargs["end"])

    def get_history(self, method: str, *args, end: pd.Timestamp, **kwargs):
        """
        Run query, on all existing historical data in the base cache, for the specified method and arguments.

        Parameters:
        -----------
        method : str
            The method name to be used for querying data.
        *args : tuple
            Additional positional arguments to be passed to the query method.
        end : pd.Timestamp
            The exclusive end timestamp to filter the data.
        **kwargs : dict
            Additional keyword arguments to be passed to the query method.

        Returns:
        --------
        pd.DataFrame | pd.Series
            A dataframe/series containing the concatenated and sorted historical data.
        """
        data = pd.concat(
            [
                self.query(method, *args, **kwargs, start=tp["t0"], end=tp["t1"])
                for tp in EntsoePyWrapper.base_entries["time_periods"]
            ],
            axis=0,
        ).sort_index()
        return data[data.index < end]

    def _fix(self, data: pd.DataFrame | pd.Series, *, ts_end) -> pd.DataFrame | pd.Series:
        """
        Checks if the dataframe/series is indexed by hourly or 15-minute timestamps. If the index is hourly, returns the
        dataframe/series as is. If the index is 15-minute intervals, resamples it to hourly by taking the mean. If the
        "end" timestamp is present in the index, it is dropped --- this is useful for ensuring that the "end" timestamp
        is EXCLUSIVE.

        Raises a ValueError if neither of these conditions is met.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("The index must be a DatetimeIndex.")

        # Check if the index is hourly.
        hourly_diff = pd.Timedelta(hours=1)
        if all((data.index[i + 1] - data.index[i] == hourly_diff) for i in range(len(data) - 1)):
            return data.drop(index=ts_end, errors="ignore")

        # Check if the index is 15-minute intervals.
        quarter_hour_diff = pd.Timedelta(minutes=15)
        if all((data.index[i + 1] - data.index[i] == quarter_hour_diff) for i in range(len(data) - 1)):
            return data.resample("h").mean().drop(index=ts_end, errors="ignore")

        # If neither, raise an error.
        # TODO: This triggers, e.g., for "DE_LU" and "query_load_forecast". Remove the workaround below.
        self._logger.error("EntsoePyWrapper: `_fix` failed due to wrong time intervals")
        # raise ValueError(f"The data index must be hourly or 15-minute intervals")
        return data.resample("h").mean().drop(index=ts_end, errors="ignore")

    def _build_base_cache(self):
        # TODO: make sure to respect the "400 requests per minute" limit (happens automatically, but should be ensured)
        # TODO: split the base cache into smaller time periods to avoid re-downloading everything when adding a new "date" to its time period (due to immediate hash mismatch); example: split by days
        self._logger.debug("Base cache: Start building")
        self._base_cache_countries()
        self._base_cache_links()

    def _base_cache_countries(self):
        self._logger.debug("Base cache: Executing country queries")
        for tp in EntsoePyWrapper.base_entries["time_periods"]:
            self._logger.debug(f"Base cache: Time period = [{tp['t0']}, {tp['t1']}]")
            for country in EntsoePyWrapper.base_entries["countries"]:
                for query in EntsoePyWrapper.base_entries["country_queries"]:
                    self._logger.debug(f"Base cache: Request = [{query}, {country}]")
                    self.query(f"query_{query}", country, start=tp["t0"], end=tp["t1"], check_only=True)

    def _base_cache_links(self):
        self._logger.debug("Base cache: Executing link queries")
        for tp in EntsoePyWrapper.base_entries["time_periods"]:
            self._logger.debug(f"Base cache: Time period = [{tp['t0']}, {tp['t1']}]")
            for link in EntsoePyWrapper.base_entries["links"]:
                for query in EntsoePyWrapper.base_entries["link_queries"]:
                    self._logger.debug(f"Base cache: Request = [{query}, {link[0]} > {link[1]}]")
                    self.query(
                        f"query_{query['query']}", link[0], link[1], start=tp["t0"], end=tp["t1"], check_only=True
                    )
                    if query["both"]:
                        self._logger.debug(f"Base cache: Request = [{query}, {link[1]} > {link[0]}]")
                        self.query(
                            f"query_{query['query']}", link[1], link[0], start=tp["t0"], end=tp["t1"], check_only=True
                        )

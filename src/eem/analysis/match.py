import logging
import pandas as pd
from typing import NamedTuple

from ..entsoepywrapper import EntsoePyWrapper


class SquaredDistanceMatching:
    def __init__(self, epw: EntsoePyWrapper, task: dict, weights=None, scale="standardize"):
        self._logger = logging.getLogger("eem")

        self._epw = epw
        self._task = task
        self._weights = weights or dict(default=1.0)

        # TODO: implement "normalize"
        assert scale in ["standardize"], "Invalid scaling method."
        self._scale = scale

        self._t_now = None
        self._t_hist = None

        self._data = dict()
        self._prepare_data()

        self._matches = None
        self._match()

    def good(self, *, p=None, n=None):
        """
        Select a subset of the best matches based on the given proportion or number.

        Parameters:
        p (float, optional): Proportion of matches to select. Should be a value between 0 and 1.
        n (int, optional): Number of matches to select. If both p and n are provided, n takes precedence.

        Returns:
        list: A list containing the selected subset of matches.
        """
        n_matches = n or int(len(self._matches) * p)
        return self._matches[:n_matches]

    def others(self, *, p=None, n=None):
        """
        Selects a subset of matches from the list of matches based on the given proportion or number. The subset refers
        to the "in-between" matches --- those that "separate" the best and worst matches.

        Parameters:
        p (float, optional): The proportion of matches to select. Should be a value between 0 and 1.
        n (int, optional): The fixed number of matches to select. If both p and n are provided, n takes precedence.

        Returns:
        list: A list containing the selected subset of matches.
        """
        n_matches = n or int(len(self._matches) * p)
        margin = (len(self._matches) - n_matches) // 2
        return self._matches[margin : (margin + n_matches)]

    def bad(self, *, p=None, n=None):
        """
        Selects a subset of the worst matches based on the given proportion or number.

        Parameters:
        p (float, optional): A proportion of the total matches to select. Should be a value between 0 and 1.
        n (int, optional): Number of matches to select. If both p and n are provided, n takes precedence.

        Returns:
        list: A list containing the selected subset of matches.
        """
        n_matches = n or int(len(self._matches) * p)
        return self._matches[-n_matches:]

    def _match(self):
        self._logger.info("Matching task with historical data using `SquaredDistanceMatcher`.")
        dist = {(tn, th): 0.0 for tn in self._t_now for th in self._t_hist}

        # Add all distances to the dictionary.
        for entry in self._data:
            # Get scaled weights.
            w = self._weights.get(entry, self._weights["default"])

            self._logger.debug("Current step: calc_dist_country [country=%s, kpi=%s, weight=%f]", *entry, w)

            # Add to distance.
            data_now = self._data[entry]["now"]
            data_hist = self._data[entry]["hist"]
            for tn in self._t_now:
                for th in self._t_hist:
                    sq_dist = (data_now.loc[tn] - data_hist.loc[th]) ** 2
                    if isinstance(sq_dist, float):
                        dist[(tn, th)] += float(w * sq_dist)
                    else:
                        dist[(tn, th)] += float(w * sum(sq_dist))

        # Time delta of task query, in hours.
        dh = (self._task.t1 - self._task.t0).seconds // 3600

        # Get total distance of all possible matches.
        total_distances = dict()
        for th in self._t_hist:
            total_dist = 0.0
            valid = True
            for i in range(0, dh):
                if (k := (self._t_now[i], th + pd.Timedelta(hours=i))) in dist:
                    total_dist += dist[k]
                else:
                    valid = False
                    break

            if valid:
                total_distances[th] = total_dist

        # Sort all matches by total distance, and store them.
        self._matches = sorted(total_distances.items(), key=lambda x: x[1])

    # def _match(self):
    #     dist = dict()
    #     self._logger.info("Matching task with historical data using `SquaredDistanceMatcher`.")

    #     # Add all distances to the dictionary.
    #     self._calc_dist_timestamp(dist)
    #     self._calc_dist_country(dist)
    #     self._calc_dist_link(dist)

    #     # Time delta of task query, in hours.
    #     dh = (self._task.t1 - self._task.t0).seconds // 3600

    #     total_distances = dict()
    #     for th in self._t_hist:
    #         total_dist = 0.0
    #         valid = True
    #         for i in range(0, dh):
    #             if (k := (self._t_now[i], th + pd.Timedelta(hours=i))) in dist:
    #                 total_dist += dist[k]
    #             else:
    #                 valid = False
    #                 break

    #         if valid:
    #             total_distances[th] = total_dist

    #     self._matches = sorted(total_distances.items(), key=lambda x: x[1])

    # def _calc_dist_timestamp(self, dist: dict):
    #     self._logger.debug("Current step: calc_dist_timestamp")

    #     # Get scaled weights.
    #     w = {k: self._weights["general"]["time"] * v for (k, v) in self._weights["time"].items()}

    #     for tn in self._t_now:
    #         tn_prop = _get_time_properties(tn)
    #         for th in self._t_hist:
    #             dist[(tn, th)] = sum(w[k] * (tn_prop[k] - v) ** 2 for (k, v) in _get_time_properties(th).items())

    # def _calc_dist_country(self, dist: dict):
    #     # Get scaled weights.
    #     w = {k: self._weights["general"]["country"] * v for (k, v) in self._weights["country"].items()}

    #     for (kpi, weight) in w.items():
    #         if kpi in self._task.kpis:
    #             # Skip KPIs that are part of the task.
    #             continue

    #         for country in self._epw.base_entries["countries"]:
    #             self._logger.debug("Current step: calc_dist_country [country=%s, kpi=%s]", country, kpi)
    #             data_now = self._data["countries"][country][kpi]["now"]
    #             data_hist = self._data["countries"][country][kpi]["hist"]

    #             for tn in self._t_now:
    #                 for th in self._t_hist:
    #                     sq_dist = (data_now.loc[tn] - data_hist.loc[th]) ** 2
    #                     if isinstance(sq_dist, float):
    #                         dist[(tn, th)] += float(weight * sq_dist)
    #                     else:
    #                         dist[(tn, th)] += float(weight * sum(sq_dist))

    # def _calc_dist_link(self, dist: dict):
    #     # Get scaled weights.
    #     w = {k: self._weights["general"]["link"] * v for (k, v) in self._weights["link"].items()}

    #     for tn in self._t_now:
    #         for th in self._t_hist:
    #             continue  # TODO

    def _prepare_data(self):
        for country in self._task.countries:
            for kpi in self._task.kpis:
                _now = self._epw.query(f"query_{kpi}", country, start=self._task.t0, end=self._task.t1)
                _hist = self._epw.get_history(f"query_{kpi}", country, end=self._task.t0)
                _mean, _std = _hist.mean(), _hist.std()
                self._data[(country, kpi)] = dict(now=(_now - _mean) / _std, hist=(_hist - _mean) / _std)

        # TODO: the following may fail if some data sets are not fully aligned
        if self._t_now is None:
            self._t_now = self._data[(country, kpi)]["now"].index

        if self._t_hist is None:
            self._t_hist = self._data[(country, kpi)]["hist"].index

    #         self._data["countries"][country] = dict()
    #         self._prepare_country_data(country)

    #     for link in self._epw.base_entries["links"]:
    #         self._data["links"][link] = dict()
    #         self._prepare_link_data(link)

    # def _prepare_country_data(self, country: str):
    #     for kpi in self._weights["country"].keys():
    #         _now = self._epw.query(f"query_{kpi}", country, start=self._task.t0, end=self._task.t1)
    #         _hist = self._epw.get_history(f"query_{kpi}", country, end=self._task.t0)
    #         _mean, _std = _hist.mean(), _hist.std()
    #         self._data["countries"][country][kpi] = dict(now=(_now - _mean) / _std, hist=(_hist - _mean) / _std)

    #     if self._t_now is None:
    #         self._t_now = self._data["countries"][country][kpi]["now"].index
    #     if self._t_hist is None:
    #         self._t_hist = self._data["countries"][country][kpi]["hist"].index

    # def _prepare_link_data(self, link: dict):
    #     # TODO
    #     pass

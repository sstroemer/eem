import pandas as pd
import numpy as np

from ..entsoepywrapper import EntsoePyWrapper


def _est_range_perc(df: pd.DataFrame, range: tuple[float, float]) -> tuple[float, float]:
    sorted_values = np.sort(df.values)
    return (
        float(np.searchsorted(sorted_values, range[0], side="left") / len(df)),
        float(np.searchsorted(sorted_values, range[1], side="right") / len(df)),
    )


def describe(epw: EntsoePyWrapper, task: dict):
    descr = dict()

    for country, kpi in zip(task.countries, task.kpis):
        # Get KPI for the current task.
        data_kpi_now = epw.query(f"query_{kpi}", country, start=task.t0, end=task.t1)
        data_kpi_now_range = (float(data_kpi_now.min()), float(data_kpi_now.max()))

        # Get the historical data for the KPI (before "t0").
        data_kpi_hist = epw.get_history(f"query_{kpi}", country, end=task.t0)

        # Get percentiles of the current KPI range in the historical data.
        p = _est_range_perc(data_kpi_hist, data_kpi_now_range)
        descr[(country, kpi)] = {"perc_min": p[0], "perc_max": p[1]}

    return descr

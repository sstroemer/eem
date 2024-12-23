import logging
import pandas as pd
import numpy as np
import scipy.stats

from ..entsoepywrapper import EntsoePyWrapper


def _f_test_pvalue(s1: list, s2: list):
    # Use `ddof=1` for "unbiased" variance estimation ("sample variance").
    var = [np.var(s1, ddof=1), np.var(s2, ddof=1)]
    dof = [len(s1) - 1, len(s2) - 1]

    # Check for special cases that would result in division by zero, or similar.
    eps = np.finfo(float).eps
    if abs(var[0] - var[1]) < eps:
        return 1.0
    if abs(var[0]) < eps or abs(var[1]) < eps:
        return 0.0

    F = var[0] / var[1]
    if F > 1.0:
        F = 1 / F

    return 2 * (1 - scipy.stats.f.cdf(F, dof[0], dof[1]))


def _est_p_values(s1: list, s2: list):
    # Use `equal_var=False` for Welch's t-test (different lengths / variances).
    return float(scipy.stats.ttest_ind(s1, s2, equal_var=False)[1]), float(_f_test_pvalue(s1, s2))


def compare_matching(task, matching, *, p=None, n=None):
    logger = logging.getLogger("eem")
    logger.info("Begin comparison of matching results.")

    # NOTE: this could also include `oth = matching.others(p=p, n=n)`
    matches = dict(pos=matching.good(p=p, n=n), neg=matching.bad(p=p, n=n * 3))

    data_compare = dict()
    for country in EntsoePyWrapper.base_entries["countries"]:
        for kpi in EntsoePyWrapper.base_entries["country_queries"]:
            if kpi in task.kpis:
                continue

            data_hist = task._epw.get_history(f"query_{kpi}", country, end=task.t0)

            for mtype in matches.keys():
                for t0, dist in matches[mtype]:
                    dh = pd.Timedelta(hours=(task.t1 - task.t0).seconds // 3600 - 1)
                    data_hist_match = data_hist.loc[t0 : t0 + dh]

                    if isinstance(data_hist_match, pd.Series):
                        if (k := (country, kpi)) not in data_compare:
                            data_compare[k] = dict(pos=[], neg=[])
                        data_compare[k][mtype].extend(data_hist_match.tolist())
                    elif isinstance(data_hist_match, pd.DataFrame):
                        for column in data_hist_match.columns:
                            val = data_hist_match[column]

                            if isinstance(column, tuple):
                                if column[1] == "Actual Consumption":
                                    continue
                                elif column[1] == "Actual Aggregated":
                                    if (column[0], "Actual Consumption") in data_hist_match.columns:
                                        val -= data_hist_match[(column[0], "Actual Consumption")]

                                    # Just use the main technology.
                                    k = (country, kpi, column[0].lower().replace(" ", "_"))
                                else:
                                    # This is necessary to handle multi-level columns in DFs.
                                    k = (country, kpi, *[c.lower().replace(" ", "_") for c in column])
                            else:
                                k = (country, kpi, column.lower().replace(" ", "_"))

                            if k not in data_compare:
                                data_compare[k] = dict(pos=[], neg=[])

                            data_compare[k][mtype].extend(val.tolist())
                    else:
                        raise ValueError("Invalid data type.")

    differences = []
    for index, sample in data_compare.items():
        # Check if the samples are "empty" (happens for zero-only columns).
        if all(s == 0 for s in sample["pos"]) and all(s == 0 for s in sample["neg"]):
            logger.debug(f"Skipping empty sample: {index}")
            continue

        # Check if the samples are element-wise identical (happens for "dummy" columns like "generation, other").
        if all(s1 == s2 for s1, s2 in zip(sample["pos"], sample["neg"])):
            logger.debug(f"Skipping identical sample: {index}")
            continue

        logger.debug(f"Comparing sample: {index}")

        # Construct the KPI name.
        if len(index) == 2:
            kpi_name = index[1]
        if len(index) == 3:
            kpi_name = f"{index[2]}; source: {index[1]}"
        if len(index) == 4:
            kpi_name = f"{index[2]} ({index[3]}); source: {index[1]}"

        # Get the p-values for the mean and variance tests.
        p_values = _est_p_values(sample["pos"], sample["neg"])

        # Get the means and variances.
        means = (float(round(np.mean(sample["pos"]), 1)), float(round(np.mean(sample["neg"]), 1)))
        vars = (np.var(sample["pos"], ddof=1), np.var(sample["neg"], ddof=1))

        # Save the differences.
        if not np.isnan(p_values[0]):
            differences.append(
                dict(
                    type="mean",
                    p=p_values[0],
                    country=index[0],
                    kpi=kpi_name,
                    val_pos=means[0],
                    val_neg=means[1],
                    change=abs((means[0] - means[1]) / max(1, abs(means[1]))),
                )
            )
        if not np.isnan(p_values[1]):
            differences.append(
                dict(
                    type="var",
                    p=p_values[1],
                    country=index[0],
                    kpi=kpi_name,
                    dir="hi" if vars[0] > vars[1] else "lo",
                    change=abs((vars[0] - vars[1]) / max(1, abs(vars[1]))),
                )
            )

    # Sort the differences by p-value and return them.
    return sorted(differences, key=lambda x: x["p"])


def _fmt_diff_to_md(differences: list, n_mean: int = 4, n_var: int = 1, n_neg: int = 1, p_threshold: float = 0.025):
    # Ensure the sorting internally.
    differences = sorted(differences, key=lambda x: x["p"] / x["change"])
    rev_diff = differences[::-1]

    entries = dict(mean=[], var=[], negative=[])

    # Detect all significant differences.
    for diff in differences:
        if (diff["type"] == "mean") and (len(entries["mean"]) < n_mean):
            if diff["p"] < p_threshold:
                entries["mean"].append(diff)
        elif (diff["type"] == "var") and (len(entries["var"]) < n_var):
            if diff["p"] < p_threshold:
                entries["var"].append(diff)

    # Detect all non-significant differences.
    for diff in rev_diff:
        if (diff["type"] == "mean") and (len(entries["negative"]) < n_neg):
            if diff["p"] > 2.0 * p_threshold:
                entries["negative"].append(diff)

    ret = ""

    if len(entries["mean"]) > 0:
        ret += "1. We observe the following significant differences for MEANS:\n"
        for entry in entries["mean"]:
            ret += f"- The mean hourly value of KPI \"{entry['kpi']}\" in country \"{entry['country']}\" is `{entry['val_pos']}` during the investigated time period\n"
            ret += f"- The historic mean hourly value of KPI \"{entry['kpi']}\" in country \"{entry['country']}\" is `{entry['val_neg']}` during times which are distinct from the investigated time period\n\n"
    else:
        ret += "We do not observe any significant differences for any mean KPIs.\n"

    ret += "\n\n"

    if len(entries["var"]) > 0:
        ret += "2. We observe the following significant differences for VARIANCES:\n"
        for entry in entries["var"]:
            dir = "HIGHER" if entry["dir"] == "hi" else "LOWER"
            ret += f"- The variance of hourly value  of KPI \"{entry['kpi']}\" in country \"{entry['country']}\" during the given time interval is {dir} --- compared to times which are distinct from the investigated time period\n"
    else:
        ret += "We do not observe any significant differences for any variances of any KPIs.\n"

    ret += "\n\n"

    if len(entries["negative"]) > 0:
        ret += "3. There are factors that we cannot identify as having had any impact on market outcomes:\n"
        for entry in entries["negative"]:
            ret += f"- The mean hourly value of KPI \"{entry['kpi']}\" in country \"{entry['country']}\" does not show any significant difference between the investigated time periods and distinct historic times; no conclusions, at all (!), can be drawn from this factor.\n"
    else:
        ret += "All investigated factors seem to be significantly different during the investigated time period --- this is unusual!\n"

    return ret + "\n"

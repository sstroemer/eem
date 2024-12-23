# EEM: Explain Electricity Markets

**Important:**  
This repository serves ONLY as PROOF-OF-CONCEPT. It is in no way a product or service or anything similar. LLM-based
result summaries can be partially wrong (play around and you'll find some, depending on how good your model of choice
is). Further, the current matching/sorting/analyzing workflow is overly simplified.

> Make sure to only use models with appropriate licenses, and do not copy paste the placeholders in this README.

## About

What this is?

1. Create a "task", e.g., "why are electricity prices in X the way they are during time Y?"
2. Analyse it, comparing to historic data, based on multiple KPIs
3. Create some "take-aways" from the observed differences

And (4.) ... use a LLM to dynamically create summaries/reports from those KPIs, trying to get some automated
explainability into the results (using certain guidelines in the different prompts).

Doing that manually, for a given situation, will always be better and more sophisticated -- but not everyone out there
can actually do that.

## The `.env` file

Put the following keys/tokens in there:

```text
ENTSOE_API_KEY=asd123415-asd324-asd1-4111-1135f3fgfg
HUGGINGFACE_ACCESS_TOKEN=hf_asdfg23452sdfubg8792g34t7
```

## Updating packages

Currently the proposed way is:

```shell
uv lock --upgrade
uv sync
```

## Downloading a model manually

Run

```python
from llama_cpp import Llama

Llama.from_pretrained(repo_id="user/model-xyz-gguf", filename="*some_quant_filter.gguf")
```

## Example task

```python
import eem
import pandas as pd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~ BASE SECTION                                                                                                   ~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Initialize an EntsoePyWrapper instance, which is used to query data.
epw = eem.EntsoePyWrapper()

# Define a task that specifies the country, KPI, and time period to analyze.
task = eem.Task(
    epw,
    countries=["AT"],
    kpis=["day_ahead_prices"],
    t0=pd.Timestamp(year=2024, month=12, day=12, hour=15, minute=0, tz="Europe/Vienna"),
    t1=pd.Timestamp(year=2024, month=12, day=12, hour=19, minute=0, tz="Europe/Vienna"),
)

# Create a matching, that finds (non-)similar time periods in the past.
matching = eem.analysis.SquaredDistanceMatching(task)

# Calculate all differences (using p-values for their comparison internally).
differences = eem.analysis.compare_matching(task, matching, n=15)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~ LLM SECTION                                                                                                    ~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

installed_models = {
    1: dict(repo_id = "MaziyarPanahi/Qwen2.5-1.5B-Instruct-GGUF", filename = "*Q4_K_S.gguf", n_ctx=2**11, seed=42),
    2: dict(repo_id = "MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF", filename = "*Q8_0.gguf", n_ctx=2**15, seed=42),
    3: dict(repo_id = "MaziyarPanahi/Mistral-Nemo-Instruct-2407-GGUF", filename = "*Q8_0.gguf", n_ctx=2**18, seed=42),
}

llm = eem.llm.BasicSummarizer(task, **installed_models[3], verbose=False)
ans = llm.execute_all_steps(differences)

llm = None  # clean-up memory!
```

### Answer

<details>

<summary>Click to show answer</summary>

```markdown
# Task

Analyze the day-ahead electricity prices in Austria from 12:00 to 19:00 on December 12, 2024.

# Analysis

## Overview

During the inspected period, the day-ahead electricity prices in Austria were consistently higher than the historical average.

## Details

### Mean

**Highlights:**
1. Solar generation in Germany (DE_LU) is extremely low.
2. Wind and solar forecast in Germany (DE_LU) is significantly lower than usual, being 1581.9 MWh compared to 32102.4 MWh during other times.
3. Net position in Germany (DE_LU) is unusually negative, indicating high imports.

**Implications:**
The low solar generation and wind and solar forecast in Germany (DE_LU) might lead to increased reliance on fossil fuels for generation, meanwhile the high imports suggest that Germany (DE_LU) is meeting its demand through imports, potentially driving up prices due to increased demand and reliance on expensive fossil fuels.

### Variance

**Highlights:**
1. The variance of fossil gas generation in Austria (AT) is significantly higher than usual.

**Implications:**
The high variance in fossil gas generation in Austria (AT) suggests increased uncertainty and volatility in the energy market, potentially leading to price fluctuations and increased risk for market participants.

# Summary

In Austria, day-ahead electricity prices were consistently higher than the historical average, driven by low renewable generation and increased reliance on expensive fossil fuels. High variance in fossil gas generation introduced market uncertainty, potentially exacerbating price volatility. Notably, hydro pumped storage generation in Austria did not significantly impact the market during this period. Overall, the Austrian electricity market experienced unusual conditions, with low renewable generation and increased fossil fuel reliance driving up prices and introducing market uncertainty.
```

</details>

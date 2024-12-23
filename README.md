# EEM: Explain Electricity Markets

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
    1: dict(repo_id = "MaziyarPanahi/Qwen2.5-1.5B-Instruct-GGUF", filename = "*Q4_K_S.gguf", n_ctx=2048, seed=42),
    2: dict(repo_id = "MaziyarPanahi/Qwen2.5-7B-Instruct-GGUF", filename = "*Q8_0.gguf", n_ctx=32768, seed=42),
}

llm = eem.llm.BasicSummarizer(task, **installed_models[2], verbose=False)
llm.execute_all_steps(differences)
```

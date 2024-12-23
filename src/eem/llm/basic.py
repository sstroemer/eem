import os
from llama_cpp import Llama
from rich.console import Console
from rich.markdown import Markdown

from ..task import Task
from ..analysis.compare import _fmt_diff_to_md


class BasicSummarizer:
    _guide = (
        "You must use the following details to guide your answer and interpretation:\n"
        "- `net_position` is the current import/export state of a country\n"
        "- Negative (lower) `net_position` means a country is importing!\n"
        "- Positive (higher) `net_position` means a country is exporting!\n"
        "- Generation, load, and similar values have the unit `MWh`.\n"
        "- Prices are in `EUR/MWh`.\n"
        "- Renewable generation (solar, wind, etc.) are heavily influenced by weather conditions.\n"
        "- 'other market conditions', 'different time periods', etc. are all synonyms for 'historic times during which market results, that the analysis accounted for, are significantly different from the one studied in this task'.\n"
        "- If the reason for something is not clear or highly likely, it is better to not make any assumptions -- saying 'unknown reasons' is fine.\n"
        "- Contents inside <<>> are placeholders for the actual values you need to fill in, e.g., specific countries or KPIs.\n"
        "- You MUST ALWAYS refer to KPIs with a human readable name and not the technical name that is supplied.\n"
        "- Renewable generation is most-often inexpensive\n"
        "- Fossil generation often comes with higher marginal costs, driving up prices\n"
    )

    def __init__(self, task: Task, **kwargs):
        self._interactions = []

        self._model = Llama.from_pretrained(**kwargs)
        self._console = Console()

        self._task = task

        # TODO: This is EXTREMELY hardcoded to "1 country, 1 KPI" tasks.
        assert len(self._task.countries) == 1, "Only 1 country is supported."
        assert len(self._task.kpis) == 1, "Only 1 KPI is supported."

    def execute_all_steps(self, differences: list, echo: bool = True):
        self.step_task_summary(echo=echo)
        self.step_kpi_summary(echo=echo)
        self.step_mean(differences, echo=echo)
        self.step_variance(echo=echo)
        self.step_finalize(echo=echo)

        return "\n\n".join([ia[2] + ia[1] for ia in self._interactions])

    def step_task_summary(self, *, echo: bool = False):
        request = {
            "system": (
                f"You are given the task to analyze:\n"
                f"- The KPI `{self._task.kpis[0]}`\n"
                f"- For the country `{self._task.countries[0]}`\n"
                f"- During the investigated / studied time period from `{self._task.t0}` to `{self._task.t1}`\n"
            ),
            "user": (
                f"Create a single sentence to summarize the task: Include the full country name, a human readable name "
                f"of the KPI, and give the inspected time period in a short format. The end timestamp is non-inclusive."
                f"Only return one sentence. DO NOT REPEAT THE TASK DESCRIPTION."
            ),
        }

        return self._llm_execute_fenced_request(request, prefix="# Task\n\n", echo=echo)

    def step_kpi_summary(self, *, echo: bool = False):
        descr = self._task.description[(self._task.countries[0], self._task.kpis[0])]

        str_descr = ""
        if descr["perc_min"] > 0.5:
            val = round(descr["perc_min"] * 100, 1)
            str_descr += f"- The minimal / lowest `{self._task.kpis[0]}` during the inspected period was higher than {val}% of historical `{self._task.kpis[0]}`s\n"
        else:
            val = round((1 - descr["perc_min"]) * 100, 1)
            str_descr += f"- The minimal / lowest `{self._task.kpis[0]}` during the inspected period was lower than {val}% of historical `{self._task.kpis[0]}`s\n"
        if descr["perc_max"] > 0.5:
            val = round(descr["perc_max"] * 100, 1)
            str_descr += f"- The maximal / highest `{self._task.kpis[0]}` during the inspected period was higher than {val}% of historical `{self._task.kpis[0]}`s\n"
        else:
            val = round((1 - descr["perc_max"]) * 100, 1)
            str_descr += f"- The maximal / highest `{self._task.kpis[0]}` during the inspected period was lower than {val}% of historical `{self._task.kpis[0]}`s\n"

        request = {
            "system": (f"You know the following high-level information:\n{str_descr}"),
            "user": (
                f"Summarize your knowledge about the high-level information given, in relation to the task that you already described, in one sentence."
            ),
        }

        return self._llm_execute_fenced_request(request, prefix="# Analysis\n\n## Overview\n\n", echo=echo)

    def step_mean(self, differences: list, *, echo: bool = False):
        stats_info = _fmt_diff_to_md(differences)

        request = {
            "system": (
                f"You are given the following background information from a statistical analysis:\n\n"
                f"{stats_info}\n\n"
                f"{BasicSummarizer._guide}\n\n"
                f"A TEMPLATE ANSWER LOOKS LIKE THIS:\n\n---\n"
                f"**Highlights:**\n"
                f"1. The <<KPI>> in <<COUNTRY>> (<<COUNTRY SHORTCODE>>) seems to be extremely <<HIGH/LOW>>.\n"
                f"2. The <<KPI>> in <<COUNTRY>> (<<COUNTRY SHORTCODE>>) is <<HIGHER/LOWER>> than usual, being <<OBSERVED VALUE>> compared to <<OTHER VALUE>> during other times.\n"
                f"3. The <<KPI>> in <<COUNTRY>> is <<HIGHER/LOWER>> than expected.\n\n"
                f"**Implications:**\n"
                f"The <<HIGH/LOW>> <KPI>> in <<COUNTRY>> might lead to <<HIGHER/LOWER>> <KPI>>; <<MEANWHILE / HOWEVER>> the <<HIGH/LOW>> <KPI>> in <<COUNTRY>> and unusually <<HIGH/LOW>> <KPI>> in <<COUNTRY>>.\n"
                f"---\n\n"
            ),
            "user": (
                f"Explain the results and their implications only based on the background information related to MEAN results. Do not copy the template answer verbatim but use its style."
            ),
        }

        return self._llm_execute_fenced_request(request, prefix="## Details\n\n### Mean\n\n", echo=echo)

    def step_variance(self, *, echo: bool = False):
        request = {
            "system": "",
            "user": (
                f"Make use of the previously given background information, as well as the guide and template, "
                f"to explain the results and their implications related to VARIANCE results. "
                f"Do not copy the template answer. Focus on the variance results only."
            ),
        }

        return self._llm_execute_fenced_request(request, prefix="### Variance\n\n", echo=echo)

    def step_finalize(self, *, echo: bool = False):
        request = {
            "system": "An executive summary does not contain headers, bold text, lists, or similar formatting. It is concise and to the point.",
            "user": (
                f"Finalize the analysis now, by giving an executive summary of the whole task and analysis."
                f"You should include - based on the previously given background information and guide - a small note related to "
                f"step (3.) which touches upon KPIs that seem to have no impact. Do not copy the template answer. Maximum four sentences."
                f"Finalize the text with the derivation of the high-level / overview insights on the task, triggered by the effects you analyzed."
            ),
        }

        return self._llm_execute_fenced_request(request, prefix="# Summary\n\n", echo=echo)

    def _llm_execute_fenced_request(self, request: dict, *, prefix: str = "", echo: bool):
        prompt = ""
        prompt += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        prompt += "<|im_start|>system\n"
        prompt += (
            "You act as electricity market expert, you do not guess, and only make use of knowledge that you have.\n\n"
        )

        if len(self._interactions) > 0:
            # Include previous interactions.
            for ia in self._interactions:
                prompt += ia[0]["system"]
                prompt += "<|im_end|>\n<|im_start|>user\n"
                prompt += ia[0]["user"]
                prompt += "<|im_end|>\n<|im_start|>assistant\n"
                prompt += ia[1]
                prompt += "<|im_end|>\n<|im_start|>system\n"

        prompt += request["system"]
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>user\n"
        prompt += request["user"]
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        # Can be used to measure "used up tokens" for the prompt:
        # prompt_tokens = len(llm.tokenize(prompt.encode(), special=True))

        prev_output = ""
        for ia in self._interactions:
            prev_output += ia[2] + ia[1] + "\n\n"

        answer = ""
        for ans in self._model(prompt, stop=["<|im_end|>"], stream=True, temperature=0.0, max_tokens=None):
            answer += ans["choices"][0]["text"]
            if echo:
                self._trm_cls()
                self._trm_move(0, 0)
                self._console.print(Markdown(prev_output + prefix + answer))

        self._interactions.append((request, answer, prefix))
        return answer

    def _trm_move(self, line, col):
        print("\033[%d;%dH" % (line, col))

    def _trm_cls(self):
        os.system("cls" if os.name == "nt" else "clear")

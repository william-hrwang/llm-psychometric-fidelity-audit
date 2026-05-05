from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_CSV_PATH = PROJECT_ROOT / "models.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
ERROR_DIR = PROJECT_ROOT / "logs" / "error"

DEFAULT_MAX_TOKENS = 2
REQUEST_DELAY_SECONDS = 0.5
DEFAULT_TEMPERATURE = 1.0
DEFAULT_REASONING_ENABLED = False
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


QUESTIONS = {
    "EXT1": "I am the life of the party.",
    "EXT2": "I don't talk a lot.",
    "EXT3": "I feel comfortable around people.",
    "EXT4": "I keep in the background.",
    "EXT5": "I start conversations.",
    "EXT6": "I have little to say.",
    "EXT7": "I talk to a lot of different people at parties.",
    "EXT8": "I don't like to draw attention to myself.",
    "EXT9": "I don't mind being the center of attention.",
    "EXT10": "I am quiet around strangers.",
    "EST1": "I get stressed out easily.",
    "EST2": "I am relaxed most of the time.",
    "EST3": "I worry about things.",
    "EST4": "I seldom feel blue.",
    "EST5": "I am easily disturbed.",
    "EST6": "I get upset easily.",
    "EST7": "I change my mood a lot.",
    "EST8": "I have frequent mood swings.",
    "EST9": "I get irritated easily.",
    "EST10": "I often feel blue.",
    "AGR1": "I feel little concern for others.",
    "AGR2": "I am interested in people.",
    "AGR3": "I insult people.",
    "AGR4": "I sympathize with others' feelings.",
    "AGR5": "I am not interested in other people's problems.",
    "AGR6": "I have a soft heart.",
    "AGR7": "I am not really interested in others.",
    "AGR8": "I take time out for others.",
    "AGR9": "I feel others' emotions.",
    "AGR10": "I make people feel at ease.",
    "CSN1": "I am always prepared.",
    "CSN2": "I leave my belongings around.",
    "CSN3": "I pay attention to details.",
    "CSN4": "I make a mess of things.",
    "CSN5": "I get chores done right away.",
    "CSN6": "I often forget to put things back in their proper place.",
    "CSN7": "I like order.",
    "CSN8": "I shirk my duties.",
    "CSN9": "I follow a schedule.",
    "CSN10": "I am exacting in my work.",
    "OPN1": "I have a rich vocabulary.",
    "OPN2": "I have difficulty understanding abstract ideas.",
    "OPN3": "I have a vivid imagination.",
    "OPN4": "I am not interested in abstract ideas.",
    "OPN5": "I have excellent ideas.",
    "OPN6": "I do not have a good imagination.",
    "OPN7": "I am quick to understand things.",
    "OPN8": "I use difficult words.",
    "OPN9": "I spend time reflecting on things.",
    "OPN10": "I am full of ideas.",
}
QUESTION_KEYS = list(QUESTIONS.keys())

SYSTEM_TEMPLATE = """\
You are a human participant completing an online personality survey.
Answer every item honestly and consistently based on who you are.

You will receive 50 statements. Rate each on a scale of 1–5:
  1 = Disagree
  2 = Slightly disagree
  3 = Neutral
  4 = Slightly agree
  5 = Agree

Return ONLY a valid JSON object mapping each item key to its integer score.
No explanation, no markdown fences, no extra keys. Example format:
{{"EXT1": 4, "EXT2": 2, ...}}
"""


@dataclass(frozen=True)
class ModelConfig:
    index: int
    family: str
    model_name: str
    api_model_name: str
    max_tokens: int

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Big Five survey for one model or a row range from latest_models.csv.",
    )
    parser.add_argument(
        "index",
        type=str,
        help=(
            "1-based row index in models/latest_models.csv, or inclusive range "
            "formatted as [start,end]."
        ),
    )
    parser.add_argument(
        "--run_time",
        type=int,
        default=2,
        help="How many times to run the selected model.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default="",
        help="Unique suffix for output file (e.g. SLURM array task ID).",
    )
    return parser.parse_args()


def parse_index_selection(raw_value: str) -> list[int]:
    value = raw_value.strip()
    if not value:
        raise ValueError("index must not be empty")

    range_match = re.fullmatch(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", value)
    if range_match:
        start_index = int(range_match.group(1))
        end_index = int(range_match.group(2))
        if start_index < 1 or end_index < 1:
            raise ValueError("range values must be >= 1")
        if start_index > end_index:
            raise ValueError("range start must be <= end")
        return list(range(start_index, end_index + 1))

    try:
        single_index = int(value)
    except ValueError as error:
        raise ValueError(
            "index must be an integer (e.g. 58) or range (e.g. [55,58])"
        ) from error

    if single_index < 1:
        raise ValueError("index must be >= 1")
    return [single_index]


def build_user_prompt() -> str:
    lines = ["Here are the 50 items:\n"]
    for key, text in QUESTIONS.items():
        lines.append(f"{key}: {text}")
    lines.append("\nRespond with a JSON object only.")
    return "\n".join(lines)


def normalize_api_model_name(_family: str, model_name: str) -> str:
    family = _family.strip().lower()
    normalized_model = model_name.strip()
    if "/" in normalized_model:
        return normalized_model
    return f"{family}/{normalized_model}"


def parse_max_tokens(raw_value: str) -> int:
    token_text = raw_value.strip() if raw_value else ""
    if not token_text:
        return DEFAULT_MAX_TOKENS

    max_tokens = int(token_text)
    if max_tokens < 1:
        raise ValueError(f"max token must be >= 1, got {max_tokens}")
    return max_tokens


def load_model_config(index: int) -> ModelConfig:
    if index < 1:
        raise ValueError("index must be >= 1")

    if not MODELS_CSV_PATH.exists():
        raise FileNotFoundError(f"latest_models.csv not found: {MODELS_CSV_PATH}")

    last_row_index = 0
    with MODELS_CSV_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            last_row_index = row_index
            if row_index != index:
                continue

            family = (row.get("model family") or "").strip()
            model_name = (row.get("model name") or "").strip()
            max_tokens = parse_max_tokens(row.get("max token") or "")

            if not family:
                raise ValueError(f"Row {index} is missing model family.")
            if not model_name:
                raise ValueError(f"Row {index} is missing model name.")

            return ModelConfig(
                index=row_index,
                family=family,
                model_name=model_name,
                api_model_name=normalize_api_model_name(family, model_name),
                max_tokens=max_tokens,
            )

    raise IndexError(
        f"index {index} is out of range for {MODELS_CSV_PATH} "
        f"(available rows: 1-{last_row_index})"
    )


def build_client(_family: str) -> OpenAI:
    api_key = os.getenv(OPENROUTER_API_KEY_ENV_VAR)
    if not api_key:
        raise EnvironmentError(
            f"Missing environment variable {OPENROUTER_API_KEY_ENV_VAR} for OpenRouter."
        )

    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def make_safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip())
    return cleaned or "model"


def get_output_path(model_name: str, suffix: str = "") -> Path:
    suffix_part = f"_{suffix}" if suffix else ""
    return RESULTS_DIR / f"big_five_{make_safe_filename(model_name)}{suffix_part}.csv"


def get_error_path(model_name: str) -> Path:
    return ERROR_DIR / f"{make_safe_filename(model_name)}.txt"


def strip_code_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    parts = cleaned.split("```")
    if len(parts) < 2:
        return cleaned

    fenced = parts[1].strip()
    if fenced.startswith("json"):
        fenced = fenced[4:].strip()
    return fenced


def extract_message_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue

            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
                continue

            text_value = getattr(item, "text", None)
            if isinstance(text_value, str):
                text_parts.append(text_value)

        return "".join(text_parts).strip()

    return str(content).strip()


def validate_scores(raw_output: str, model_name: str) -> dict[str, int | str]:
    cleaned_output = strip_code_fences(raw_output)
    if not cleaned_output:
        raise ValueError("Model returned empty message content.")

    scores = json.loads(cleaned_output)
    if not isinstance(scores, dict):
        raise ValueError("Response JSON must be an object.")

    row: dict[str, int | str] = {"model": model_name}
    for key in QUESTION_KEYS:
        if key not in scores:
            raise ValueError(f"Missing key in response JSON: {key}")

        value = int(scores[key])
        if not 1 <= value <= 5:
            raise ValueError(f"{key}={value} is out of range 1-5")

        row[key] = value

    return row


def run_survey(client: OpenAI, model_config: ModelConfig) -> dict[str, int | str]:
    response = client.chat.completions.create(
        model=model_config.api_model_name,
        max_completion_tokens=model_config.max_tokens,
        temperature=DEFAULT_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE.strip()},
            {"role": "user", "content": build_user_prompt()},
        ],
        extra_body={"reasoning": {"enabled": DEFAULT_REASONING_ENABLED}},
    )

    if not response.choices:
        raise ValueError("Model returned no choices.")

    content = response.choices[0].message.content
    print(content)
    return validate_scores(
        raw_output=extract_message_text(content),
        model_name=model_config.model_name,
    )


def append_error_log(error_path: Path, run_number: int, error: Exception) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    with error_path.open("a", encoding="utf-8") as handle:
        handle.write(f"run={run_number}\n")
        handle.write(
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
        )
        handle.write("\n")


def write_results(
    client: OpenAI,
    model_config: ModelConfig,
    run_time: int,
    output_path: Path,
    error_path: Path,
) -> tuple[int, int]:
    fieldnames = ["model"] + QUESTION_KEYS
    success_count = 0
    failure_count = 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    should_write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()

        for run_number in range(1, run_time + 1):
            print(
                f"Running {run_number}/{run_time} "
                f"for model={model_config.model_name}..."
            )

            try:
                row = run_survey(client, model_config)
                writer.writerow(row)
                handle.flush()
                success_count += 1
                print(
                    f"  OK  EXT1={row['EXT1']}, AGR4={row['AGR4']}, OPN10={row['OPN10']}"
                )
            except Exception as error:
                failure_count += 1
                append_error_log(error_path=error_path, run_number=run_number, error=error)
                print(f"  FAIL run={run_number}: {error}")

            if run_number < run_time:
                time.sleep(REQUEST_DELAY_SECONDS)

    return success_count, failure_count


def main() -> None:
    args = parse_args()

    if args.run_time < 1:
        raise ValueError("run_time must be >= 1")

    selected_indices = parse_index_selection(args.index)
    total_success = 0
    total_failure = 0

    for selected_index in selected_indices:
        model_config = load_model_config(selected_index)
        client = build_client(model_config.family)

        output_path = get_output_path(model_config.model_name, args.job_id)
        error_path = get_error_path(model_config.model_name)

        print(
            f"Selected row={model_config.index}, family={model_config.family}, "
            f"model={model_config.model_name}, api_model={model_config.api_model_name}, "
            f"max_tokens={model_config.max_tokens}"
        )
        print(f"Results file: {output_path}")
        print(f"Error file:   {error_path}")

        success_count, failure_count = write_results(
            client=client,
            model_config=model_config,
            run_time=args.run_time,
            output_path=output_path,
            error_path=error_path,
        )
        total_success += success_count
        total_failure += failure_count

        print(
            f"\nFinished row={model_config.index}: "
            f"success={success_count}, failure={failure_count}, results={output_path}"
        )
        if failure_count > 0:
            print(f"Errors were written to {error_path}")

    if len(selected_indices) > 1:
        print(
            f"\nAll rows finished: rows={len(selected_indices)}, "
            f"success={total_success}, failure={total_failure}"
        )


if __name__ == "__main__":
    main()
    # nohup python -u main.py "[55,58]" --run_time 300 2>&1 | tee output.log
    # nohup python -u main.py 61 --run_time 300 2>&1 | tee output.log
    # nohup python -u main.py 61 --run_time 200 2>&1 | tee output.log
    # python main.py 1 --run_time 2

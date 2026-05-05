# llm-psychometric-fidelity-audit

This repository contains the code and analysis notebooks for auditing the psychometric fidelity of large language model responses on the IPIP Big Five personality questionnaire. The project compares LLM-generated response profiles with cleaned human survey responses across central tendency, distributional shape, reliability, latent structure, and response-style behavior.

The accompanying dataset is hosted on Hugging Face:

<https://huggingface.co/datasets/Anonymous0624/llm-psychometric-fidelity-audit/tree/main>

A ready-to-run Google Colab version of the analysis is also available:

<https://colab.research.google.com/drive/1xrlhp9X2piBFk4BWCa8qITads7vfHrQB>

If you only want to inspect or reproduce the main results, the Colab notebook is the fastest option. It runs directly in the browser and uses the released analysis data package.

## Repository layout

```text
code/
  analysis/
    experiment/
      analysis_full.ipynb              # Main end-to-end analysis notebook
    figures/                           # Figure notebooks, scripts, and exports
  data/
    human_data/
      IPIP-FFM-data-8Nov2018/
        codebook.txt                   # IPIP item and column documentation
        data-final.csv                 # Raw human survey export
      data_process.ipynb               # Human-data quality-control notebook
      output/                          # QC outputs and cleaned human data
    llm_data/
      main.py                          # Collect LLM Big Five survey responses
      check.py                         # Check/backfill generated LLM CSV files
      models.csv                       # Model list used by the collection script
    result/
      big5_llm.csv                     # Merged LLM response data
      human_data_cleaned.csv           # Cleaned human sample data
```

## Data

The main analysis uses two released CSV files:

- `human_data_cleaned.csv`: cleaned human IPIP Big Five responses after quality control.
- `big5_llm.csv`: merged LLM responses to the same 50 IPIP items.

These files are available from the Hugging Face dataset linked above. The main notebook also downloads them automatically from Hugging Face if they are not already present in its local `./data/` folder.

To download the whole data package manually, use the Hugging Face web interface or clone the dataset repository with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/Anonymous0624/llm-psychometric-fidelity-audit hf-data
```

## Environment setup

Use Python 3.10 or newer. The analysis uses standard scientific Python packages plus several psychometric and statistical dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scipy matplotlib statsmodels factor_analyzer pingouin openpyxl jupyter openai pillow
```

The Colab notebook installs its notebook-specific dependencies in the first setup cell.

## Running the main analysis

Open and run:

```text
code/analysis/experiment/analysis_full.ipynb
```

The notebook expects its working directory to be `code/analysis/experiment/`. When run from that location, it creates `./data/` and automatically downloads:

```text
data/human_data_cleaned.csv
data/big5_llm.csv
```

It then writes tables, figures, and packaged outputs to:

```text
outputs/
big5_psychometric_fidelity_outputs.zip
```

The notebook evaluates five psychometric fidelity levels:

- Level 1: central tendency fidelity.
- Level 2: distribution fidelity.
- Level 3: reliability fidelity.
- Level 4: structural fidelity.
- Level 5: behavioral and response-style fidelity.

It also produces calibration baselines and diagnostic tables used by the figure notebooks.

## Running the Colab notebook

Use this link:

<https://colab.research.google.com/drive/1xrlhp9X2piBFk4BWCa8qITads7vfHrQB>

Run the setup cell first, then run the remaining cells in order. The Colab version is intended for direct reproduction of the released analysis outputs.

## Regenerating the cleaned human dataset

This step is only needed if you want to rerun the human-data quality-control pipeline from the raw IPIP survey export.

Download the raw IPIP Big Five data from Open Psychometrics:

<https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip>

Place the zip file under `code/data/human_data/` and unzip it so that the raw CSV is available at:

```text
code/data/human_data/IPIP-FFM-data-8Nov2018/data-final.csv
```

Then open and run:

```text
code/data/human_data/data_process.ipynb
```

The notebook writes:

```text
code/data/human_data/output/data-flagged-qc.csv
code/data/human_data/output/human_data_cleaned.csv
code/data/human_data/output/qc-flag-summary.csv
code/data/human_data/output/qc-flag-overlap.csv
```

The QC pipeline flags duplicate IP submissions, missing or out-of-range items, very fast responses, straightlining, long response strings, Mahalanobis outliers, and screen/test-duration outliers.

## Collecting new LLM responses

This step is only needed if you want to regenerate or extend the LLM response dataset. The released Hugging Face package already includes the merged `big5_llm.csv` used by the analysis.

The LLM collection code is in:

```text
code/data/llm_data/
```

The scripts use OpenRouter's OpenAI-compatible API endpoint. Before collecting new responses, set your API key:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

Run one model by 1-based row index from `models.csv`:

```bash
cd code/data/llm_data
python main.py 1 --run_time 2
```

Run an inclusive row range:

```bash
python main.py "[1,4]" --run_time 300 --job_id batch01
```

Outputs are written to:

```text
code/data/llm_data/results/
```

Errors are written to:

```text
code/data/llm_data/logs/error/
```

Use `check.py` to verify row counts and backfill missing runs:

```bash
python check.py "[1,4]" --check_times 300 --double_check --job_id batch01
```

## Notes

- The Big Five item text and raw survey column documentation are in `code/data/human_data/IPIP-FFM-data-8Nov2018/codebook.txt`.
- The LLM collection script asks each model to return a JSON object with 50 integer scores on a 1-5 Likert scale.
- API-based LLM data collection may incur provider costs and is not required for reproducing the released analysis.
- For paper reproduction, the recommended path is the released Hugging Face data package plus `analysis_full.ipynb` or the Colab notebook.

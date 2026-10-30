# RamanCopilot

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=fff)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=fff)](#)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](#)
[![LangChain](https://img.shields.io/badge/LangChain-ffffff?logo=langchain&logoColor=black)](#)
[![Windows](https://custom-icon-badges.demolab.com/badge/Windows-11-0078D6?logo=windows11&logoColor=white)](#)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](#)
[![Claude](https://img.shields.io/badge/Claude-D97757?logo=claude&logoColor=fff)](#)
[![OpenAI](https://custom-icon-badges.demolab.com/badge/OpenAI-412991?logo=openai&logoColor=white)](#)
[![Gemini](https://img.shields.io/badge/Gemini-8E75B2?logo=googlegemini&logoColor=white)](#)

RamanCopilot is a VLM agent framework for operating a Raman spectrometer. A single ReAct agent controls the stage, laser, Raman CCD and brightfield camera through tools, analyses the measured spectra, and reports the results. The agent runs on VLMs provided by Ollama, OpenAI, Anthropic and Google. This document describes how to install and run RamanCopilot and where to find the benchmark data.

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running RamanCopilot](#running-ramancopilot)
5. [Benchmark](#benchmark)
6. [License](#license)
7. [Contact](#contact)

---

## Getting Started
RamanCopilot requires Python 3.12 and Node.js (for the web interface).
Instrument control runs on Windows. Without the instrument, RamanCopilot can run in a virtual environment (see [Instrument](#instrument)).

---

## Installation

### Python packages
```bash
pip install -r requirements.txt
```
The `requirements.txt` file specifies the versions of the Python packages used in the experiment. Since installing these may alter the versions of existing packages, it is recommended to perform the installation within a Python 3.13 virtual environment (created via `python -m venv .venv`, followed by `.venv\Scripts\activate`).


### Web interface
```bash
cd frontend
npm install
```

---

## Configuration
All environment variables and API keys below go in the `.env` file at the project root.

### VLM
The VLM is set with `RAMAN_OLLAMA_MODEL` in the form `<provider>:<model>`. The provider is one of `anthropic`, `openai`, `google` or `ollama`; without a prefix, the model is served by Ollama.
```bash
RAMAN_OLLAMA_MODEL=anthropic:claude-sonnet-4-6
# RAMAN_OLLAMA_MODEL=openai:gpt-5
# RAMAN_OLLAMA_MODEL=google:gemini-3.1-pro-preview
# RAMAN_OLLAMA_MODEL=gemma4:31b
```
For Ollama models, set the Ollama server address with `RAMAN_OLLAMA_HOST` (or `OLLAMA_HOST`).
```bash
RAMAN_OLLAMA_HOST=http://localhost:11434
```
Per-model parameters are set in `backend/repo/llm/model_profiles.json`.

### API keys
Add the key for the provider you use.
```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

### Instrument
- RamanCopilot controls the WEVE Mantaray; running on real hardware requires a Mantaray.
- The Andor SDK is licensed by the instrument vendor and is not distributed.
- Instrument constants are read from `Config.ini`, the configuration file of the vendor software (the path is set in `backend/common/config/hw.py`).
- Without a Mantaray, set `RAMAN_VIRTUAL_HW=1` to run RamanCopilot in a virtual environment. The virtual environment reads `backend/common/config/virtual_config.ini`, which holds only the required instrument constants, instead of `Config.ini`.

### Objective
The objective magnification at startup is set with `RAMAN_OBJECTIVE` (`20x`, `50x` or `100x`; default `50x`).
```bash
RAMAN_OBJECTIVE=20x
```
While running, the objective can be switched with the toggle in the hardware panel of the web interface.

---

## Running RamanCopilot
**1.** Start the backend server (port 8000)
```bash
python -m backend.controller.web.main
```

**2.** Start the web interface in a second terminal (port 3000)
```bash
cd frontend
npm run dev
```

**3.** Open `http://localhost:3000` in a browser.

In the virtual environment, a single command starts the backend, the web interface and the browser (stop with Ctrl+C).
```bash
python run_virtual.py        # virtual sample seed 0
python run_virtual.py 7      # virtual sample seed 7
```
This always runs in the virtual environment, regardless of `RAMAN_VIRTUAL_HW` in `.env`. If port 8000 or 3000 is already in use, it exits without starting, so stop any running RamanCopilot server first.

---

## Benchmark
There are two benchmarks: a benchmark for quantitative performance evaluation, and a virtual-environment benchmark that runs without the instrument.

### Quantitative performance benchmark
The benchmark consists of 140 items across seven capability axes (single operation, compound operation, preprocessing and peak analysis, acquisition and analysis, safety and exception handling, diagnosis and remediation, library similarity matching).

- [benchmark/raman_benchmark.xlsx](benchmark/raman_benchmark.xlsx): item list with questions, grading modes, grading criteria and setup
- [benchmark/ans/](benchmark/ans/): answer file for each item
- [benchmark/input/](benchmark/input/): input files used by the items
- [benchmark/items.json](benchmark/items.json): merged item definitions used by the runner

```bash
python -m benchmark.test_pipeline.run                  # all items
python -m benchmark.test_pipeline.run --models gemma4:31b,anthropic:claude-sonnet-4-6
```

### Virtual-environment benchmark
RamanCopilot is evaluated on virtual hardware (stage, laser, CCD, camera) and a virtual sample generated from a seed, without the instrument. The sample consists of polystyrene and PMMA particles on a Si substrate; every laser shot accumulates dose and damages the sample irreversibly (the dose is superlinear in laser power). The benchmark evaluates whether the agent finds the damage threshold of the polystyrene particles and reports the highest safe laser power and exposure time. The same task can be run under three memory settings for comparison.

| Scenario | Description |
|---|---|
| `off` | memory off |
| `iso` | memory on; each session starts from an empty memory |
| `transfer` | memory on; records and skills of earlier sessions carry over to later sessions |

Each episode generates the environment from the same seed, so the scenarios are compared under the same conditions.

- [virtual_benchmark/prompts.json](virtual_benchmark/prompts.json): task prompts (one prompt = one session)
- `virtual_benchmark/runs/<stamp>_<model>/sessions/<scenario>/<session>.json`: per-session results (final conditions, timing, memory processing time)

Run the virtual-environment benchmark with `RAMAN_VIRTUAL_HW=1` in `.env`. Without it, the benchmark tries to read the instrument's `Config.ini` and does not run on a PC without a Mantaray.
```bash
python -m virtual_benchmark.auto                                  # 3 scenarios × 1 episode
python -m virtual_benchmark.auto --scenarios off,iso,transfer --episodes 3
python -m virtual_benchmark.auto --episodes 3 --seeds 0,7,42      # sample seed per episode
python -m virtual_benchmark.auto --model anthropic:claude-sonnet-4-6
```

All sample and substrate properties are set in a single file, [virtual_benchmark/config.json](virtual_benchmark/config.json). The component keys are `bg` = Si substrate, `tgt2` = PMMA, `tgt` = polystyrene.

| Key | Setting |
|---|---|
| `best_laser_pow`, `best_laser_exp_s` | damage threshold conditions (laser power in mW, exposure time in s); values above 45.7 mW (the laser maximum) are clipped to 45.7 mW |
| `target_peak_cm1` | polystyrene peak used for evaluation (1001.4 cm⁻¹; must be in `bands.tgt`) |
| `all_bands_FWHM` | minimum FWHM of all bands (cm⁻¹) |
| `target_area` | area fraction of polystyrene (default 0.12) |
| `bands` | Raman bands of each component; each entry is `[center cm⁻¹, relative height, FWHM]` |
| `scale` | signal strength of each component |
| `colors_rgb` | camera color of each component (RGB 0-255) |
| `spot_w_um` | laser spot size (µm) |
| `damage` | damage model: `alpha` = power exponent, `w_decay` = decay width above the threshold |
| `ccd` | counts of the brightest peak (at the threshold conditions), full well, read noise |
| `world` | sample map size (pixels), particle distribution shape (`fbm_base`, `fbm_octaves`, `gamma`), reference magnification for the map scale (`ref_objective`, default 50x) |

The particle layout is chosen at run time with `RAMAN_VBENCH_SEED` or `--seeds`.

---

## License
This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact
If you encounter any issues or have questions, please open an issue on the GitHub repository.

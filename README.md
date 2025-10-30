# vLLM Apptainer SLURM

## Overview
This repository contains scripts and source code to build and run vLLM inside an Apptainer container on an HPC system with SLURM.

- **jobs/**: SLURM job scripts to build the container and run inference.
- **src/**: Python source code for running vLLM inference.

This repository is the codebase for the tutorial here: https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/232851290/LLM+and+VLM+inference+on+Snellius+with+vLLM

## Repository Structure

```
vllm-inference-slurm/
├── README.md
├── .gitignore
├── jobs/
│   ├── build_vllm.job
│   └── run_vllm_serve.job
├── src/
│   └── vllm_serve.py
```



## Usage

### 1. Build the container (SLURM)
Skip this step if you the prebuilt container on Snellius is sufficient. Refer to [here](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/232851290/LLM+and+VLM+inference+on+Snellius+with+vLLM)

#### Build your own container
```bash
sbatch jobs/build_vllm.job
```

### 2. Run vLLM inference

#### Via queue
```bash
sbatch jobs/run_vllm_serve.job
```


#### Interactively
```bash
# salloc a GPU...
chmod +x jobs/run_vllm_serve.job
./jobs/run_vllm_serve.job
```

#### Options
Please specify in `jobs/run_vllm_serve.job` the environment variables corresponding to the vLLM task. Below is a machine translation task with the Tower-Plus architecture and the template provided in `src/vllm_serve.py`
```
MODEL_CHECKPOINT=Unbabel/Tower-Plus-2B
DATASET=openai/gsm8k
TEMPLATE_PRESET=gsm8k
DATA_SPLIT=test[:100]
VLLM_BASE_URL=http://localhost:$PORT/v1
TEMPERATURE=0.7
MAX_TOKENS=256
MAX_CONCURRENT=64
OUTPUT_JSON=predictions.json
```


This corresponds to `src/vllm_serve.py`
```
options:
  -h, --help            show this help message and exit
  --model MODEL         Model name
  --dataset DATASET     HuggingFace dataset name
  --split SPLIT         Dataset split
  --subset SUBSET       Dataset subset
  --base_url BASE_URL   vLLM server URL
  --temperature TEMPERATURE
                        Sampling temperature
  --max_tokens MAX_TOKENS
                        Max tokens to generate
  --max_concurrent MAX_CONCURRENT
                        Max concurrent requests
  --instruction_template INSTRUCTION_TEMPLATE
                        Instruction template with {field_name} placeholders (e.g., 'Solve: {question}')
  --template_preset {gsm8k,alpaca,squad,mmlu,tower,default}
                        Use a preset template
  --output OUTPUT       Output file
```
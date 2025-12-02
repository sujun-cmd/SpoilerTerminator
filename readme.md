


# SpoilerTerminator

Automatic Spoiler Detection and Spoiler-Free Rewriting via Embedding-Based Weak Supervision

Course Project for STATS 507, University of Michigan

## Overview

SpoilerTerminator is an end-to-end data science pipeline designed to detect and neutralize spoilers in movie and book reviews.

The core challenge in spoiler detection is Label Noise (Weak Supervision): existing datasets often provide document-level labels (e.g., "This review contains a spoiler"), but individual sentences within the review might be safe (e.g., "The acting was great").

This project introduces a novel Teacher-Student Framework to solve this:

- **Vector Teacher**: A Logistic Regression model trained on MPNet embeddings identifies the semantic "direction" of spoilers (e.g., death, plot twists) to clean noisy data.
- **RoBERTa Student**: A fine-tuned RoBERTa model trained on high-quality data via Self-Training.
- **Generative Despoiler**: A quantized Qwen-2.5-14B LLM that rewrites detected spoilers into suspenseful teaser-style sentences.

## Pipeline Architecture

```mermaid
graph LR
    A[Raw Noisy Data] -->|Sentence Embedding| B[MPNet-Base]
    B --> C[LR Teacher Model]
    C -->|Filter: Score > 0.75 or < 0.4| D[Cleaned Data]
    D -->|Fine-tuning| E[RoBERTa Student]
    E -->|Detection| F{Is Spoiler?}
    F -->|Yes| G[Qwen LLM Despoiler]
    F -->|No| H[Keep Original]
    G --> I[Safe Teaser]

````

## Repository Structure

```
SpoilerTerminator/
├── train_vector.py          # Step 1: Train the LR Teacher using MPNet embeddings
├── eval_vector.py           # Step 1.5: Evaluate the Teacher's direction finding
├── run_self_train.py        # Step 2: Teacher cleans data -> Trains RoBERTa Student
├── compare_models_kaggle.py # Step 3: Cross-Domain Evaluation (IMDB -> Goodreads)
├── despoiler.py             # LLM-based rewriter (Qwen-14B-Int4)
├── main.py                  # End-to-end inference pipeline
├── reviews.txt              # Sample input file
└── *.slurm                  # Slurm submission scripts
```

## Installation & Requirements

This project is optimized for Linux servers with NVIDIA GPUs (tested on V100 16GB/32GB).

### Clone the repository

```bash
git clone https://github.com/sujun-cmd/SpoilerTerminator.git
cd SpoilerTerminator
```

### Environment Setup

Recommended Python version: 3.10+

```bash
python -m venv hf_env
source hf_env/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets sentence-transformers
pip install scikit-learn accelerate bitsandbytes
pip install nltk
```

Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Usage

### 1. Train the Vector Teacher

```bash
python train_vector.py
```

### 2. Self-Training (Clean & Fine-tune)

```bash
python run_self_train.py
```

### 3. Cross-Domain Evaluation (Optional)

```bash
python compare_models_kaggle.py
```

### 4. Run the Full Pipeline (Detect & Rewrite)

```bash
python main.py
```

This processes `reviews.txt` and generates `despoiled_reviews.txt`.

## Experimental Results

### 1. Robustness against Label Noise (IMDB)

Our Teacher-Student approach significantly outperformed baselines and corrected mislabeled data.

| Model            | F1 Score | Recall | Precision | Note                |
| ---------------- | -------- | ------ | --------- | ------------------- |
| Baseline (Naive) | 0.26     | Low    | Low       | Overfitted to noise |
| RoBERTa (Ours)   | 0.49     | 0.69   | 0.40      | Corrects noise      |

Example correction:

* **Input**: "Why didn't Obi-Wan kill Anakin?"
* **Original Label**: Safe (Incorrect)
* **Our Model**: Spoiler 1.0 (Correct)

### 2. Cross-Domain Generalization (Books)

* **LR Teacher**: Recall 0.84
* **RoBERTa Student**: Accuracy 0.66

### 3. Generative Rewriting (Qwen-2.5-14B)

| Original Spoiler                               | Rewritten Version                                               |
| ---------------------------------------------- | --------------------------------------------------------------- |
| "Bruce Willis is actually a ghost in the end." | "There is a major revelation about Bruce Willis's true nature." |
| "She dies in the car accident."                | "A severe car accident leaves one character's fate uncertain."  |

## Citation

```
@article{Su2025SpoilerTerminator,
  title={SpoilerTerminator: Automatic Spoiler Detection and Rewriting via Weak Supervision},
  author={Jun Su},
  journal={STATS 507 Final Project, University of Michigan},
  year={2025}
}
```

## Contact

Jun Su — [sujun@umich.edu](mailto:sujun@umich.edu)

```

```


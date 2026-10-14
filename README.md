# Fine-Tuning Llama-3.1-Nemotron-Nano-4B-v1.1 on SAMSum with QLoRA

Fine-tunes **Llama-3.1-Nemotron-Nano-4B-v1.1** on the **SAMSum dialogue-summarization dataset** using **QLoRA**. It compares the original zero-shot model against the fine-tuned model using ROUGE and BERTScore.

## Objective

Given a multi-speaker chat dialogue, generate a short, factual summary in the SAMSum style.

Example:

**Input dialogue**

```text
Amy: Are we still on for dinner tonight?
Ben: Yes! 7pm at the Italian place?
Amy: Perfect. Can you book a table? I'll be late from work.
Ben: Sure, I'll book for 7:30 instead.
Amy: Thanks, see you there!
```

**Expected summary**

```text
Ben will book a table for 7:30 pm at the Italian place.
```

## Approach

The notebook follows this workflow:

1. Download and preprocess the SAMSum dataset from Kaggle.
2. Use an 8,000-example training subset and a 100-example validation subset.
3. Format each example as a chat prompt.
4. Load the Nemotron model in 4-bit precision.
5. Attach LoRA adapters and train with QLoRA.
6. Generate summaries for the SAMSum test set.
7. Compare zero-shot and fine-tuned performance.
8. Save the adapter, predictions, metrics, and plots.

## Model and Training Setup

| Item | Value |
|---|---|
| Base model | `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` |
| Dataset | SAMSum |
| Training subset | 8,000 examples |
| Validation subset | 100 examples |
| Test set | Full SAMSum test set, 819 examples |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | All linear layers |
| Effective batch size | 16 |
| Gradient accumulation | 16 |
| Learning rate | 2e-4 |
| Scheduler | Cosine decay with 3% warmup |
| Epochs | 1 |
| Maximum sequence length | 512 tokens |
| Optimizer | Paged AdamW 8-bit |
| Random seed | 42 |

## How the Model Predicts

This is a causal language model, not a classifier. It does not predict a single label. Instead, it predicts the next token repeatedly to generate a complete summary.

For each dialogue, the model receives:

```text
System: detailed thinking off
User: Summarize the following dialogue in one or two sentences.

<dialogue>
```

It then predicts one token at a time:

```text
Ben -> will -> book -> a -> table -> ...
```

Generation stops when the model emits an end-of-text token or reaches the maximum generation length of 100 new tokens.

## Baseline vs. Fine-Tuned Model

| Model | Adapter | Description |
|---|---|---|
| Zero-shot baseline | Disabled | Uses the original pretrained Nemotron model |
| Fine-tuned model | Enabled | Uses the same base model plus the SAMSum-trained QLoRA adapter |

Both models use the same prompt and generation settings. The only difference is whether the trained LoRA adapter is active.

## Training Objective

During training, the model sees:

- Prompt tokens: instruction and dialogue
- Summary tokens: human-written SAMSum reference summary

Prompt tokens are masked with `-100`, so they provide context but are not used in the loss. The model is trained only to predict the reference summary tokens.

The training loss is cross-entropy loss:

```text
loss = -log P(correct next summary token | dialogue and previous summary tokens)
```

By minimizing this loss, the model learns to assign higher probability to concise, SAMSum-style summaries.

## Results

| Metric | Zero-shot base | Fine-tuned QLoRA | Improvement |
|---|---:|---:|---:|
| ROUGE-1 | 29.42 | 51.38 | +21.96 |
| ROUGE-2 | 7.66 | 26.96 | +19.30 |
| ROUGE-L | 21.18 | 43.04 | +21.86 |
| BERTScore-F1 | 88.04 | 92.57 | +4.53 |
| Average words | 46.73 | 17.26 | -29.47 |

The fine-tuned model produces much shorter summaries that are closer to the human reference summaries. It also achieves substantially higher ROUGE and BERTScore scores.

## Example Comparison

**Dialogue**

```text
Pete: Did you clean the hamster cage?
Adelle: No. Is it my turn?
Pete: Yes. After school, no excuses.
Adelle: Fine.
```

**Reference**

```text
Adelle has to clean the hamster cage after school.
```

**Zero-shot baseline**

```text
The dialogue between Pete and Adelle involves a negotiation about pet responsibility...
```

**Fine-tuned model**

```text
Adelle didn't clean the hamster cage. Pete will clean it after school.
```

The fine-tuned output is shorter and closer to the reference style, although it incorrectly assigns the cleaning task to Pete rather than Adelle.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| ROUGE-1 | Measures overlap of individual words |
| ROUGE-2 | Measures overlap of two-word sequences |
| ROUGE-L | Measures longest matching word sequence |
| BERTScore-F1 | Measures semantic similarity using embeddings |
| Average words | Checks whether generated summaries match reference length |

ROUGE compares generated summaries with human reference summaries. BERTScore provides an additional semantic-quality check.

## Visualizations

The notebook produces:

1. Training and validation loss curves.
2. Learning-rate schedule.
3. Zero-shot versus fine-tuned metric comparison.
4. Summary-length distributions.
5. Per-sample ROUGE-L distributions.
6. Base-versus-fine-tuned ROUGE-L scatter plot.
7. Win-rate calculation showing how often the fine-tuned model beats the baseline.

## Files Saved

The notebook saves outputs to Google Drive when `USE_DRIVE = True`:

```text
/content/drive/MyDrive/nemotron_samsum/
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── ckpt/
│   └── training checkpoints
├── preds_base.json
├── preds_ft.json
├── log_history.json
├── results.csv
└── predictions.csv
```

## Requirements

Install the dependencies with:

```bash
pip install -U transformers peft accelerate bitsandbytes datasets evaluate rouge_score bert_score kagglehub sentencepiece ipywidgets
```

A GPU is required. The notebook was designed for Google Colab with a T4, L4, or A100 GPU.

## Kaggle Authentication

Provide Kaggle credentials using either method:

1. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets and enable Notebook access.
2. Run the notebook and enter the credentials when prompted.

## How to Run

1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Run the cells from top to bottom.
4. Wait for dataset download, training, generation, and evaluation.
5. Inspect the metrics, plots, and side-by-side examples.

If a trained adapter already exists in the output directory, the notebook loads it and skips training. Set `FORCE_RETRAIN = True` to retrain from scratch.

## Key Takeaway

Fine-tuning taught the model the SAMSum summarization style: short, factual, dialogue-grounded summaries. The zero-shot baseline could write fluent text, but the fine-tuned model produced summaries that were much closer to the human references in length, wording, structure, and meaning.

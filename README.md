# Generalizable multilingual medical de-identification model by generative instruction tuning
> Code for *"Generalizable multilingual medical de-identification model by generative instruction tuning"*

## Quickstart

The pipeline consists of two parts:
1. Synthetic Data Generation `gen_synthetic_data.py` - Use LLMs to replicate the style of your sensitive data.
We've included prompts for UK NHS patient requests (`/prompts`) but you could adapt this for other sources of sensitive data.
This will generate pairs of raw text and the same text but with PII replaced with `[identifiers]`.
2. Finetune LLM `axolotl train model_config.yaml` - Uses the synthetic data to train a LLM to perform anonymization.

```bash
pip install -r requirements.txt
python gen_synthetic_data.py
axolotl train model_config.yaml
```

## Adapting to New Domains

To adapt this for your own problem domain you'll need to change a few things:

1. Change the `seed_text.json` to match the structure, style and content of your real data
2. Change `prompts/000.txt` to give instructions for your domain
3. In `gen_synthetic_data.py`, modify the `ALLOWED_TAGS` with your list of PII types, and modify the system prompt for your domain.

Once you've run `gen_synthetic_data.py` and created some initial data, compare it to your real data. If needed write `prompts/001.txt` etc.

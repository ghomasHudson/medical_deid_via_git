'''Generates synthetic data for anonymization'''

from litellm import completion
from tqdm import tqdm
import json
import random
import re

ALLOWED_TAGS = ["name", "tel", "dob", "pharmacy", "nhs_num", "practice", "address", "email", "social"]
PROMPT_FN = "prompts/000.txt"

tags_str = " ".join([f"[{tag}]" for tag in ALLOWED_TAGS])

def is_good_text(anonymized_text):
    '''Check that the text only has pii anonymized from our list'''
    tags = re.findall(r"\[(.*?)\]", anonymized_text)
    tags = [t for t in tags if t not in ALLOWED_TAGS]
    return len(tags) == 0

seeds = json.load(open("seed_text.json"))

# Load existing text as examples (for subsequent prompting rounds)
texts = []
with open("synthetic_anonymization_data.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        texts.append(entry["messages"][0]["content"])

# Make new synthetic data
with open("synthetic_anonymization_data.jsonl", "a") as f:
    for seed in tqdm(seeds):
        example_text = random.choice(texts)

        prompt = open(PROMPT_FN).read().format(example_text=example_text)

        response = completion(
          model="openai/gpt-4o",
          response_format={ "type": "json_object" },
          messages=[
            { "role": "system", "content": "You are a data quality expert who is working on training a large language model to anonymize medical text"},
            { "role": "user", "content": prompt + """
Generate lists of pairs in a

{"pairs": [{
    "plain": \"""" + seed["plain"] + """\",
    "anonymized": \"""" + seed["anonymized"] + """\",
}]} JSON format.

Change the disease/problem and grammatical structure. Anonymize """ + tags_str}])

        output = json.loads(response.choices[0].message.content)
        for pair in output["pairs"]:
            if is_good_text(pair["anonymized"]):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": pair["plain"]},
                        {"role": "assistant", "content": pair["anonymized"]}
                    ]
                }) + "\n")
                f.flush()

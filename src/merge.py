from src.models.model import VIGIA
from transformers import AutoTokenizer
import os

import argparse
parser = argparse.ArgumentParser(description="Merge LoRA weights into the VIGIA model.")
parser.add_argument("--path", type=str, required=True, help="Path to the VIGIA model directory.")
args = parser.parse_args()
PATH = args.path

model = VIGIA.from_pretrained(PATH)
model.model.merge_lm_lora()
tokenizer = AutoTokenizer.from_pretrained(PATH)

model.save_pretrained(os.path.join(PATH, "merged"))
tokenizer.save_pretrained(os.path.join(PATH, "merged"))
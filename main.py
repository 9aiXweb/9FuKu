import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b")
#hello
if torch.cuda.is_available():
    model = model.to("cuda")

text = "西田幾多郎は、"
token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=100,
        min_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output = tokenizer.decode(output_ids.tolist()[0])
print(output)
"""西田幾多郎は、この「絶対矛盾的自己同一」を「世界の自己同一」と置きかえ、さらに西田哲学を出発点として「絶対無」を「世界の成立」に変え、世界と自己を一つの統一物とみなす哲学として展開する。この世界と自己は絶対矛盾的自己同一として同一の性質を有し、同じ働きをする。西田哲学においては、この世界と自己は矛盾しあうのではなく、同一の性質をもっている。この世界と自己は同一である。絶対"""

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
from tqdm import tqdm
import torch

max_length = 1024
stride = 512

models_name = [
    'allenai/OLMo-7B',
    'allenai/OLMo-1B',
    'openai-community/gpt2',
    'openai-community/gpt2-medium',
    'openai-community/gpt2-large',
    'openai-community/gpt2-xl',
]

full_dataset_df = pd.read_csv(
    f"Wiki_no_title_reduced_DreamBank_reduced_en.csv"
)

Wiki_Text = full_dataset_df[full_dataset_df["Data"].isin(["WikiText2"])]["Text"].tolist()
DrBk_Text = full_dataset_df[full_dataset_df["Data"].isin(["DreamBank"])]["Text"].tolist()

wiki_all = "\n\n".join([t for t in Wiki_Text])
dbnk_all = "\n\n".join([t for t in DrBk_Text])

device = "cuda"
model_id = "gpt2"

results = []

for mdl_id in models_name:

    if 'gpt' in mdl_id:
        model = GPT2LMHeadModel.from_pretrained(mdl_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(mdl_id)
    elif 'gemma' in mdl_id or 'llama' in mdl_id:
        tokenizer = AutoTokenizer.from_pretrained(mdl_id)
        model = AutoModelForCausalLM.from_pretrained(mdl_id).to(device)
    else:
        print(f'Mode {mdl_id} not recognised')
        break

    model_name = mdl_id.split('/')[1]
    for testing_data, testing_data_NAME in [[dbnk_all, "DreamBank"], [wiki_all, "WikiText2"]]:

        print(f"Data: {testing_data_NAME}; Model: {model_name}")
        seq_len = len(testing_data.split())

        nlls = []
        prev_end_loc = 0

        for begin_loc in tqdm(range(0, seq_len, stride)):

            end_loc = min(begin_loc + max_length, seq_len)
            text = " ".join(testing_data.split()[begin_loc:end_loc])
            trg_len = end_loc - prev_end_loc 

            encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = encodings.input_ids[:,].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

        results.append([testing_data_NAME, model_name, ppl.item(), seq_len])

        results_df = pd.DataFrame(results, columns=['Data', 'Model', 'Perplexity', 'Length'])
        results_df.to_csv('full_datasets_ppl.csv', index=False)
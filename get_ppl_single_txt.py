import pandas as pd
from ppl_eval import compute_ppl
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = 'cuda'

models_name = [
    'openai-community/gpt2',
    'openai-community/gpt2-medium',
    'openai-community/gpt2-large',
    'openai-community/gpt2-xl',
    'allenai/OLMo-1B',
    'allenai/OLMo-7B',
]

full_dataset_df = pd.read_csv(
    "Wiki_no_title_reduced_DreamBank_reduced_en.csv"
)

data_as_list = full_dataset_df["Text"].tolist()

for mdl_id in models_name:
    model_name = mdl_id.split('/')[1]
    print(f"Model: {model_name}")

    if 'gpt' in mdl_id:
        model = GPT2LMHeadModel.from_pretrained(mdl_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(mdl_id)

    elif 'OLMo' in mdl_id:
        model = AutoModelForCausalLM.from_pretrained(
            mdl_id,
            trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(mdl_id)

    else:
        print(f" Model {model_name} not recognised")
        break

    p = compute_ppl(
        predictions=data_as_list,
        model=model,
        tokenizer=tokenizer,
        add_start_token=False,
    )

    full_dataset_df[f"{model_name}_perplexities"] = list(p["perplexities"])

    full_dataset_df.to_csv(
        "Wiki_no_title_reduced_DreamBank_reduced_en_pptx_multiGPT.csv",
        index=False
    )

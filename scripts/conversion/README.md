# Conversion scripts

This readme explains how to perform megatron to huggingface conversions.
To do this conversion, the first thing to ask is if you have been saving megatron checkpoints using `torch_dist` format.
This is specified by the `--ckpt-format=torch_dist`, and is the default value.
If you have been using the swiss-ai templates, you most likely are.
In this case, first you will need to save your checkpoint in the `torch` format by running the `torchdist_2_torch.py`:
```
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun scripts/conversion/torchdist_2_torch.py \
	--bf16 \
	--load CHECKPOINT_PATH \
	--ckpt-convert-save INTERMEDIATE_CHECKPOINT_PATH \
	--ckpt-step ITERATION_STEP  # Optional, if not specified you will load the latest checkpoint available.
```
If you get a `ModuleNotFoundError: No module named 'megatron'`, try setting `PYTHONPATH=$PWD`.
For converting larger models (e.g. 70B) you will need to use pass `--nproc-per-node=4` to `torchrun` and `--pipeline-model-parallel-size=4` to the `torchdist_2_torch.py` script.
Note that this decision will not change how you call the `convert.py` next (i.e. don't set `--pipeline-model-parallel-size` with the `convert.py` script).

Keep in mind that the `CHECKPOINT_PATH` is the root directory that stores all of the checkpoints, and usually its contents like this: `iter_0001000/ iter_0002000/ latest_checkpointed_iteration.txt progress.txt`.
The `INTERMEDIATE_CHECKPOINT_PATH` will be needed for the following step, after which can be safely removed.

The next step is to perform the actual huggingface conversions.
In the next snippet, replace `CHECKPOINT_PATH` with `INTERMEDIATE_CHECKPOINT_PATH` if you ran the previous step:
```
python tools/checkpoint/convert.py \
	--model-type GPT \
	--loader core \
	--saver llama_hf \
	--load-dir CHECKPOINT_PATH \
	--save-dir SAVE_DIR \
	--hf-tokenizer HF_TOKENIZER_NAME  # Optional, set it to save the tokenizer config in `SAVE_DIR`.
```
Set `--test-logits` if you want to make sure logits of the converted model match with the megatron implementation (only possible with TP1,PP1).
In order to be able to convert Apertus models, we instantiate a custom HF `SwissAIForCausalLM`; make sure to install the latest version of the transformers fork before running the conversion:
```
git clone https://github.com/swiss-ai/transformers.git
cd transformers
pip install -e .
```

Your Huggingface converted model will be in `SAVE_DIR` and you can use it normally:
```
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, device_map="auto", torch_dtype="auto")

prompt = "What's the best way to get in shape?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=256,do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

See `scripts/conversion/do-convert.sh` for an end-to-end megatron->hf conversion example.

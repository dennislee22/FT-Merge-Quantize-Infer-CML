import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

torch.cuda.empty_cache()

model_name = "tiiuae/falcon-7b"
adapters_name = "dlee-falcon-7b-fine-tuned/checkpoint-439"

print(f"Starting to load the model {model_name} into memory")

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
m = PeftModel.from_pretrained(m, adapters_name)
m = m.merge_and_unload()
tok = AutoTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1

stop_token_ids = [0]

print(f"Successfully loaded the model {model_name} into memory")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

prompt = "Today was an amazing day because"
inputs = tok(prompt, return_tensors="pt").to(device)

outputs = m.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)

tok.batch_decode(outputs, skip_special_tokens=True)

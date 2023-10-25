import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

torch.cuda.empty_cache()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = "dlee-falcon-7b-fine-tuned3"
base_model = "tiiuae/falcon-7b"
adapters_name = "dlee-falcon-7b-fine-tuned2/checkpoint-4653"

print(f"Load the model {base_model} into " + device + " memory")

m = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
m = PeftModel.from_pretrained(m, adapters_name)
m = m.merge_and_unload()
m.save_pretrained(output_dir) 
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(output_dir) 

print(f"Successfully merged and loaded the model into " + device + " memory")

print(f"Test the new model")
prompt = "Generate a Python program that adds two then doubles the result."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = m.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)

tokenizer.batch_decode(outputs, skip_special_tokens=True)

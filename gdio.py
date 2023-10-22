import gradio as gr
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.cuda.is_available()

chatbox = gr.inputs.Textbox(lines=2, placeholder="Type a message...")
output = gr.outputs.Textbox()

#model_paths = {
#    "falcon-7b": "falcon-7b",
#    "dlee-falcon-7b-fine-tuned": "dlee-falcon-7b-fine-tuned",
#}

model_dropdown = gr.Dropdown(label="Model", value="model1", choices=["falcon-7b", "dlee-falcon-7b-fine-tuned"])
#model_dropdown = gr.Dropdown(choices=list(model_paths.keys()), label="Select LLM Model")



#def load_data():    
# tokenizer = AutoTokenizer.from_pretrained("falcon-7b",torch_dtype=torch.bfloat16,device_map="auto")
# model = AutoModelForCausalLM.from_pretrained("falcon-7b",torch_dtype=torch.bfloat16,device_map="auto")
# return tokenizer, model
#tokenizer, model = load_data()

#title="Chatbox using Fine-tuned Falcon 7B Model with Custom Dataset",
# Function to generate responses using the LLM model
def generate_response(input_text, selected_model):
    tokenizer = AutoTokenizer.from_pretrained(model_dropdown,torch_dtype=torch.bfloat16,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_dropdown,torch_dtype=torch.bfloat16,device_map="auto")
    return tokenizer, model
    tokenizer, model = load_data()
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    response = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    return response_text

iface = gr.Interface(fn=generate_response, inputs=[chatbox, model_dropdown], outputs=output, title="Dropdown Example")



iface.launch(server_name="127.0.0.1", server_port=8090)

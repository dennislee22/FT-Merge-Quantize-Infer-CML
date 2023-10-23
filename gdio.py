import gradio as gr
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.cuda.is_available()

def unload_model():
    model = tokenizer = None
    torch.cuda.empty_cache()
    
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("falcon-7b",torch_dtype=torch.bfloat16,device_map="auto")
    model = AutoModelForCausalLM.from_pretrained("falcon-7b",torch_dtype=torch.bfloat16,device_map="auto")
    return tokenizer, model
    tokenizer, model = load_data()
    
def reload_model():
    unload_model()
    tokenizer, model = load_model()
    
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    response = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    return response_text

def create_ui():
    with gr.Tab("tab1"):
        with gr.Row():
            with gr.Column():
                model_selected = gr.Dropdown(choices=["falcon-7b", "dlee-falcon-7b-fine-tuned"], label='Choose a GenAI Model')        
                reload_button = gr.Button("Reload Model")
                reload_button.click(unload_model).then(reload_model)
                with gr.Row():
                    with gr.Column():
                           gr.Interface(
                           fn=generate_response,
                           inputs="text",
                           outputs="text",
                           )
                            
with gr.Blocks(title="asd", theme=gr.themes.Default()) as demo:
    create_ui()

demo.launch(share=True)
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

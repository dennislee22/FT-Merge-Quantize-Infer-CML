import gradio as gr
import torch
import transformers
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize a global variable to store the Gradio interface
loaded_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_model = None

def flush_gpu_memory():
    tokenizer = model = None
    torch.cuda.empty_cache()

def load_model(model_name):
    global selected_model  # Access the selected_model variable
    selected_model = model_name
    if selected_model == None:
        yield f"No model selected"
        print("No model selected")
        return
    else:   
        if selected_model != []:
                yield f"Selected model is `{selected_model}`"
                time.sleep(2)
                yield f"Loading `{selected_model}`... into {device}"
                tokenizer = AutoTokenizer.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto")
                model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map="auto")
                loaded_models[model_name] = {"tokenizer": tokenizer, "model": model}
                yield f"Successfully loaded `{selected_model}` into {device}."
        else:
            yield f"Failed to load model `{selected_model}`. Please select the model and press Reload Model button "

def generate_response(input_text, _):
    return _generate_response(input_text)

def generate_response(input_text):
        if selected_model == None:
            return "Model not loaded. Select a model in the dropdown menu and click the 'Reload Model' button to load a model."
        if not input_text:
            return "Please enter some text in the input field before submitting."
        tokenizer = loaded_models[selected_model]["tokenizer"]
        model = loaded_models[selected_model]["model"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        response = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        return response_text

def create_ui():
    gr.HTML("<h1>Chatbox using Fine-tuned Falcon 7B Model with Custom Dataset</h1>")
    with gr.Tab("AI Text Generator"):
        with gr.Row():
            with gr.Column():
                model_selected = gr.Dropdown(choices=["falcon-7b", "dlee-falcon-7b-fine-tuned"], label='Select a GenAI Model')        
                reload_button = gr.Button("Reload Model")
                status_message = gr.Label(label="Model Status")
                inp = model_selected
                reload_button.click(flush_gpu_memory).then(load_model, inputs=inp, outputs=status_message)
                
                with gr.Row():
                    with gr.Column():
                        global iface  
                        iface = gr.Interface(
                        fn=generate_response,
                        inputs="text",  
                        outputs="text",
                        allow_flagging="never",
                        title="LLM Chatbox",
                        description="Enter a message to chat with the loaded model.",    
                        )
theme = gr.themes.Soft()
theme.css = """
    .output-text {
        height: 300px;  /* Set the height to your desired size in pixels */
        overflow-y: scroll;  /* Add a scrollbar if the content overflows the box */
    }
"""
flush_gpu_memory()
with gr.Blocks(title="My Chatbox", theme=theme) as demo:
    create_ui()
demo.queue()
demo.launch(server_name="127.0.0.1", server_port=8090)

import gradio as gr
import torch
import transformers
import time
import subprocess
import datetime
import pytz

from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize a global variable to store the Gradio interface
loaded_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_model = None
model1 = "codegen-350M-multi"
model2 = "merged_ft_model"

def flush_gpu_memory():
    torch.cuda.empty_cache()
    tokenizer = model = None 

def load_model(model_name):
    flush_gpu_memory()
    global selected_model
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
            yield f"Failed to load model `{selected_model}`. Please select a model and press Reload Model button."

def generate_response(input_text, _):
    return _generate_response(input_text) #, which is a SQL statement that can solve the Task.

def generate_response(input_text):
        prompt = f"""
        # Instruction:
        Use the context below to produce the result
        # context:
        {input_text}
        # result:
        """
        if selected_model == None or selected_model == []:
            return "Model not loaded. Select a model in the dropdown menu and click the 'Reload Model' button to load a model."
        if not input_text:
            return "Please enter some text in the input field before submitting."
        tokenizer = loaded_models[selected_model]["tokenizer"]
        model = loaded_models[selected_model]["model"]
        #input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
        #response = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2,attention_mask=attention_mask)
        #response_text = tokenizer.decode(response[0], skip_special_tokens=True).replace(input_text, "").strip() #.replace removes the input text from the generated output
        response = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_p=0.9,temperature=0.5,attention_mask=attention_mask)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True).replace(prompt, "").strip() #.replace removes the input text from the generated output
        return response_text
    
def run_os_command_nvidia_smi():
    current_time = datetime.datetime.now(pytz.timezone('Asia/Singapore')).strftime("%Y-%m-%d %H:%M:%S %Z")  
    command = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    output_lines = process.stdout.read().splitlines()

    # Construct a table in HTML
    table_html = "<table>"

    # Add headers
    table_html += "<tr><th>GPU</th><th>Memory Used</th><th>Memory Total</th><th>Memory Used %</th></tr>"

    for line in output_lines:
        columns = line.split(',')
        if len(columns) != 0:
            gpu_index, memory_used, memory_total = columns[:3]
            gpu_index = f"GPU {gpu_index}"
            memory_used = f"{memory_used} MiB"
            memory_total = f"{memory_total} MiB"

            # Calculate memory used percentage
            memory_used_value = float(memory_used.split()[0])
            memory_total_value = float(memory_total.split()[0])
            memory_used_percentage = f"{(memory_used_value / memory_total_value * 100):.2f}%"
        else:
            gpu_index = memory_used = memory_total = memory_used_percentage = "N/A"

        table_html += f"<tr><td>{gpu_index}</td><td>{memory_used}</td><td>{memory_total}</td><td>{memory_used_percentage}</td></tr>"
    table_html += "</table>"
    result_html = f"<h4>GPU Status ({current_time}):</h4>"
    result_html += table_html
    return result_html

    
def create_ui():
    gr.HTML(f"<h1>TextAI using Fine-tuned '{model1}' Model with Custom Dataset</h1>")
    with gr.Tab("AI Text Generator"):
        with gr.Row():
            with gr.Column():
                model_selected = gr.Dropdown(choices=[model1,model2], label='Select a GenAI Model')        
                reload_button = gr.Button("Reload Model", variant="secondary")
                status_message = gr.Label(label="Model Status")
                inp = model_selected   
                reload_button.click(load_model, inputs=inp, outputs=status_message)
                with gr.Row():
                    with gr.Column():
                        gpuinfo = gr.HTML(lambda: run_os_command_nvidia_smi())
                        reload_button.click(run_os_command_nvidia_smi, outputs=gpuinfo)
                        gpu_button = gr.Button("Refresh GPU Status", variant="secondary")
                        gpu_button.click(run_os_command_nvidia_smi, outputs=gpuinfo)
                        with gr.Row():
                            with gr.Column():
                                global iface2  
                                iface2 = gr.Interface(
                                fn=generate_response,
                                inputs="text",  
                                outputs="text",
                                allow_flagging="never",
                                title="Test the Loaded Model:",
                                #description="Enter a message to chat with the loaded model.",
                                examples=[
                                ["CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?"],
                                ["CREATE TABLE trip (bus_stop VARCHAR, duration INTEGER). List all the bus stops from which a trip of duration below 100 started."],
                                ["def hello_world():"],
                                ],    
                                )          

mytheme = gr.themes.Soft().set(
    button_secondary_background_fill="#ade6d8",
    button_secondary_background_fill_hover="#AAAAAA",
    button_primary_background_fill="#2c9178",
    button_primary_background_fill_hover="#AAAAAA",
    button_shadow="*shadow_drop_lg",
)

with gr.Blocks(theme=mytheme) as demo:
    create_ui()

demo.launch(server_port=int(os.getenv('CDSW_APP_PORT')),
           enable_queue=True,
           show_error=True,
           server_name='127.0.0.1',
)
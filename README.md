LLM: Fine-Tune > Merge > Quantize > Infer .. on CML
===

<p align="center"><img src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/blob/main/images/peft.gif" width="600"></p>

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[1. Objective](#toc_0)<br>
[2. Benchmark Score & Summary](#toc_1)<br>
[3. Preparation](#toc_2)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1. Dataset & Model](#toc_3)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2. CML Session](#toc_4)<br>
[4. bigscience/bloom-1b1](#toc_5)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_6)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2. Quantize (GPTQ 8-bit) > Inference](#toc_7)<br>
[5. bigscience/bloom-7b1](#toc_8)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_9)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. Fine-Tune (4-bit) > Merge > Inference](#toc_10)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.3. Quantize (GPTQ 8-bit) > Inference](#toc_11)<br>
[6. tiiuae/falcon-7b](#toc_12)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_13)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. Fine-Tune (8-bit) > Merge > Inference](#toc_14)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.3. Quantize (GPTQ 8-bit) > Inference](#toc_15)<br>
[7. Salesforce/codegen2-1B](#toc_16)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_17)<br>
[8. Bonus: Use Custom Gradio for Inference](#toc_18)<br>

### <a name="toc_0"></a>1. Objective

1. To create a LLM that is capable of achieving an AI task with specific dataset, the traditional ML approach would need to train a model from the scratch. Study shows it would take nearly 300 years to train a GPT model using a single V100 GPU card. This excludes the iterative process to test, retrain and tune the model to achieve satisfactory results. This is where Parameter-Efficient Fine-tuning (PEFT) comes in handy. PEFT trains only a subset of the parameters with the defined dataset, thereby substantially decreasing the computational resources and time.
2. The provided iPython codes in this repository serve as a comprehensive illustration of the complete lifecycle for fine-tuning a particular Transformers-based model using specific datasets. This includes merging LLM with the trained adapters, quantization, and, ultimately, conducting inferences with the correct prompt. The outcomes of these experiments are detailed in the following section. The target use case of the experiments is making use the Text-to-SQL dataset to train the model, enabling the translation of plain English into SQL query statements.<br>
&nbsp;a. [ft-trl-train.ipynb](ft-trl-train.ipynb): Run the code cell-by-cell interactively to fine-tune the base model with local dataset using TRL (Transformer Reinforcement Learning) mechanism. Merge the trained adapters with the base model. Subsequently, perform model inference to validate the results.<br>
&nbsp;b. [quantize_model.ipynb](ft-trl-train.ipynb): Quantize the model (post-training) in 8, or even 2 bits using `auto-gptq` library.<br>
&nbsp;c. [infer_Qmodel.ipynb](ft-trl-train.ipynb): Run inference on the quantized model to validate the results.<br>
&nbsp;d. [gradio_infer.ipynb](gradio_infer.ipynb): You may use this custom Gradio interface to compare the inference results between the base and fine-tuned model.<br>
5. The experiments also showcase the post-quantization outcome. Quantization allows model to be loaded into VRAM with constrained capacity. `GPTQ` is a post-training method to transform the fine-tuned model into a smaller footprint. According to [🤗 leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), quantized model is able to infer without significant results degradation based on the scoring standards such as MMLU and HellaSwag. `BitsAndBytes` (zero-shot) helps further by applying 8-bit or even 4-bit quantization to model in the VRAM to facilitate model training. 
6. Experiments were carried out using `bloom`, `falcon` and `codegen2` models with 1B to 7B parameters. The idea is to find out the actual GPU memory consumption when carrying out specific task in the above PEFT fine-tuning lifecycle. Results are detailed in the following section. These results can also serve as the GPU buying guide to achieve a specific LLM use case.
 
#### <a name="toc_1"></a>2. Summary & Benchmark Score

- Graph below depicts the GPU memory utilization during a specific stage. This graph is computed based on the results obtained from the experiments as detailed in the tables below.

<img width="901" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/3a30ab71-29b3-49d8-b070-0189c43f64cc"><br>

- Tables below summarize the benchmark result when running the experiments using 1 unit of Nvidia A100-PCIE-40GB GPU on CML with Openshift (bare-metal):<br>

&nbsp;&nbsp;a. Time taken to fine-tune different LLM with 10% of Text-to-SQL dataset (File size=20.7 MB):<br>

| Model     | Fine-Tune Technique | Fine-Tune Duration | Inference Result     |
| :---      |     :---:           |   ---:             | :---                 |
| bloom-1b1  | No Quantization     | ~12 mins           | Good                |
| bloom-7b1  | No Quantization    | OOM                | N/A                  |
| bloom-7b1  | 4-bit BitsAndBytes  | ~83 mins          | Good                 |
| falcon-7b  | No Quantization    | OOM                | N/A                  |
| falcon-7b  | 8-bit BitsAndBytes  | ~65 mins          | Good                 |
| codegen2-1B  | No Quantization    | ~12 mins         | Bad                  |

OOM = Out-Of-Memory

&nbsp;&nbsp;b. Time taken to quantize the fine-tuned (merged with PEFT adapters) model using `auto-GPTQ` technique:<br>

| Model      | Quantization Technique| Quantization Duration | Inference Result  |
| :---       |     :---:           |   ---:                  | :---              |
| bloom-1b1  | auto-gptq 8-bit     | ~5 mins                 | Bad               |
| bloom-7b1  | auto-gptq 8-bit     | ~35 mins                | Good              |
| falcon-7b  | auto-gptq 8-bit     | ~22 mins                | Good              |

&nbsp;&nbsp;c. Table below shows the amount of memory of a A100-PCIE-40GB GPU utilised during specific experiment stage with different models.

| Model     | Fine-Tune Technique| Load (Before Fine-Tune) | During Training  | Inference Merged Model | During Quantization | Inference 8-bit GPTQ Model |
| :---      |     :---:          |   ---:          | :---             |     :---:              |   ---:              | ---:                        |                    
| bloom-1b1  | No Quantization    | ~4.5G           |~21G              | ~6G                   | ~6G                 | ~2G                         |
| bloom-7b1  | No Quantization    | ~27G           |OOM             | N/A                   | N/A                  | N/A                         |
| bloom-7b1 | 4-bit BitsAndBytes  | ~6G           |~17G              | ~31G                   | ~23G                 | ~9G                       |
| falcon-7b  | No Quantization    | ~28G           |OOM             | N/A                   | N/A                  | N/A                         |
| falcon-7b | 8-bit BitsAndBytes  | ~8G           |~16G              | ~28G                   | ~24G                 | ~8G                       |
| codegen2-1B  | No Quantization    | ~4.5G           |~16G              | ~5G                   | N/A                 | N/A                         |

**Summary:**
1. LLM fine-tuning and quantization are VRAM-intensive activities. If you are buying a GPU for fine-tuning purposes, please take note of the benchmark results.
2. During model training, the model states such as optimizer, gradients, and parameters contribute heavily to the VRAM usage. The outcome of the experiments shows that model 1B parameter consumes more than 2GB VRAM when loaded for inference. When model fine-tuning/training is being carried out, VRAM consumption increases by 2x to 4x. Training a model without quantization (fp32) has a high memory overhead. Try reducing the batch size in the event of hitting OOM when loading the model.
3. During model inference, each billion parameters consumes 4GB memory in FP32 precision, 2GB in FP16, and 1GB in int8, all excluding additional overhead (estimated ≤ 20%).
4. When loading a huge model (without quantization) with OOM error, `BitsAndBytes` quantization allows the model to fit into the VRAM but at the expense of lower precision. Despite that limitation, the result was acceptable, depending on the use cases. As expected, `4-bit BitsAndBytes` took longer duration to train compared to `8-bit BitsAndBytes` setting.
5. `auto-gptq` post-quantization mechanism helps to reduce the model size permanently.
6. Not all pre-trained models are suitable for fine-tuning with the same dataset. Experiments show that `falcon-7b` and `bloom-7b1` produce acceptable results but not for `codegen2-1B` model.
7. CPU cores are heavily used when saving/copying the quantized model. You may enable CML's CPU bursting feature to speed up the process.
8. GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain in float16.
9. During the training process using `BitsAndBytes` config, the forward and backward steps are done using FP16 weights and activations. 
  
### <a name="toc_2"></a>3. Preparation

#### <a name="toc_3"></a>3.1 Dataset & Model

- You may download the model (using curl) into the local folder or pinpoint the model in the code so that the API will connect and download directly from 🤗 site.<br> 
&nbsp;a. `bigscience/bloom-1b1` and `bigscience/bloom-7b1`<br>
&nbsp;b. `tiiuae/falcon-7b`<br>
&nbsp;c. `Salesforce/codegen2-1B`<br>

- You may download the dataset (using curl) into the local folder or pinpoint the dataset in the code so that the API will connect and download directly from 🤗 site.<br> 
&nbsp;a. Dataset for fine-tuning: `Shreyasrp/Text-to-SQL`<br> 
&nbsp;b. Dataset for quantization: Quantization requires sample data to calibrate and enhance quality of the quantization. In this benchmark test, [C4 dataset](https://huggingface.co/datasets/c4) is utilized as only certain datasets are allowed.<br> 

#### <a name="toc_4"></a>3.2 CML Session

- CML runs on the Kubernetes platform. When a `CML session` is requested, CML instructs K8s to schedule and provision a pod with the required resource profile.
1. Create a CML project using `Python 3.9` with `Nvidia GPU runtime`.
2. Create a CML session (Jupyter) with the resource profile of 4CPU and 64GB memory and 1GPU.
3. In the CML session, install the necessary Python packages.
```
pip install -r requirements.txt
```

### <a name="toc_5"></a>4. `bigscience/bloom-1b1`

#### <a name="toc_6"></a>4.1. Fine-Tune (w/o Quantization) > Merge > Inference

- In CML session, run this Jupyter code [ft-merge-qt.ipynb](ft-merge-qt.ipynb) to fine-tune, merge and perform a simple inference on the merged/fine-tuned model.
  
- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Below shows the outcome after loading the model into the VRAM before running the fine-tuning/training code.
```
Base Model Memory Footprint in VRAM: 4063.8516 MB
--------------------------------------
Parameters loaded for model bloom-1b1:
Total parameters: 1065.3143 M
Trainable parameters: 1065.3143 M


Data types for loaded model bloom-1b1:
torch.float32, 1065.3143 M, 100.00 %
```

<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/4e557656-abb9-409f-8a56-23601af785f9"><br>

- During fine-tuning/training:
<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/d1a594f6-c284-4cb7-bda5-17d19227626d">

- It takes ~12mins to complete the training.
```
{'loss': 0.8376, 'learning_rate': 0.0001936370577755154, 'epoch': 2.03}
{'loss': 0.7142, 'learning_rate': 0.0001935522185458556, 'epoch': 2.03}
{'loss': 0.6476, 'learning_rate': 0.00019346737931619584, 'epoch': 2.03}
{'train_runtime': 715.2236, 'train_samples_per_second': 32.96, 'train_steps_per_second': 16.48, 'train_loss': 0.8183029612163445, 'epoch': 2.03}
Training Done
```

- Inside the training_output directory:
```
$ ls -lh
total 23M
-rw-r--r--. 1 cdsw cdsw  427 Nov  6 02:07 adapter_config.json
-rw-r--r--. 1 cdsw cdsw 9.1M Nov  6 02:07 adapter_model.bin
drwxr-xr-x. 2 cdsw cdsw   11 Nov  6 01:59 checkpoint-257
drwxr-xr-x. 2 cdsw cdsw   11 Nov  6 02:03 checkpoint-514
drwxr-xr-x. 2 cdsw cdsw   11 Nov  6 02:07 checkpoint-771
-rw-r--r--. 1 cdsw cdsw   88 Nov  6 02:07 README.md
-rw-r--r--. 1 cdsw cdsw   95 Nov  6 02:07 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw  983 Nov  6 02:07 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw  14M Nov  6 02:07 tokenizer.json
-rw-r--r--. 1 cdsw cdsw 4.5K Nov  6 02:07 training_args.bin
```

- After the training is completed, merge the base model with the PEFT-trained adapters.

- Inside the merged model directory:
```
$ ls -lh
total 4.0G
-rw-r--r--. 1 cdsw cdsw  777 Nov  6 02:07 config.json
-rw-r--r--. 1 cdsw cdsw  137 Nov  6 02:07 generation_config.json
-rw-r--r--. 1 cdsw cdsw 4.0G Nov  6 02:07 pytorch_model.bin
-rw-r--r--. 1 cdsw cdsw   95 Nov  6 02:07 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw  983 Nov  6 02:07 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw  14M Nov  6 02:07 tokenizer.json
```

- Inside the base model directory:
```
$ ls -lh
total 6.0G
-rw-r--r--. 1 cdsw cdsw  693 Oct 28 02:22 config.json
-rw-r--r--. 1 cdsw cdsw 2.0G Oct 28 01:32 flax_model.msgpack
-rw-r--r--. 1 cdsw cdsw  16K Oct 28 01:27 LICENSE
-rw-r--r--. 1 cdsw cdsw 2.0G Oct 28 01:31 model.safetensors
drwxr-xr-x. 2 cdsw cdsw   11 Oct 28 01:27 onnx
-rw-r--r--. 1 cdsw cdsw 2.0G Oct 28 01:29 pytorch_model.bin
-rw-r--r--. 1 cdsw cdsw  21K Oct 28 01:27 README.md
-rw-r--r--. 1 cdsw cdsw   85 Oct 28 01:27 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw  222 Oct 28 01:33 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw  14M Oct 28 01:33 tokenizer.json
```

- Load the merged model into VRAM:
```
Merged Model Memory Footprint in VRAM: 4063.8516 MB

Data types:
torch.float32, 1065.3143 M, 100.00 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/021a1854-f943-4257-9165-f90bde98c5e8"><br>

- Run inference on the fine-tuned/merged model and the base model, compare the results.
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Fine-tuned Model Result :
SELECT Title FROM book WHERE Writer <> 'Dennis Lee'
```

```
Base Model Result :
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is Dennis Lee?
# result:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is Dennis Lee?
```

#### <a name="toc_7"></a>4.2. Quantize (GPTQ 8-bit) > Inference

- In CML session, run this Jupyter code [quantize_model.ipynb](quantize_model.ipynb) to quantize the merged model. Run [infer_Qmodel.ipynb](infer_Qmodel.ipynb) to perform a simple inference on the quantized model.

- During quantization:
<img width="1059" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/414dca58-025a-48b2-93e4-816b5781e0ce">

<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/218470a5-4358-41ce-8661-0dc8b21bf224"><br>

- Time taken to quantize:
```
Total Seconds Taken to Quantize Using cuda:0: 282.6761214733124
```

- Load the quantized model into VRAM:
```
cuda:0 Memory Footprint: 1400.0977 MB

Data types:
torch.float16, 385.5053 M, 100.00 %

```
<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/75965ac1-81ce-4c5e-8aca-83246cf674ab"><br>

- Run inference on the quantized model and check the result:
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Quantized Model Result :
SELECT Title FROM book WHERE Writer = 'Not Dennis Lee'
```

- Inside the quantized directory:
```
$ ls -lh
total 1.4G
-rw-r--r--. 1 cdsw cdsw 1.4K Nov  6 02:39 config.json
-rw-r--r--. 1 cdsw cdsw  137 Nov  6 02:39 generation_config.json
-rw-r--r--. 1 cdsw cdsw 1.4G Nov  6 02:39 pytorch_model.bin
-rw-r--r--. 1 cdsw cdsw  551 Nov  6 02:39 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw  983 Nov  6 02:39 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw  14M Nov  6 02:39 tokenizer.json
```

- Snippet of `config.json` file in the quantized model folder:
```
pretraining_tp: 1
▶
quantization_config:
batch_size: 1
bits: 8
block_name_to_quantize: "transformer.h"
damp_percent: 0.1
dataset: "c4"
desc_act: false
disable_exllama: false
group_size: 128
max_input_length: null
model_seqlen: 2048
▶
module_name_preceding_first_block: [] 2 items
pad_token_id: null
quant_method: "gptq"
sym: true
tokenizer: null
true_sequential: true
use_cuda_fp16: true
```

### <a name="toc_8"></a>5. `bigscience/bloom-7b1`

#### <a name="toc_9"></a>5.1. Fine-Tune (w/o Quantization) > Merge > Inference

- In CML session, run this Jupyter code [ft-merge-qt.ipynb](ft-merge-qt.ipynb) to fine-tune, merge and perform a simple inference on the merged/fine-tuned model.
 
- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Below shows the outcome after loading the model into the VRAM before running the fine-tuning/training code.
```
Base Model Memory Footprint in VRAM: 26966.1562 MB
--------------------------------------
Parameters loaded for model bloom-7b1:
Total parameters: 7069.0161 M
Trainable parameters: 7069.0161 M


Data types for loaded model bloom-7b1:
torch.float32, 7069.0161 M, 100.00 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/6244a0d9-f2d8-4f64-b13e-c01acf49755d"><br>

- During fine-tuning/training:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacty of 39.39 GiB of which 373.94 MiB is free. Process 1793579 has 39.02 GiB memory in use. Of the allocated memory 38.23 GiB is allocated by PyTorch, and 305.27 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

#### <a name="toc_10"></a>5.2. Fine-Tune (4-bit) > Merge > Inference

- Code Snippet:
```
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)
base_model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, use_cache = False, device_map=device_map)
```

- Below shows the outcome after loading the model into the VRAM before running the fine-tuning/training code.
```
Base Model Memory Footprint in VRAM: 4843.0781 MB
--------------------------------------
Parameters loaded for model bloom-7b1:
Total parameters: 4049.1172 M
Trainable parameters: 1028.1124 M

Data types for loaded model bloom-7b1:
torch.float16, 1029.2183 M, 25.42 %
torch.uint8, 3019.8989 M, 74.58 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/1276e377-00c3-4951-afdd-1cd42282bf0e"><br>

- During fine-tuning/training:
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/6d1ec652-9bd8-42d4-8e08-b2bb2cdc82d4">

- It takes ~83mins to complete the training.
```
'loss': 0.5777, 'learning_rate': 0.0001935522185458556, 'epoch': 2.03}
{'loss': 0.5486, 'learning_rate': 0.0001935097989310257, 'epoch': 2.03}
{'loss': 0.465, 'learning_rate': 0.00019346737931619584, 'epoch': 2.03}
{'train_runtime': 5024.8159, 'train_samples_per_second': 4.692, 'train_steps_per_second': 4.692, 'train_loss': 0.6570684858410584, 'epoch': 2.03}
Training Done
```

- After training is completed, merge the base model with the PEFT-trained adapters.
  
- Load the merged model into VRAM:
```
Merged Model Memory Footprint in VRAM: 26966.1562 MB

Data types:
torch.float32, 7069.0161 M, 100.00 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/21e7c7a3-3e50-40b4-a807-75d6a61ed3ab"><br>


- Run inference on the fine-tuned/merged model and check the result:

```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Fine-tuned Model Result :
SELECT Title FROM book WHERE Writer <> "Dennis Lee"
```

#### <a name="toc_11"></a>5.3. Quantize (GPTQ 8-bit) > Inference

- In CML session, run this Jupyter code [quantize_model.ipynb](quantize_model.ipynb) to quantize the merged model. Run [infer_Qmodel.ipynb](infer_Qmodel.ipynb) to perform a simple inference on the quantized model.
  
- During quantization:
<img width="971" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/8f0c7a71-a3b1-467f-a83c-0284e6e85dbe">
<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/2b47132c-c0e1-406c-b331-25611f1402bb"><br>
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/abc6ad51-148a-47c0-b8bb-384fe7bf2539"><br>

- Time taken to quantize:
```
Total Seconds Taken to Quantize Using cuda:0: 2073.348790884018
```

- Load the quantized model into VRAM:
```
cuda:0 Memory Footprint: 7861.3594 MB

Data types:
torch.float16, 1028.1124 M, 100.00 %
```
<img width="1060" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/1d2cb609-df98-4b62-81d9-200f65ba68d3">


- Run inference on the quantized model and check the result:
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Quantized Model Result :
SELECT Title FROM book WHERE Writer <> "Dennis Lee"
```

- Snippet of `config.json` file in the quantized model folder:
```
▶
quantization_config:
batch_size: 1
bits: 8
block_name_to_quantize: "transformer.h"
damp_percent: 0.1
dataset: "c4"
desc_act: false
disable_exllama: true
group_size: 128
max_input_length: null
model_seqlen: 2048
▶
module_name_preceding_first_block: [] 2 items
pad_token_id: null
quant_method: "gptq"
sym: true
tokenizer: null
true_sequential: true
use_cuda_fp16: true
```

### <a name="toc_12"></a>6. `tiiuae/falcon-7b`

#### <a name="toc_13"></a>6.1. Fine-Tune (w/o Quantization) > Merge > Inference

- In CML session, run this Jupyter code [ft-merge-qt.ipynb](ft-merge-qt.ipynb) to fine-tune, merge and perform a simple inference on the merged/fine-tuned model.

- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Below shows the outcome after loading the model into the VRAM before running the fine-tuning/training code.
```
Base Model Memory Footprint in VRAM: 26404.2729 MB
--------------------------------------
Parameters loaded for model falcon-7b:
Total parameters: 6921.7207 M
Trainable parameters: 6921.7207 M


Data types for loaded model falcon-7b:
torch.float32, 6921.7207 M, 100.00 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/dd499a83-f7b9-41d3-8c59-955e9e16a0fc"><br>

- During fine-tuning/training:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.11 GiB. GPU 0 has a total capacty of 39.39 GiB of which 345.94 MiB is free. Process 1618370 has 39.04 GiB memory in use. Of the allocated memory 37.50 GiB is allocated by PyTorch, and 1.05 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/0e91da7b-f704-4b03-a824-b5391819a6c8"><br>

#### <a name="toc_14"></a>6.2. Fine-Tune (8-bit) > Merge > Inference

- Code Snippet:
```
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
base_model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, use_cache = False, device_map=device_map)
```

- Below shows the outcome after loading the model into the VRAM before running the fine-tuning/training code.
```
Base Model Memory Footprint in VRAM: 6883.1384 MB
--------------------------------------
Parameters loaded for model falcon-7b:
Total parameters: 6921.7207 M
Trainable parameters: 295.7690 M


Data types for loaded model falcon-7b:
torch.float16, 295.7690 M, 4.27 %
torch.int8, 6625.9517 M, 95.73 %
```
<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/a228e9ef-6e4d-438b-8b3a-53d74f7d127b"><br>

- During fine-tuning/training:
<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/6a743e35-672f-4163-916b-0b491d88bf42">

```
warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
```

- It takes ~65mins to complete the training.
```
{'loss': 0.5285, 'learning_rate': 0.00019198269279714942, 'epoch': 2.04}
{'loss': 0.4823, 'learning_rate': 0.00019194027318231952, 'epoch': 2.04}
{'loss': 0.4703, 'learning_rate': 0.00019189785356748962, 'epoch': 2.04}
{'train_runtime': 3911.2114, 'train_samples_per_second': 6.027, 'train_steps_per_second': 6.027, 'train_loss': 0.5239265531830902, 'epoch': 2.04}
Training Done
```

- After the training is completed, merge the base model with the PEFT-trained adapters.

- Inside the merged model directory:
```
$ ls -lh
total 26G
-rw-r--r--. 1 cdsw cdsw 1.2K Nov  6 04:55 config.json
-rw-r--r--. 1 cdsw cdsw  118 Nov  6 04:55 generation_config.json
-rw-r--r--. 1 cdsw cdsw 4.7G Nov  6 04:55 pytorch_model-00001-of-00006.bin
-rw-r--r--. 1 cdsw cdsw 4.7G Nov  6 04:55 pytorch_model-00002-of-00006.bin
-rw-r--r--. 1 cdsw cdsw 4.7G Nov  6 04:55 pytorch_model-00003-of-00006.bin
-rw-r--r--. 1 cdsw cdsw 4.7G Nov  6 04:55 pytorch_model-00004-of-00006.bin
-rw-r--r--. 1 cdsw cdsw 4.7G Nov  6 04:55 pytorch_model-00005-of-00006.bin
-rw-r--r--. 1 cdsw cdsw 2.7G Nov  6 04:55 pytorch_model-00006-of-00006.bin
-rw-r--r--. 1 cdsw cdsw  17K Nov  6 04:55 pytorch_model.bin.index.json
-rw-r--r--. 1 cdsw cdsw  313 Nov  6 04:55 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw 2.6K Nov  6 04:55 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw 2.7M Nov  6 04:55 tokenizer.json
```

- Load the merged model into VRAM:
```
Merged Model Memory Footprint in VRAM: 26404.2729 MB

Data types:
torch.float32, 6921.7207 M, 100.00 %
```
<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/2a411f32-4220-4b90-a1ab-4df1db8c4c8d"><br>


- Run inference on the fine-tuned/merged model and the base model, compare the results.

```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Fine-tuned Model Result :
SELECT Title FROM book WHERE Writer <> 'Dennis Lee'
```

```
Base Model Result :
Title Writer
# Explanation:
The result shows the titles of the books whose writer is not Dennis Lee.
# 5.3.3.4.4.3.4.3.4.3.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
```

#### <a name="toc_15"></a>6.3. Quantize (GPTQ 8-bit) > Inference

- In CML session, run this Jupyter code [quantize_model.ipynb](quantize_model.ipynb) to quantize the merged model. Run [infer_Qmodel.ipynb](infer_Qmodel.ipynb) to perform a simple inference on the quantized model.

- During quantization:
<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/116479a1-2941-485d-953d-63791e024ff7">


- Time taken to quantize:
```
Total Seconds Taken to Quantize Using cuda:0: 1312.4991219043732
```

- Load the quantized model into VRAM:
```
cuda:0 Memory Footprint: 7038.3259 MB

Data types:
torch.float16, 295.7690 M, 100.00 %
```
<img width="976" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/b4eedd48-fa3d-48a5-bf88-863975f58438">


- Run inference on the quantized model and check the result:
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Quantized Model Result :
SELECT Title FROM book WHERE Writer <> 'Dennis Lee'
```

- Inside the quantized directory:
```
$ ls -lh
total 6.9G
-rw-r--r--. 1 cdsw cdsw 1.7K Nov  6 05:26 config.json
-rw-r--r--. 1 cdsw cdsw  118 Nov  6 05:26 generation_config.json
-rw-r--r--. 1 cdsw cdsw 4.7G Nov  6 05:26 pytorch_model-00001-of-00002.bin
-rw-r--r--. 1 cdsw cdsw 2.3G Nov  6 05:26 pytorch_model-00002-of-00002.bin
-rw-r--r--. 1 cdsw cdsw  61K Nov  6 05:26 pytorch_model.bin.index.json
-rw-r--r--. 1 cdsw cdsw  541 Nov  6 05:26 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw 2.6K Nov  6 05:26 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw 2.7M Nov  6 05:26 tokenizer.json
```

- Snippet of `config.json` file in the quantized model folder:
```
▶
quantization_config:
batch_size: 1
bits: 8
block_name_to_quantize: "transformer.h"
damp_percent: 0.1
dataset: "c4"
desc_act: false
disable_exllama: false
group_size: 128
max_input_length: null
model_seqlen: 2048
▶
module_name_preceding_first_block: [] 1 item
pad_token_id: null
quant_method: "gptq"
sym: true
tokenizer: null
true_sequential: true
use_cuda_fp16: true
rope_scaling: null
rope_theta: 10000
torch_dtype: "float16"
transformers_version: "4.35.0.dev0"
use_cache: true
vocab_size: 65024
```

### <a name="toc_16"></a>7. `Salesforce/codegen2-1B`

#### <a name="toc_17"></a>7.1. Fine-Tune (w/o Quantization) > Merge > Inference

- In CML session, run this Jupyter code [ft-merge-qt.ipynb](ft-merge-qt.ipynb) to fine-tune, merge and perform a simple inference on the merged/fine-tuned model.
  
- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Below shows the outcome after loading the model into the VRAM before running the fine-tuning/training code.
```
Base Model Memory Footprint in VRAM: 3937.0859 MB
--------------------------------------
Parameters loaded for model codegen2-1B:
Total parameters: 1015.3062 M
Trainable parameters: 1015.3062 M


Data types for loaded model codegen2-1B:
torch.float32, 1015.3062 M, 100.00 %
```

<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/0c7e3350-5bf7-45f0-a7cb-2afdf7abafbf">

- During fine-tuning/training:
<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/88679eb2-c280-4a7f-a2ef-1020c87fa120">


- It takes ~12mins to complete the training.
```
{'loss': 2.8109, 'learning_rate': 0.00019189785356748962, 'epoch': 2.04}
{'loss': 2.2957, 'learning_rate': 0.00019185543395265972, 'epoch': 2.04}
{'loss': 2.598, 'learning_rate': 0.00019181301433782982, 'epoch': 2.04}
{'train_runtime': 683.683, 'train_samples_per_second': 34.481, 'train_steps_per_second': 34.481, 'train_loss': 3.380507248720025, 'epoch': 2.04}
Training Done
```

- After the training is completed, merge the base model with the PEFT-trained adapters.
- Load the merged model into VRAM:
```
Merged Model Memory Footprint in VRAM: 3937.0859 MB

Data types:
torch.float32, 1015.3062 M, 100.00 %
```
<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/c9f9e672-2d61-40e5-af24-1e1b07e6e9fa"><br>

- Run inference on the fine-tuned/merged model and the base model:
```
--------------------------------------
Prompt:
# Instruction:
Use the context below to produce the result
# context:
CREATE TABLE book (Title VARCHAR, Writer VARCHAR). What are the titles of the books whose writer is not Dennis Lee?
# result:

--------------------------------------
Fine-tuned Model Result :
    Result:
    SELECT t1.name FROM table_code JOINCT (name INTEGER), How many customers who have a department?
```

```
Base Model Result :
port,,vt,(vt((var(,st#
```


### <a name="toc_18"></a>8. Bonus: Use Custom Gradio for Inference

- In CML session, execute this Jupyter code [gradio_infer.ipynb](gradio_infer.ipynb) to run inference on a specific model using the custom Gradio interface.
- This Gradio interface is designed to compare the inference results between the base model and the fine-tuned/merged model.
- It also displays the GPU memory status after loading the selected model successfully. User experience is depicted below.

<p align="left"><img src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/blob/main/images/gradio_infer.gif" width="700"></p>

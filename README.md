<img width="973" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/8948c012-15e6-44be-86c1-4fb7f103ff83">LLM: Fine-Tune > Merge > Quantize > Infer .. on CML
===

## <a name="toc_0"></a>Table of Contents
[//]: # (TOC)
[1. Objective](#toc_0)<br>
[2. Summary & Benchmark Score](#toc_2)<br>
[3. Preparation](#toc_3)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1. Dataset & Model](#toc_4)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2. CML Session](#toc_5)<br>
[4. bigscience/bloom-1b1](#toc_6)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_7)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2. Quantize > Inference](#toc_8)<br>
[5. bigscience/bloom-7b1](#toc_9)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_10)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1. Fine-Tune (8-bit) > Merge > Inference](#toc_11)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2. Quantize > Inference](#toc_12)<br>
[6. tiiuae/falcon-1b](#toc_13)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_14)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[6.2. Quantize > Inference](#toc_15)<br>
[7. tiiuae/falcon-7b](#toc_16)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_17)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.1. Fine-Tune (8-bit) > Merge > Inference](#toc_18)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[7.2. Quantize > Inference](#toc_19)<br>
[8. Salesforce/codegen2-1B](#toc_20)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[8.1. Fine-Tune (w/o Quantization) > Merge > Inference](#toc_21)<br>

[//]: # (/TOC)

### <a name="toc_0"></a>1. Objective

- In the event that you have limited GPU resources or even have no GPU in your infrastructure landscape, you may run your GenAI application using quantized models. This articles focuses on how to quantize your language models in 8, 4, or even 2 bits without **significant** performance degradation and quicker inference speed, with the help of Transformers API.
GPTQ, a Post-Training Quantization (PTQ) technique.
- GPTQ adopts a mixed int4/fp16 quantization scheme where weights are quantized as int4 while activations remain in float16. During inference, weights are dequantized on the fly and the actual compute is performed in float16.
- bitsandbytes (zero-shot quantization)
- To comprehend this, it’s crucial to realize that during model training, the model states are the main contributors to memory usage. These include tensors composed of optimizer states, gradients, and parameters. In addition to these model states, there are activations, temporary buffers, and fragmented memory, collectively known as residual states, that consume the remaining memory.
- The latest way to train big models using the newest NVIDIA graphics cards uses a method known as mixed-precision (FP16/32) training. FP32 is called full precision (4 bytes), while FP16 are referred to as half-precision (2 bytes). Here, important model components like parameters and activations are stored as FP16. This storage method allows these graphics cards to process large amounts of data very quickly.
- During this training process, both the forward and backward steps are done using FP16 weights and activations. However, to properly calculate and apply the updates at the end of the backward step, the mixed-precision optimizer keeps an FP32 copy of the parameters and all other states used in the optimizer.



#### <a name="toc_2"></a>2. Summary & Benchmark Score

- Table shows the benchmark result of fine-tuning the specific base model with **Text-to-SQL** dataset.
  
| Model | Training | Duration | 
| :---      |     :---:      |   ---: |
| bloom-1b  | No quantization     | sec   |
| bloom-1b  | BitsAndBytes      | sec     |

- Quantization: A quick check at the Open LLM Leaderboard reveals that performance degradation is quite minimal.
  
### <a name="toc_3"></a>3. Preparation

#### <a name="toc_4"></a>3.1 Dataset & Model

- Download or use the following the following model directly from 🤗.<br> 
&nbsp;a. `bigscience/bloom-1b1`<br>
&nbsp;b. `tiiuae/falcon-7b`<br>
&nbsp;c. `Salesforce/codegen2-1B`<br>

- Download or use the following sample dataset directly from 🤗. <br> 
&nbsp;a. Dataset for fine-tuning: <br> 
&nbsp;b. Dataset for quantization: Quantization requires sample data to calibrate and enhance quality of the quantization. In this benchmark test, [C4 dataset](https://huggingface.co/datasets/c4) is utilized. C4 is a large-scale, multilingual collection of web text gathered from the Common Crawl project. <br> 

#### <a name="toc_5"></a>3.2 CML Session

- CML (Cloudera Machine Learning) runs on the Kubernetes platform. When a `CML session` is requested, CML instructs K8s to schedule and provision a pod with the required resource profile.
1. Create a CML project using Python 3.9 with Nvidia GPU runtime.
2. Create a CML session (Jupyter) with the resource profile of 4CPU and 64GB memory and 1GPU.
3. In the CML session, install the necessary Python packages.
```
pip install -r requirements.txt
```

### <a name="toc_6"></a>4. `bigscience/bloom-1b1`

#### <a name="toc_7"></a>4.1. Fine-Tune (w/o Quantization) > Merge > Inference

- Use this Jupyter code to fine-tune, merge and perform a simple inference on the merged model.
  
- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Load model before fine-tuning/training starts:
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

#### <a name="toc_8"></a>4.2. Quantize > Inference
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

- Run inference on the quantized model:
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


### <a name="toc_16"></a>7. `Bigscience/bloom-1b1`

#### <a name="toc_17"></a>7.1. Fine-Tune (wo Quantization) > Merge > Inference

- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Load model into VRAM before fine-tuning/training starts:
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

#### <a name="toc_18"></a>7.2. Fine-Tune (w 4-bit Quantization) > Merge > Inference

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

- Load model into VRAM before fine-tuning/training starts:
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


- Run inference on the fine-tuned/merged model:

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

#### <a name="toc_19"></a>7.3. Quantize > Inference
- During quantization:
<img width="971" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/8f0c7a71-a3b1-467f-a83c-0284e6e85dbe"><br>


- Time taken to quantize:
```

```

- Snippet of `config.json` file in the quantized model folder:
```

```

- Load the quantized model into VRAM:
```
cuda:0 Memory Footprint: 7038.3259 MB

Data types:
torch.float16, 295.7690 M, 100.00 %
```
<img width="976" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/b4eedd48-fa3d-48a5-bf88-863975f58438">


- Run inference on the quantized model:
```

```



### <a name="toc_16"></a>7. `tiiuae/falcon-7b`

#### <a name="toc_17"></a>7.1. Fine-Tune (wo Quantization) > Merge > Inference

- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Load model into VRAM before fine-tuning/training starts:
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

#### <a name="toc_18"></a>7.2. Fine-Tune (w 8-bit Quantization) > Merge > Inference

- Code Snippet:
```
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
base_model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, use_cache = False, device_map=device_map)
```

- Load model into VRAM before fine-tuning/training starts:
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

- After training is completed, merge the base model with the PEFT-trained adapters.
  
- Load the merged model into VRAM:
```
Merged Model Memory Footprint in VRAM: 26404.2729 MB

Data types:
torch.float32, 6921.7207 M, 100.00 %
```
<img width="974" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/2a411f32-4220-4b90-a1ab-4df1db8c4c8d"><br>


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
SELECT Title FROM book WHERE Writer <> 'Dennis Lee'
```

```
Base Model Result :
Title Writer
# Explanation:
The result shows the titles of the books whose writer is not Dennis Lee.
# 5.3.3.4.4.3.4.3.4.3.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
```

#### <a name="toc_19"></a>7.3. Quantize > Inference
- During quantization:
<img width="975" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/116479a1-2941-485d-953d-63791e024ff7">


- Time taken to quantize:
```
Total Seconds Taken to Quantize Using cuda:0: 1312.4991219043732
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

- Load the quantized model into VRAM:
```
cuda:0 Memory Footprint: 7038.3259 MB

Data types:
torch.float16, 295.7690 M, 100.00 %
```
<img width="976" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-Infer-CML/assets/35444414/b4eedd48-fa3d-48a5-bf88-863975f58438">


- Run inference on the quantized model:
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

### <a name="toc_20"></a>8. `Salesforce/codegen2-1B`

#### <a name="toc_21"></a>8.1. Fine-Tune (w/o Quantization) > Merge > Inference

- Use this Jupyter code `` to fine-tune, merge and perform a simple inference on the merged model.
  
- Code Snippet:
```
base_model = AutoModelForCausalLM.from_pretrained(base_model, use_cache = False, device_map=device_map)
```

- Load model before fine-tuning/training starts:
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

#### <a name="toc_8"></a>4.2. Quantize > Inference
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

- Run inference on the quantized model:
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



#### Notes

- During quantization process:

<img width="1056" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/a7935a5b-3b3d-419b-8257-8635f829e4e9">

<img width="1023" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/e07355d0-6f08-4fe0-a708-40380c1323cd">

- When exllama is enabled, 'Assertion error:`
- Disabling exllama allowing the quantization process to complete. Notice that CPU is also being used:

<img width="1022" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/127c3e28-b194-407d-acef-f7f3a75b70ce">

<img width="900" alt="image" src="https://github.com/dennislee22/FT-Merge-Quantize-CML/assets/35444414/daf4c454-4689-4e5f-99e4-5dfb5216ebfa">

```
Total Seconds Taken to Quantize Using cuda:0: 1350.0081555843353
```

```
ls -lh gptq-merged_falcon-7b_4bit
total 3.8G
-rw-r--r--. 1 cdsw cdsw 1.7K Nov  1 05:42 config.json
-rw-r--r--. 1 cdsw cdsw  118 Nov  1 05:42 generation_config.json
-rw-r--r--. 1 cdsw cdsw 3.8G Nov  1 05:42 pytorch_model.bin
-rw-r--r--. 1 cdsw cdsw  541 Nov  1 05:42 special_tokens_map.json
-rw-r--r--. 1 cdsw cdsw 2.6K Nov  1 05:42 tokenizer_config.json
-rw-r--r--. 1 cdsw cdsw 2.7M Nov  1 05:42 tokenizer.json
```

8-bit Parameter Precision Info:
```
cuda:0 Memory Footprint: 7038.3259 MB
Total parameters: 295.7690 M
Trainable parameters: 295.7690 M

Data types:
torch.float16, 295.7690 M, 100.00 %
```

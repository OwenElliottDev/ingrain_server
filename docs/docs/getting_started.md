# Getting Started

Ingrain is a scalable wrapper for [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) that makes using it with sentence transformers, timm, and open CLIP models easy.

The server handles all model conversion, loading, and preprocessing. It manages tokenizers, image downloading, image preprocessing, and other complexities of running a model. 

## Setup

To run Ingrain, you need to have Docker installed. Ingrain server works in tandem with Triton to automate much of the process for serving OpenCLIP, sentence transformers, and timm models. 

The easiest way to get started is with a docker compose file, which will run Triton and the Ingrain server.

```yml
version: "3.9"

services:
  ingrain:
    image: owenpelliott/ingrain-server:latest
    container_name: ingrain
    ports:
      - "8686:8686"
      - "8687:8687"
    environment:
      TRITON_GRPC_URL: triton:8001
      MAX_BATCH_SIZE: 8
    depends_on:
      - triton
    volumes:
      - ./model_repository:/app/model_repository 
      - ${HOME}/.cache/huggingface:/app/model_cache/
  triton:
    image: nvcr.io/nvidia/tritonserver:25.04-py3
    container_name: triton
    runtime: nvidia # comment out if not using GPU
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - LD_PRELOAD=/usr/lib/$(uname -m)-linux-gnu/libtcmalloc.so.4
    shm_size: "256m"
    command: >
      tritonserver
      --model-repository=/models
      --model-control-mode=explicit
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./model_repository:/models
    restart:
      always
```

To run without a GPU comment out the `runtime: nvidia` in the triton container.

Spin up the server with:

```bash
docker compose up
```


You can also install a python client to get started quickly:

```bash
pip install ingrain
```

## Example Usage

Ingrain has two main components: the model server and the inference server. The model server handles all model loading and unloading operations whereas the inference server handles all inference requests on loaded models.

### Loading Models

Models are accessable via the same naming conventions as their underlying libraries (`open_clip_torch`, `sentence_transformers`, or `timm`). For example to load a MobileCLIP model in `open_clip` you would use the following code:

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP-S2', pretrained='datacompdr')
model.eval()
tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')
```

To do the same in Ingrain, you would use the following code:

```python
import ingrain

client = ingrain.Client()
client.load_clip_model(name="MobileCLIP-S2", pretrained="datacompdr")
```

A similar pattern holds for `sentence_transformers` models:

Sentence Transformers:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-small-v2')
```

Ingrain:

```python
import ingrain

client = ingrain.Client()
client.load_sentence_transformer_model(name="intfloat/e5-small-v2")
```

### Inference

Inference requests are made to the Inference Server. The server can handle both text and image inputs. Images can be URLs to image assets or base64 encoded images. Requests can include single items or batches of items - the server will also dynamically batch requests to optimize performance.

For example to encode a list of sentences using the `intfloat/e5-small-v2` model:
```python
import ingrain

client = ingrain.Client()
client.load_sentence_transformer_model(name="intfloat/e5-small-v2")

response = client.infer(
    name="intfloat/e5-small-v2", 
    text=["I am a sentence.", "I am another sentence.", "I am a third sentence."]
)
```

This will return a dictionary like so:

```python
{
    'textEmbeddings': [
        [-0.028349684551358223, 0.03690376505255699, ...],
        [-0.657342546546734223, 0.46323376505255699, ...],
        [-0.035745636365643294, 0.95440376505255699, ...]
    ], 
    'imageEmbeddings': None, 
    'processingTimeMs': 17.091903000618913
}
```
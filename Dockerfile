FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive


# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
    git \
    bash \
    nano \
    wget \
    python3.10 \
    python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# Install PyTorch and related packages (part 1)
RUN pip install --upgrade torch==2.2.1

# Install other Python packages
RUN pip3 install --upgrade datasets
RUN pip3 install --upgrade wandb
RUN pip3 install --upgrade tokenizers
RUN pip3 install --upgrade tqdm
RUN pip3 install --upgrade nltk
RUN pip3 install --upgrade scipy
RUN pip3 install --upgrade huggingface_hub

RUN pip3 install transformers==4.40.1
RUN pip3 install peft==0.10.0

RUN pip3 install git+https://github.com/huggingface/accelerate.git
RUN pip3 install git+https://github.com/huggingface/trl.git

# required for flash attention
RUN pip3 install --upgrade packaging
RUN pip3 install --upgrade ninja
RUN pip3 install --upgrade flash-attn==2.5.8 --no-build-isolation
RUN pip3 install --upgrade bitsandbytes==0.43.0
RUN pip3 install --upgrade streamlit==1.33.0 langchain==0.1.16 langchain-community==0.0.34 sentence-transformers==2.7.0 vllm==0.4.1 pypdf python-dotenv chromadb==0.5.0 cryptography==3.1

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
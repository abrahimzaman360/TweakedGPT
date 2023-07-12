# TweakedGPT - a PrivateGPT Implementation for Alpaca Model (LLAMA)
using PrivateGPT implementation for using alpaca models;

## Install Dependencies (use ! before each command for Google Colab or Notebooks):
``` pip -q install git+https://github.com/huggingface/transformers ``` 

``` pip install -q -U git+https://github.com/huggingface/peft.git ``` 

``` pip -q install bitsandbytes accelerate python-telegram-bot xformers datasets loralib sentencepiece langchain llama-cpp-python deeplake ``` 

``` pip install -r requirements.txt ```

## Use Notebook or Manually Run Files (In Sequence):
``` python constants.py ``` 

``` python ingest.py ``` 

``` python main.py ```

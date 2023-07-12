from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import DeepLake
from telegram import Update
from transformers import LlamaTokenizer, LlamaForCausalLM
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
# from langchain.llms import GPT4All, LlamaCpp
from transformers import pipeline, GenerationConfig
from langchain.llms import HuggingFacePipeline
import torch
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ['EMBEDDINGS_MODEL_NAME'] = 'all-MiniLM-L6-v2'

import argparse

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')


embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = DeepLake(dataset_path="./my_deeplake/", embedding_function=embeddings, read_only=True)
retriever = db.as_retriever()
# activate/deactivate the streaming StdOut callback for LLMs
callbacks = [StreamingStdOutCallbackHandler()]
# Prepare the LLM
tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
model = LlamaForCausalLM.from_pretrained(
    "chainyo/alpaca-lora-7b",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1000,
)

llm = HuggingFacePipeline(pipeline=pipe)
# llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                 return_source_documents=True)

def Ask(question):
    query = question
    res = qa(query)
    answer, docs = res['result'], res['source_documents']

    print(f"Question: {query}")
    print(f"Answer: {answer}")

    print(f"\nSources: ")
    for document in docs:
        print(f'{document.metadata["source"]} :')

while True:
    query = input("Ask Me Questions: ")
    Ask(query)

# For Telegram Bot Only:
"""
async def greet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f'Hello {update.effective_user.full_name}, How may I help you?')


async def respondMessages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.effective_message.text
    res = qa(update.effective_message.text)
    answer, docs = res['result'], res['source_documents']

    # Print the result
    await update.message.reply_text(f'Question: {query}')
    await update.message.reply_text(f'Answer: {answer}')

    # Print the relevant sources used for the answer
    for document in docs:
        await update.message.reply_text(f'{document.metadata["source"]} :')

# Add Your Bot Token Here:
app = ApplicationBuilder().token("Your Bot Token Here").build()

app.add_handler(CommandHandler("greet", greet))
app.add_handler(MessageHandler(filters.TEXT, respondMessages))

print('Bot is running...')
try:
    app.run_polling()
except Exception as e:
    print(e)
"""


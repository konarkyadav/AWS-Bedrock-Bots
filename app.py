import json
import os
import sys
import boto3
import streamlit as web_app_interface

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock_client=boto3.client(service_name="bedrock-runtime")
embeddings_service=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock_client)

def load_data():
    pdf_loader=PyPDFDirectoryLoader("data")
    pdf_documents=pdf_loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
    split_docs=splitter.split_documents(pdf_documents)
    return split_docs

def setup_vector_store(split_docs):
    vector_store=FAISS.from_documents(
        split_docs,
        embeddings_service
    )
    vector_store.save_local("vector_index")

def bedrock_model():
    model=Bedrock(model_id="ai21.j2-mid-v1",client=bedrock_client,
                model_kwargs={'maxTokens':512})
    
    return model

def llama_model():
    model=Bedrock(model_id="meta.llama2-70b-chat-v1",client=bedrock_client,
                model_kwargs={'max_gen_len':512})
    
    return model

prompt_structure = """
Human: Use the context to provide a concise answer to the question below. Summarize in 
250 words with detailed explanations. If unsure, state "I don't know" rather than guessing.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_structure, input_variables=["context", "question"]
)

def generate_response(model,vector_store,query):
    qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="response_generation",
    retriever=vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    response=qa_chain({"query":query})
    return response['result']


def main():
    web_app_interface.set_page_config("PDF Chat Bot")
    
    web_app_interface.header("Interactive PDF Chat using AWS Bedrock💁")

    question_input = web_app_interface.text_input("Enter your question related to the PDF content")

    with web_app_interface.sidebar:
        web_app_interface.title("Refresh or Build Vector Store:")
        
        if web_app_interface.button("Refresh Vectors"):
            with web_app_interface.spinner("Updating..."):
                documents = load_data()
                setup_vector_store(documents)
                web_app_interface.success("Vector Store Updated")

    if web_app_interface.button("Generate with Claude"):
        with web_app_interface.spinner("Fetching..."):
            vector_index = FAISS.load_local("vector_index", embeddings_service)
            ai_model=bedrock_model()
            
            web_app_interface.write(generate_response(ai_model,vector_index,question_input))
            web_app_interface.success("Response Generated")

    if web_app_interface.button("Generate with Llama2"):
        with web_app_interface.spinner("Fetching..."):
            vector_index = FAISS.load_local("vector_index", embeddings_service)
            ai_model=llama_model()
            
            web_app_interface.write(generate_response(ai_model,vector_index,question_input))
            web_app_interface.success("Response Generated")

if __name__ == "__main__":
    main()

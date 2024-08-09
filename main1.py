# FastAPI Libraries
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


import pandas as pd
import os
import json
import shutil
from typing import List

# Import Weaviate and langchian libraries
import weaviate
from weaviate.classes.query import Filter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Import necessary functions for data processing and analysis
from process_sheet import process_sheet
from filehandle import save_uploaded_file, load_local_file
from load_select_list import load_selection_list, filter_selection_list
from store_selected_list import store_selected_list

app = FastAPI()

# CORS middleware to allow requests from a frontend (like Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Environment Variables
load_dotenv()

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

# Retrieve the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

# Weaviate URL and API initialization
URL = "https://6jjbfwkcrz6kszkg8z3mg.c0.asia-southeast1.gcp.weaviate.cloud"
client = weaviate.connect_to_wcs(
    cluster_url = URL,
    auth_credentials = weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
)

# Load the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1000)

# Function to load and split PDF documents
def load_pdf_documents(file_path):
    pdf_loader = PyPDFLoader(file_path=file_path)
    pdf_pages = pdf_loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pdf_pages)
    texts = [d.page_content for d in docs]
    return texts

@app.post("/admin")
async def admin_page(
    sector: str = Form(...),
    sub_sector: str = Form(...),
    sales_person: str = Form(...),
    uploaded_file_excel: UploadFile = File(...),
    uploaded_file_pdf: UploadFile = File(...)
):
    try:
        excel_file_path = save_uploaded_file(uploaded_file_excel)
        pdf_file_path = save_uploaded_file(uploaded_file_pdf)

        excel_texts = []
        pdf_texts = []

        # Process Excel file
        excel_file = pd.ExcelFile(excel_file_path)
        sheets = excel_file.sheet_names
        list_of_items = []
        for sheet in sheets:
            if "Units" in sheet:
                continue
            else:
                data = process_sheet(excel_file_path, sheet)
                for sheet_name, sheet_data in data.items():
                    for table_name, values_df in sheet_data.items():
                        list_of_items.append({
                            "sector": sector,
                            "Sub Sector": sub_sector,
                            "Sales Person": sales_person,
                            "Sheet Name": sheet_name,
                            "Table Name": table_name,
                            "Values Table": values_df.to_dict(),
                        })
                json_file_name = f"{sheet}.json"
                directory = os.path.join("Json", uploaded_file_excel.filename)
                os.makedirs(directory, exist_ok=True)
                file_path = os.path.join(directory, json_file_name)
                with open(file_path, 'w') as f:
                    json.dump({"data": list_of_items}, f, indent=4)
                with open(file_path, 'r') as file:
                    sdata = json.load(file)
                sheet_data = sdata['data']
                excel_texts = [str(text) for text in sheet_data]

        # Process PDF file
        pdf_texts = load_pdf_documents(pdf_file_path)

        # Combine all texts from Excel and PDF files for indexing in Weaviate Vector Store
        all_texts = excel_texts + pdf_texts

        db = WeaviateVectorStore.from_texts(
            texts=all_texts,
            client=client,
            embedding=OpenAIEmbeddings(chunk_size=500, model="text-embedding-ada-002"),
            index_name="Akshay_Final_DB"
        )

        return JSONResponse(content={"message": "Files processed and indexed successfully"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/user")
async def user_page(question: str):
    try:
        new_db = WeaviateVectorStore(
            client=client,
            index_name="Akshay_Final_DB",
            text_key="text",
            embedding=OpenAIEmbeddings(chunk_size=500, model="text-embedding-ada-002")
        )
        retriever = new_db.as_retriever(search_type="mmr", search_kwargs={'k': 5})
        system_prompt = (
            "You are an assistant for question answering tasks providing tabular format if possible. "
            "Fetch the relevant context from the provided documents"
            "If the context does not provide enough information, say you don't know. "
            "Provide a concise answer based on the context. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, qa_chain)
        result = chain.invoke({"input": question})
        return JSONResponse(content={"answer": result["answer"]}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


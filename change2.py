# Import Libraries
import pandas as pd
import streamlit as st
import os
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries for Weaviate
import weaviate
# from weaviate.classes.query import Filter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Import necessary functions for data processing and analysis
from process_sheet import process_sheet
from filehandle import save_uploaded_file, load_local_file
#from filter_data import filter_data
from load_select_list import load_selection_list, filter_selection_list
from store_selected_list import store_selected_list


# Load Environment Variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI model
llm = ChatOpenAI(model = "gpt-4o-mini")

# Retrieve the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

# Weaviate URL and API initialization
URL = "https://6jjbfwkcrz6kszkg8z3mg.c0.asia-southeast1.gcp.weaviate.cloud"
client = weaviate.connect_to_wcs(
    cluster_url = URL,
    auth_credentials = weaviate.auth.AuthApiKey("bJeXddKn5Uue59q5Fcp7PYzLg5aB2rKWO48w")
)

# Load the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size = 1536)

# Session State variable for Excel load button
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

# Function to load and split PDF documents
def load_pdf_documents(file_path):
    pdf_loader = PyPDFLoader(file_path = file_path)
    pdf_pages = pdf_loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size = 1536, chunk_overlap = 128)
    docs = text_splitter.split_documents(pdf_pages)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    return texts, metadatas

# Main app
st.header("Ask Your Query with QueryMate")


# Sidebar for Sector, subsector, and teams input
with st.sidebar:
    sector, sub_sector, sales_person = load_selection_list("Full_Reports_Sold.xlsx")
    filtered_list = {
        "sector": sector,
        "sub_sector": sub_sector,
        "sales_person": sales_person
    }
    # Load and split PDF documents
    uploaded_file_excel = st.file_uploader("Upload an Excel file", type = ["xlsx", "xls"])
    uploaded_file_pdf = st.file_uploader('Upload a PDF file', type = 'pdf')
    excel_file_path = None
    pdf_file_path = None

# Process uploaded files and save them as JSON files
data = {}
excel_texts = []
pdf_texts = []

with st.spinner('Processing files...'):
    if uploaded_file_excel:
        excel_file_path = save_uploaded_file(uploaded_file_excel)

    if uploaded_file_pdf:
        pdf_file_path = save_uploaded_file(uploaded_file_pdf)

with st.spinner('Loading files...'):
    if not uploaded_file_excel or not uploaded_file_pdf:
        local_files_excel = [f for f in os.listdir(os.path.join(os.getcwd(), "Excel")) if f.endswith(".xlsx")]
        local_files_pdf = [f for f in os.listdir(os.path.join(os.getcwd(), "PDF")) if f.endswith(".pdf")]


with st.spinner('Processing Excel files...'):
    if excel_file_path or pdf_file_path:
        try:
            excel_file = pd.ExcelFile(excel_file_path)
            pdf_texts, pdf_metadata = load_pdf_documents(pdf_file_path)
            sheets = excel_file.sheet_names

            all_texts = []
            meta_data = []

            for sheet in sheets:
                if "Units" in sheet:
                    continue
                else:
                    list_of_items = []
                    list_of_metadatas = []
                    data = process_sheet(excel_file_path, sheet)

                    for sheet_name, sheet_data in data.items():
                        for table_name, values_df in sheet_data.items():
                            list_of_items.append({
                                "sheet_name": sheet_name,
                                "table_name": table_name,
                                "values_table": values_df.to_dict()
                                # "pdf_content": pdf_texts
                            })

                            for i, text in enumerate(pdf_metadata):
                                list_of_metadatas.append({
                                    "sector": sector,
                                    "sub_sector": sub_sector,
                                    "sales_person": sales_person,                                
                                    "pdf_metadata": text
                                })
                    list_of_items.append(
                        {
                            "pdf_content": pdf_texts
                        }
                    )
                    json_file_name = f"{sheet}.json"
                    meta_name = f"{sheet}_metadata.json"
                    directory = os.path.join("Json", uploaded_file_excel.name)
                    os.makedirs(directory, exist_ok=True)

                    file_path_data = os.path.join(directory, json_file_name)
                    file_path_metadata = os.path.join(directory, meta_name)

                    with open(file_path_data, 'w') as f:
                        json.dump({"data": list_of_items}, f, indent=4)

                    with open(file_path_metadata, 'w') as f:
                        json.dump({"metadata": list_of_metadatas}, f, indent=4)

                    with open(file_path_data, 'r') as file:
                        sdata = json.load(file)

                    with open(file_path_metadata, "r") as file:
                        smetadata = json.load(file)

                    sheet_data = sdata['data']
                    sheet_meta = smetadata['metadata']

                    all_texts.extend([str(text) for text in sheet_data])
                    meta_data.extend([meta for meta in sheet_meta])

        except FileNotFoundError:
            st.error(f"Excel file not found: {excel_file_path}")
            st.error(f"PDF file not found: {pdf_file_path}")
        except Exception as e:
            st.error(f"Error loading file: {e}")


all_texts = all_texts
meta_data = meta_data

# Sidebar for actions
sidebar = st.sidebar
sidebar.title("Sidebar Actions")

# Process PDF documents
with st.spinner('Loading Excel and PDF files...'):
    if sidebar.button("Load Excel and PDF File"):
        view_list = store_selected_list(filtered_list, uploaded_file_excel, uploaded_file_pdf)
        try:
            db = WeaviateVectorStore.from_texts(
                texts = all_texts,
                client = client,
                embedding = OpenAIEmbeddings(chunk_size = 512, model = "text-embedding-3-small"),
                index_name = "Akshay_Final_DB",
                metadatas = meta_data
            )
            st.success("Excel and PDF documents loaded")
            st.session_state.clicked = True
        except Exception as e:
            st.error(f"Error creating Weaviate vector store: {e}")


# To Show the Selected Sector and Sub Sectors
text_container = st.empty()
text_container.text("Please Select Category to Continue:")

cat, sub_cat, person, pdf, excel = filter_selection_list("./Store_Local/Store_Local.xlsx")
print([cat,sub_cat,person,pdf, excel])


# QA System Section
# Show after Excel and pdf loading
with st.spinner('Retrieving Answer...'):
    # if st.session_state.clicked:
        st.subheader("Question and Response System")
        question = st.text_input("Enter your question:")
        if question:
            st.write(f"**Question: {question}**")
            new_db = WeaviateVectorStore(
                client = client,
                index_name = "Akshay_Final_DB",
                text_key = "text",
                embedding = OpenAIEmbeddings(chunk_size = 512, model = "text-embedding-3-small"),
                # attributes = [cat,sub_cat,person]
            )
            retriever = new_db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 1, 'lambda_mult': 0.75}
            )
            system_prompt = (
                #  "You are an assistant for question answering tasks."
                #  "Fetch the relevant context from the provided sheet name and table name along with their values. "
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
            st.write("**Answer:**", result["answer"])

# Import Libraries
import pandas as pd
import streamlit as st
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries for Weaviate
import weaviate
from weaviate.classes.query import Filter
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
from load_select_list import load_selection_list, filter_selection_list
from store_selected_list import store_selected_list

# Load Environment Variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI model
llm = ChatOpenAI(model="gpt-4")

# Retrieve the environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')

# Weaviate URL and API initialization
# URL = "https://h1h9i3jtngzd6tcu777sq.c0.asia-southeast1.gcp.weaviate.cloud"
URL = "https://jax9mzpmtceaypdblg5w.c0.asia-southeast1.gcp.weaviate.cloud"
# client = weaviate.Client(
#     url=URL,
#     auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
# )
client = weaviate.connect_to_wcs(
    cluster_url = URL,
    auth_credentials = weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
)

# Load the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1000)

# Session State variable for Excel load button
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

# Function to load and split PDF documents
def load_pdf_documents(file_path):
    pdf_loader = PyPDFLoader(file_path=file_path)
    pdf_pages = pdf_loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pdf_pages)
    texts = [d.page_content for d in docs]
    return texts

# Main app
st.title("Ask Your Query with QueryMate")

# Navigation
page = st.selectbox("Choose a page", ["Admin Page", "User Page"])

if page == "Admin Page":
    st.header("Admin Page")
    # Sidebar for Sector, Sub Sector, and Sales Person input
    with st.sidebar:
        sector, sub_sector, sales_person = load_selection_list("Full_Reports_Sold.xlsx")
        filtered_list = {
            "sector": sector,
            "sub_sector": sub_sector,
            "sales_person": sales_person
        }
        # Load and split PDF documents
        uploaded_file_excel = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
        uploaded_file_pdf = st.file_uploader('Upload a PDF file', type='pdf')
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
        if excel_file_path:
            try:
                excel_file = pd.ExcelFile(excel_file_path)
                sheets = excel_file.sheet_names
                list_of_items = []
                for sheet in sheets:
                    if "Units" in sheets:
                        continue
                    else:
                        data = process_sheet(excel_file_path, sheet)
                        for sheet_name, sheet_data in data.items():
                            for table_name, (values_df) in sheet_data.items():
                                list_of_items.append({
                                    "sector": sector,
                                    "Sub Sector": sub_sector,
                                    "Sales Person": sales_person,
                                    "Sheet Name": sheet_name,
                                    "Table Name": table_name,
                                    "Values Table": values_df.to_dict(),
                                    })
                        json_file_name = f"{sheet}.json"
                        directory = os.path.join("Json", uploaded_file_excel.name)
                        os.makedirs(directory, exist_ok=True)
                        file_path = os.path.join(directory, json_file_name)
                        with open(file_path, 'w') as f:
                            json.dump({"data": list_of_items}, f, indent=4)
                        with open(file_path, 'r') as file:
                            sdata = json.load(file)
                        sheet_data = sdata['data']
                        excel_texts = [str(text) for text in sheet_data]
            except FileNotFoundError:
                st.error(f"Excel file not found: {excel_file_path}")
            except Exception as e:
                st.error(f"Error loading Excel file: {e}")

    if pdf_file_path:
        try:
            pdf_texts = load_pdf_documents(pdf_file_path)
        except FileNotFoundError:
            st.error(f"PDF file not found: {pdf_file_path}")
        except Exception as e:
            st.error(f"Error loading PDF File: {e}")

    # Combine all texts from Excel and PDF files for indexing in Weaviate Vector Store
    all_texts = excel_texts + pdf_texts

    if st.button("Load Excel and PDF File"):
        view_list = store_selected_list(filtered_list, uploaded_file_excel, uploaded_file_pdf)
        try:
            db = WeaviateVectorStore.from_texts(
                texts=all_texts,
                client=client,
                embedding=OpenAIEmbeddings(chunk_size=500, model="text-embedding-ada-002"),
                index_name="Akshay_Final_DB"
            )
            st.success("Excel and PDF documents loaded")
            st.session_state.clicked = True
        except Exception as e:
            st.error(f"Error creating Weaviate vector store: {e}")

elif page == "User Page":
    st.header("User Page")
    cat, sub_cat, person, pdf, excel = filter_selection_list("./Store_Local/Store_Local.xlsx")

    # QA System Section
    st.subheader("Question and Response System")
    question = st.text_input("Enter your question:")
    if question:
        st.write(f"**Question: {question}**")
        new_db = WeaviateVectorStore(
            client=client,
            index_name="Akshay_Final_DB",
            text_key="text",
            embedding=OpenAIEmbeddings(chunk_size=500, model="text-embedding-ada-002")
        )
        retriever = new_db.as_retriever(search_type="mmr", search_kwargs={'k': 5})
        system_prompt = (
            "You are an assistant for question answering tasks "
            "Fetch the relevant context from the provided sheet name and table name along with their values."
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
        client.close()

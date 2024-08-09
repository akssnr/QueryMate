import pandas as pd
from extract_tables import extract_tables
from separate_tables import separate_tables
# from langchain_community.document_loaders import UnstructuredExcelLoader
# from langchain.text_splitter import CharacterTextSplitter

# Function for processing the excel sheets
def process_sheet(uploaded_file, selected_sheet):
    xls = pd.ExcelFile(uploaded_file)
    df = pd.read_excel(xls, sheet_name=selected_sheet)
    tables = extract_tables(df)
    clean_tables = [separate_tables(table) for table in tables]
    data = {}
    for i, (values_df, t_name) in enumerate(clean_tables):
        # table_name = f"Table {i + 1} : {t_name.iloc[0, 0]}"
        table_name = f"Table {i + 1} : {t_name.iloc[0, 5]}"
        if selected_sheet not in data:
            data[selected_sheet] = {}
        data[selected_sheet][table_name] = (values_df)
    return data


# def load_other_excel_documents(file_path):
#     excel_loader = UnstructuredExcelLoader(file_path, mode="elements")
#     excel_data = excel_loader.load_and_split()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     docs = text_splitter.split_documents(excel_data)
#     texts = [d.page_content for d in docs]
#     return texts

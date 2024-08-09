import streamlit as st
import pandas as pd
from filter_data import filter_data
import os


def load_selection_list(file_path):
    # Load data from Excel
    load_selection_list = pd.read_excel(os.path.join(file_path))
    load_selection_list = load_selection_list.iloc[:, 3:]  # Custom Code for the list retrieval
    
    # Get unique sectors
    sector_list = load_selection_list["Sector"].unique().tolist()
    
    # Display sector selection
    sector = st.selectbox("Sector", sector_list, placeholder="Select a Sector")
    sub_sector = None
    sales_person = None

    if sector:
        first_df = filter_data(load_selection_list, col_name="Sector", sector_filter=sector)
        subsec_list = first_df["Sub-sector"].unique().tolist()
        sub_sector = st.selectbox("Sub-Sector", subsec_list, placeholder="Select a Sub-Sector")
        if sub_sector:
            second_df = filter_data(first_df, col_name="Sub-sector", sector_filter=sub_sector)
            sales_person_list = second_df["Sales Person"].unique().tolist()
            sales_person = st.selectbox("Sales Person", sales_person_list, placeholder="Select a Sales Person")
    
    return sector, sub_sector, sales_person


def filter_selection_list(file_path):
    # Load data from Excel
    directory = os.path.dirname(file_path)
    
    # Check if the directory exists
    if not os.path.exists(directory):
        # If it does not exist, create it
        os.makedirs(directory)
        
    filter_selected_list = pd.read_excel(os.path.join(file_path))

    # Get unique sectors
    sector_list = filter_selected_list["Sector"].unique().tolist()

    # Display sector selection
    sector = st.selectbox("Sector", sector_list, placeholder = "Select a Sector", key = "fil_sec")
    sub_sector = None
    sales_person = None
    pdf = None
    excel = None

    if sector:
        first_df = filter_data(filter_selected_list, col_name="Sector", sector_filter=sector)
        subsec_list = first_df["Sub Sector"].unique().tolist()
        sub_sector = st.selectbox("Sub-Sector", subsec_list, placeholder="Select a Sub-Sector", key = "fil_sub_sec")
        if sub_sector:
            second_df = filter_data(first_df, col_name="Sub Sector", sector_filter=sub_sector)
            sales_person_list = second_df["Sales Person"].unique().tolist()
            sales_person = st.selectbox("Sales Person", sales_person_list, placeholder="Select a Sales Person", key = "fil_s_p")
            if sales_person:
                third_df = filter_data(second_df, col_name="Sales Person", sector_filter = sales_person)
                fourth_df = filter_data(second_df, col_name = "Sales Person", sector_filter = sales_person )
                excel_list = fourth_df['Excel File'].unique().tolist()
                pdf_list = third_df["PDF File"].unique().tolist()
                pdf = st.selectbox("PDF", pdf_list, placeholder="Select a PDF File", key = "fil_pdf")
                excel = st.selectbox("Excel", excel_list, placeholder = "Select a Excel file", key = "fil_excel")

    return sector, sub_sector, sales_person, pdf, excel

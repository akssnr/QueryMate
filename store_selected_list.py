import os
import json
import pandas as pd
import streamlit as st

def store_selected_list(filtered_list, uploaded_file_excel=None, uploaded_file_pdf=None):
    try:
        if uploaded_file_excel:
            excel = uploaded_file_excel.name
        else:
            excel = uploaded_file_excel
        if uploaded_file_pdf:
            pdf = uploaded_file_pdf.name
        else:
            pdf = uploaded_file_pdf
        directory = "Store_Local"
        file_name = "Store_Local.xlsx"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            local_data = pd.read_excel(file_path)
            data = pd.DataFrame({"Sector": filtered_list['sector'],
                                "Sub Sector": filtered_list['sub_sector'],
                                "Sales Person": filtered_list['sales_person'],
                                "Excel File": excel,
                                "PDF File": pdf
            }, index = [0])
            store_data = pd.concat([local_data, data], ignore_index=True)
        else:
            store_data = pd.DataFrame({"Sector": filtered_list['sector'],
                                        "Sub Sector": filtered_list['sub_sector'],
                                        "Sales Person": filtered_list['sales_person'],
                                        "Excel File": excel,
                                        "PDF File": pdf
            }, index=[0])
        # st.write(store_data)
        store_data.to_excel(file_path, index=False)
        print("File Loaded Successfully")
    except FileNotFoundError:
        print("File Not Found")
    except Exception as E:
        print(f"Exception Occured: {E}")

# def store_selected_list(filtered_list, uploaded_file_excel=None, uploaded_file_pdf=None):
#     try:
#         store_data = []
#         # Determine the file extension
#         # Set directory based on file extension
#         if uploaded_file_excel:
#             excel = uploaded_file_excel.name
#         else:
#             excel = uploaded_file_excel
#         if uploaded_file_pdf:
#             pdf = uploaded_file_pdf.name
#         else:
#             pdf = uploaded_file_pdf
#
#         store_data.append({
#             "sector": filtered_list['sector'],
#             "Sub Sector": filtered_list['sub_sector'],
#             "Sales Person": filtered_list['sales_person'],
#             "Excel File": excel,
#             "PDF File": pdf
#         })
#         directory = "Store Local"
#         file_name = "Store_Local.json"
#         os.makedirs(directory, exist_ok=True)
#         file_path = os.path.join(directory, file_name)
#         if os.path.exists(file_path):
#             try:
#                 with open(file_path, 'r') as f:
#                     existing_data = json.load(f)
#             except FileNotFoundError:
#                 existing_data = []
#             except ValueError:  # Handle empty JSON file
#                 print("Json is empty")
#                 existing_data = []
#         else:
#             print("File does not exist")
#             existing_data = []
#
#         existing_data.append(store_data)
#
#         with open(file_path, "w") as json_file:
#             json.dump(existing_data, json_file, indent= 4)
#
#         print("File Loaded Locally")
#     except FileNotFoundError:
#         print("File not found.")
#     except Exception as E:
#         print(f"Exception Occured: {E}")
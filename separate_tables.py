import pandas as pd

def separate_tables(table):
    # Find the index of the first entirely blank column
    blank_col_indices = table.columns[table.isna().all()].tolist()
    if blank_col_indices:
        blank_col_index = table.columns.get_loc(blank_col_indices[0])
        # print(blank_col_index)
    else:
        blank_col_index = None

    # Slicing based on the blank column index
    values_df = table.iloc[1:, 1:] if blank_col_index else table
    # Extract table name (assuming it's the first value column)
    t_name = table.iloc[:, 1:blank_col_index] if blank_col_index else table.iloc[:, 1:]

    # Clean and set index for values DataFrame
    if not values_df.empty and len(values_df) > 1:
        values_df.columns = values_df.iloc[1]  # Set column names from the second row
        values_df = values_df[1:]  # Remove the first row (header)
        values_df.set_index(values_df.columns[1], inplace=True)  # Set the first column as the new index    
    return values_df, t_name

# def separate_tables(table):
#     # Find the index of the first entirely blank column
#     blank_col_index = table.columns[table.isna().all()].tolist()
#     if blank_col_index:
#         blank_col_index = table.columns.get_loc(blank_col_index[2])
#     else:
#         blank_col_index = None

#     # Slicing based on the blank column index
#     values_df = table.iloc[:, 2:blank_col_index] if blank_col_index else table
#     # percentages_df = table.iloc[1:, blank_col_index+2:] if blank_col_index else pd.DataFrame()
#     # Extract table name (assuming it's the first value column)
#     t_name = table.iloc[:, 2:blank_col_index]

#     # Clean and set index for values DataFrame
#     if not values_df.empty:
#         values_df.columns = values_df.iloc[1]  # Set column names from second row
#         values_df = values_df[2:]  # Remove first two rows (header and potentially blank row)
#         values_df.set_index(values_df.columns[0], inplace=True)  # Set first column as new index    

#     # # Clean and set index for percentages DataFrame (if present)
#     # if not percentages_df.empty:
#     #     percentages_df.columns = percentages_df.iloc[0]  # Set column names from first row
#     #     percentages_df = percentages_df[1:]  # Remove first row (header)
#     #     percentages_df.set_index(percentages_df.columns[0], inplace=True)  # Set first column as index

#     return values_df, t_name

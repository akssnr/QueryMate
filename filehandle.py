import os

# Function to save uploaded file dynamically in appropriate folder
def save_uploaded_file(uploaded_file):
    # Determine the file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Set directory based on file extension
    if file_extension == '.pdf':
        directory = 'PDF'
    elif file_extension in ['.xls', '.xlsx']:
        directory = 'Excel'
    else:
        raise ValueError("Unsupported file type")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the file in the appropriate directory
    file_path = os.path.join(os.getcwd(), directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

# Function to load saved file locally from appropriate folder
def load_local_file(file_name):
    # Determine the file extension
    file_extension = os.path.splitext(file_name)[1].lower()
    
    # Set directory based on file extension
    if file_extension == '.pdf':
        directory = 'PDF'
    elif file_extension in ['.xls', '.xlsx']:
        directory = 'Excel'
    else:
        return None
    
    # Construct the file path
    file_path = os.path.join(os.getcwd(), directory, file_name)
    
    # Check if the file exists and return the path
    if os.path.exists(file_path):
        return file_path 
    
    return None

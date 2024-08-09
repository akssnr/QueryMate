# QueryMate

QueryMate is a powerful tool that allows users to interact with their Excel and PDF files using a question and answer system. The application leverages the capabilities of Weaviate for vector storage and retrieval and OpenAI's GPT-4o-mini for natural language processing.

## Features

- Upload and process Excel and PDF files.
- Store the processed data in a Weaviate vector store.
- Ask questions related to the content of the uploaded files and get precise answers.

## Requirements

- Python 3.10+
- Streamlit
- Pandas
- Weaviate-client
- Langchain-Community
- Langchain-Weaviate
- OpenAI

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tcw-akshay-soner/querymate.git
   cd querymate

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt

3. Set up your OpenAI API key and Weaviate credentials.

4. Usage
    Run the application:

    ```bash
    streamlit run main.py

5. Upload Files:
    Upload an Excel file and a PDF file using the provided file uploaders.

6. Select Local Files:

    If no files are uploaded, you can select from local files available in the current directory.

7. Load Files:

    Click the "Load Excel and PDF File" button to process and store the data in the Weaviate vector store.

8. Ask Questions:

    Enter a question related to the content of your uploaded files and get an answer.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

License
This project is licensed under the MIT License.

Acknowledgments
Streamlit
Weaviate
OpenAI
css

This `README.md` file provides a detailed explanation of the project, its features, requirements, installation steps, and usage instructions. It also includes a code overview and acknowledges the tools and libraries used.

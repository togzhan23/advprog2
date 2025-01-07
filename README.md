# advprog2  
Assignment 2 Advanced Programming. Teamwork by Togzhan Oral and Yelnura Akhmetova.  

# Chat with Ollama - Document Retrieval and Q&A  

## Overview  

This project integrates **Streamlit**, **ChromaDB**, and **OllamaLLM** to enable document retrieval and interactive Q&A functionality. The application supports:  
- Uploading documents to ChromaDB with embeddings generated using SentenceTransformer.  
- Querying documents for context and providing answers through Ollama.  

## Installation  

1. Clone the repository:  
    ```bash  
    git clone https://github.com/yourusername/chat-with-ollama.git  
    ```  

2. Navigate to the project directory:  
    ```bash  
    cd chat-with-ollama  
    ```  

3. Install the dependencies:  
    ```bash  
    pip install -r requirements.txt  
    ```  

4. Ensure API keys and environment variables are set up (if needed).  

## Usage  

1. Start the Streamlit application:  
    ```bash  
    streamlit run src/app.py  
    ```  

2. Use the app features through your browser:  
    - **Show Documents**: View all documents stored in ChromaDB.  
    - **Add Document**: Upload a new document as a text input or file, and store it in ChromaDB.  
    - **Ask a Question**: Query the stored documents and receive answers via Ollama.  

## Features  

### Add Documents  
- Add a document through text input or by uploading a `.txt` file.  
- Each document is embedded into a vector representation using SentenceTransformer and stored in ChromaDB.  

### Show Documents  
- View all documents currently stored in ChromaDB.  

### Query and Answer  
- Input a question to retrieve relevant documents from ChromaDB.  
- Generate an answer by querying Ollama with the retrieved document context.  

## Code Highlights  

### Document Addition  
- Handles both manual text input and `.txt` file uploads.  
- Automatically detects the encoding of uploaded files to ensure proper handling.  
- Stores embeddings in ChromaDB using `SentenceTransformer`.  

### Query with Contextual Augmentation  
- Retrieves relevant documents using a similarity query.  
- Augments the user query with retrieved document context before passing it to Ollama.  

### Error Handling  
- Logs issues with embedding generation or API queries.  
- Ensures robustness for invalid inputs or empty documents.  

## Example Workflow  

1. **Add a New Document**  
    - Use the text area or file upload option to add a document.  

2. **Ask a Question**  
    - Type a question in the input field.  
    - The app retrieves the most relevant document and generates an answer using Ollama.  

## Screenshot of the Interface  


<img width="1425" alt="image" src="https://github.com/user-attachments/assets/43d88fb4-4bc2-4910-a61f-cec28c94a680" />

<img width="1431" alt="image" src="https://github.com/user-attachments/assets/53105ed0-4334-475c-b7a4-e3c53299d5df" />

<img width="1432" alt="image" src="https://github.com/user-attachments/assets/6ea516aa-941a-49dd-8cfb-0e2e3903b950" />




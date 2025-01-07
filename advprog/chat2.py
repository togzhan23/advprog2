import streamlit as st
import logging
from langchain_ollama import OllamaLLM
import chromadb
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import chardet  # Для определения кодировки файлов


logging.basicConfig(level=logging.INFO)

chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

class EmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        vectors = self.model.encode(input)
        if len(vectors) == 0:
            raise ValueError("Empty embedding generated.")
        return vectors

embedding = EmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"}
)

def add_document_to_collection(documents, ids):
    try:
        embeddings = []
        for doc in documents:
            if not doc.strip():
                raise ValueError("Cannot add an empty or whitespace-only document.")
            embedding_vector = embedding(doc)  
            logging.info(f"Generated embedding for document '{doc}': {embedding_vector}")
            embeddings.append(embedding_vector[0])  

        embeddings = np.array(embeddings)

        logging.info(f"Embeddings shape: {embeddings.shape}")
        collection.add(documents=documents, embeddings=embeddings.tolist(), ids=ids)
    except Exception as e:
        logging.error(f"Error adding document: {e}")
        raise

def query_documents_from_chromadb(query_text, n_results=1):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results["documents"], results["metadatas"]

def query_with_ollama(prompt, model_name):
    try:
        logging.info(f"Sending prompt to Ollama with model {model_name}: {prompt}")
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(prompt)
        logging.info(f"Ollama response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error with Ollama query: {e}")
        return f"Error with Ollama API: {e}"

def retrieve_and_answer(query_text, model_name):
    retrieved_docs, _ = query_documents_from_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    return query_with_ollama(augmented_prompt, model_name)

st.title("Chat with Ollama")

model = st.sidebar.selectbox("Choose a model", ["llama3.2", "llama3.2"])

if not model:
    st.warning("Please select a model.")

menu = st.sidebar.selectbox("Choose an action", ["Show Documents in ChromaDB", "Add New Document to ChromaDB as Vector", "Ask Ollama a Question"])

if menu == "Show Documents in ChromaDB":
    st.subheader("Stored Documents in ChromaDB")
    documents = collection.get()["documents"]
    if documents:
        for i, doc in enumerate(documents, start=1):
            st.write(f"{i}. {doc}")
    else:
        st.write("No data available!")

elif menu == "Add New Document to ChromaDB as Vector":
    st.subheader("Add a New Document to ChromaDB")
    new_doc = st.text_area("Enter the new document:")
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
    
    if st.button("Add Document"):
        if uploaded_file is not None:
            try:
                # Определяем кодировку и читаем файл
                file_bytes = uploaded_file.read()
                detected_encoding = chardet.detect(file_bytes)['encoding']
                if not detected_encoding:
                    raise ValueError("Failed to detect file encoding.")
                file_content = file_bytes.decode(detected_encoding)

                doc_id = f"doc{len(collection.get()['documents']) + 1}"
                st.write(f"Adding document from file: {uploaded_file.name}")
                add_document_to_collection([file_content], [doc_id])
                st.success(f"Document added successfully with ID {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {e}")
        elif new_doc.strip(): 
            try:
                doc_id = f"doc{len(collection.get()['documents']) + 1}"
                st.write(f"Adding document: {new_doc}")
                add_document_to_collection([new_doc], [doc_id])
                st.success(f"Document added successfully with ID {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {e}")
        else:
            st.warning("Please enter a non-empty document or upload a file before adding.")

elif menu == "Ask Ollama a Question":
    query = st.text_input("Ask a question")
    if query:
        response = retrieve_and_answer(query, model)
        st.write("Response:", response)

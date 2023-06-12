import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import FAISS
import tempfile

api_key = st.text_input("Enter your API key", type = "password")

if api_key:
    llm = OpenAI(temperature=0.0, openai_api_key=api_key)
else:
    st.warning("Enter your OPENAI API-KEY. Get your OpenAI API key from [here](https://platform.openai.com/account/api-keys).\n")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file:
    bytes_data = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_file.write(bytes_data)

        temp_file_path = temp_file.name

        loader = UnstructuredPDFLoader(temp_file_path)
    
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs, embeddings)

    query = st.text_input("Ask something about your document!")

    if query:
        chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        similar_docs = db.similarity_search(query)
        response = chain({"input_documents": similar_docs, "question": query}, return_only_outputs=True)
        st.success(response)

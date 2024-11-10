__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from dotenv import load_dotenv
import time
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

start_time = time.time()
# Set page title and icon
st.set_page_config(page_title="Dr. Radha: The Agro-Homeopath", page_icon="🚀", layout="wide")

# Center the title
st.markdown("""
<style>
    #the-title {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Display the title
st.title("📚 Ask Dr. Radha - World's First AI based Agro-Homeopathy Doctor")

# Load images
human_image = "human.png"
robot_image = "bot.png"

# Load environment variables
load_dotenv()
end_time = time.time()
print(f"Loading environment variables took {end_time - start_time:.4f} seconds")

start_time = time.time()
# Set up Groq API
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), max_tokens=None, timeout=None, max_retries=2, temperature=0.5, model="llama-3.1-70b-versatile")

# Set up embeddings
embeddings = HuggingFaceEmbeddings()
end_time = time.time()
print(f"Setting up Groq LLM & Embeddings took {end_time - start_time:.4f} seconds")

# Initialize session state
if "documents" not in st.session_state:
    st.session_state["documents"] = None
if "vector_db" not in st.session_state:  
    st.session_state["vector_db"] = None
if "query" not in st.session_state:
    st.session_state["query"] = ""

def load_data():
    pdf_folder = "docs"
    loaders = [PyPDFLoader(os.path.join(pdf_folder, fn)) for fn in os.listdir(pdf_folder)]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    # Set up vector database  
    persist_directory = "db"
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

    return documents, vector_db

# Load and process PDFs
start_time = time.time()
# Load data if not already loaded
if st.session_state["documents"] is None or st.session_state["vector_db"] is None:
    with st.spinner("Loading data..."):
        documents, vector_db = load_data()
        st.session_state["documents"] = documents  
        st.session_state["vector_db"] = vector_db
else:
    documents = st.session_state["documents"]
    vector_db = st.session_state["vector_db"]

end_time = time.time()
print(f"Loading and processing PDFs & vector database took {end_time - start_time:.4f} seconds")

# Set up retrieval chain
start_time = time.time()
retriever = vector_db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Chat interface
chat_container = st.container()

# Create a form for the query input and submit button
with st.form(key='query_form'):
    query = st.text_input("Ask your question:", value="")#st.session_state["query"])
    submit_button = st.form_submit_button(label='Submit')

end_time = time.time()
print(f"Setting up retrieval chain took {end_time - start_time:.4f} seconds")
start_time = time.time()

if submit_button and query:
    with st.spinner("Generating response..."):
        result = qa({"query": query})  
    if result['result'].strip() == "":
        response = "I apologize, but I don't have enough information in the provided PDFs to answer your question."
    else:
        response = result['result']

    # Display human image and question
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image(human_image, width=80)
    with col2:
        st.markdown(f"{query}")
    # Display robot image and answer
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image(robot_image, width=80)
    with col2:
        st.markdown(f"{response}")
    
    st.markdown("---")
    
    # Clear the query input
    st.session_state["query"] = ""
#st.rerun()

end_time = time.time()
print(f"Actual query took {end_time - start_time:.4f} seconds")

# Reload data button
# if st.button("Reload Data"):
#     with st.spinner("Reloading data..."):
#         documents, vector_db = load_data()
#         st.session_state["documents"] = documents
#         st.session_state["vector_db"] = vector_db
#     st.success("Data reloaded successfully!")

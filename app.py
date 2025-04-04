import os
import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import uuid
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase and Gemini
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPLOAD_DIR = "uploads"  # Directory to store uploaded files

# Initialize services
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create uploads directory if it doesn't exist
Path(UPLOAD_DIR).mkdir(exist_ok=True)

st.title("PDF and URL chatbot")

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.current_document = None
    st.session_state.document_name = ""
    st.session_state.uploaded_files = {}

def save_uploaded_file(uploaded_file):
    """Save uploaded file to disk and return file path"""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return text.strip()

def scrape_text_from_url(url):
    """Scrape text content from a given URL."""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        return " ".join([p.text for p in soup.find_all('p')]).strip()
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def store_text_in_db(user_id, source, content, file_path=None):
    """Store document in Supabase with file reference."""
    try:
        embedding = model.encode(content).tolist()
        data = {
            "user_id": user_id,
            "source": source,
            "content": content,
            "file_path": file_path,
            "embedding": embedding,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Use upsert to update if exists, insert if new
        response = supabase.table("documents").upsert(
            data,
            on_conflict="user_id,source"  # Handle conflicts on these columns
        ).execute()
        
        if len(response.data) > 0:
            st.session_state.current_document = source
            st.session_state.document_name = source
            if file_path:
                st.session_state.uploaded_files[source] = file_path
            return True
        return False
    except Exception as e:
        st.error(f"Error storing data: {e}")
        return False

def retrieve_current_document_content(user_id):
    """Retrieve content of the currently active document."""
    try:
        if not st.session_state.current_document:
            return None, None
            
        response = supabase.table("documents") \
                         .select("id, content") \
                         .eq("user_id", user_id) \
                         .eq("source", st.session_state.current_document) \
                         .limit(1) \
                         .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]['id'], response.data[0]['content']
        return None, None
    except Exception as e:
        st.error(f"Error retrieving document content: {e}")
        return None, None

def fetch_document_id(user_id, source):
    """Fetch document_id from documents table."""
    try:
        response = supabase.table("documents") \
                         .select("id") \
                         .eq("user_id", user_id) \
                         .eq("source", source) \
                         .limit(1) \
                         .execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]["id"]
        return None
    except Exception as e:
        st.error(f"Error fetching document ID: {e}")
        return None

def query_gemini(user_id, document_name, question, context):
    """Query Gemini API and store conversation."""
    try:
        document_id = fetch_document_id(user_id, document_name)
        if not document_id:
            st.error("Document not found in database!")
            return None

        chat = genai.GenerativeModel("gemini-1.5-pro").start_chat()
        response = chat.send_message(f"Context: {context}\nQuestion: {question}\nAnswer:")
        answer = response.text.strip()

        data = {
            "user_id": user_id,
            "document_id": document_id,
            "document_name": document_name,
            "question": question,
            "answer": answer,
            "created_at": datetime.utcnow().isoformat()
        }
        supabase.table("chat_history").insert(data).execute()
        
        return answer
    except Exception as e:
        st.error(f"Error querying Gemini: {e}")
        return None

def fetch_document_history(user_id, document_name=None):
    """Fetch chat history with file paths if available."""
    try:
        if document_name:
            doc_id = fetch_document_id(user_id, document_name)
            if not doc_id:
                return []
                
            response = supabase.table("chat_history") \
                             .select("*, documents(file_path)") \
                             .eq("user_id", user_id) \
                             .eq("document_id", doc_id) \
                             .order("created_at", desc=True) \
                             .execute()
        else:
            response = supabase.table("chat_history") \
                             .select("*, documents(file_path)") \
                             .eq("user_id", user_id) \
                             .order("created_at", desc=True) \
                             .execute()
        
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return []

# stramlit UI
option = st.radio("Choose input method:", ("Upload PDF", "Enter URL"))

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        
        if st.session_state.current_document != uploaded_file.name:
            st.session_state.current_document = None
            
        file_path = save_uploaded_file(uploaded_file)
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            if store_text_in_db(st.session_state.user_id, uploaded_file.name, pdf_text, file_path):
                st.success("PDF uploaded & stored successfully!")
                st.write(f"Current document: {uploaded_file.name}")

elif option == "Enter URL":
    url = st.text_input("Enter a webpage URL")
    if st.button("Fetch & Store"):
        
        if st.session_state.current_document != url:
            st.session_state.current_document = None
            
        url_text = scrape_text_from_url(url)
        if url_text:
            if store_text_in_db(st.session_state.user_id, url, url_text):
                st.success("URL content stored successfully!")
                st.write(f"Current document: {url}")

# current Document Selection selection
st.divider()
user_docs_response = supabase.table("documents") \
                           .select("source") \
                           .eq("user_id", st.session_state.user_id) \
                           .execute()

if user_docs_response.data:
    doc_options = [doc['source'] for doc in user_docs_response.data]
    selected_doc = st.selectbox(
        "Select active document", 
        doc_options,
        index=doc_options.index(st.session_state.current_document) if st.session_state.current_document in doc_options else 0
    )
    st.session_state.current_document = selected_doc
    st.session_state.document_name = selected_doc

# Question Answering Section
st.divider()
question = st.text_input("Ask a question about your document")
if st.button("Get Answer") and question:
    if not st.session_state.current_document:
        st.error("Please upload a document or URL first!")
        st.stop()
    
    doc_id, doc_content = retrieve_current_document_content(st.session_state.user_id)
    
    if doc_id and doc_content:
        answer = query_gemini(
            st.session_state.user_id,
            st.session_state.current_document,
            question,
            doc_content
        )
        if answer:
            st.write(f"**Answer:** {answer}")
    else:
        st.warning("No content found in the selected document.")


# chat history section
st.divider()
st.header("History")
if st.checkbox("Show Chat History"):
    history = fetch_document_history(
        st.session_state.user_id,
        st.session_state.current_document if st.session_state.current_document else None
    )
    
    if history:
        current_doc = st.session_state.current_document or "All Documents"
        st.subheader(f"Chat History for: {current_doc}")
        
        for entry in history:
            with st.expander(f"Question: {entry.get('question', '')[:50]}...", expanded=False):
                doc_name = entry.get('document_name', 'Unknown')
                st.write(f"**Document:** {doc_name}")
                
                if entry.get('documents', {}).get('file_path'):
                    file_path = entry['documents']['file_path']
                    try:
                        with open(file_path, "rb") as f:
                            st.download_button(
                                label="Download Original Document",
                                data=f,
                                file_name=os.path.basename(file_path),
                                mime="application/pdf",
                                key=f"dl_{entry['id']}" 
                            )
                    except FileNotFoundError:
                        st.warning("File no longer exists")
                
                st.write(f"**Question:** {entry.get('question', '')}")
                st.write(f"**Answer:** {entry.get('answer', '')}")

    else:
        current_doc = st.session_state.current_document or "any documents"
        st.info(f"No chat history available for {current_doc}")

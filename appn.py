import streamlit as st
import pdfplumber
import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import uuid
from pgvector.psycopg2 import register_vector
import io

NEON_DB_URL = "postgresql://documents_owner:npg_kIaT4OsYeKw8@ep-empty-resonance-a5o1pqx0-pooler.us-east-2.aws.neon.tech/documents?sslmode=require"
GEMINI_API_KEY = "AIzaSyDa3giOkK9xlwIiP7h0lsuT80ekPiuX7rc"

conn = psycopg2.connect(NEON_DB_URL)
register_vector(conn)  # Enable vector support
genai.configure(api_key=GEMINI_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("PDF and URL chatbot")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.current_document = None
    st.session_state.document_name = ""

def init_db():
    with conn.cursor() as cur:#cursors
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        
        # Create tables
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            file_content BYTEA,
            embedding vector(384),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(user_id, source)
        );
        
        CREATE TABLE IF NOT EXISTS chat_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            document_id UUID REFERENCES documents(id),
            document_name TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        conn.commit()

init_db()

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return " ".join([p.get_text() for p in soup.find_all('p')]).strip()
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return None

def store_document(user_id, source, content, file_bytes=None):
    try:
        embedding = model.encode(content)
        with conn.cursor() as cur:
            cur.execute("""
            INSERT INTO documents 
            (user_id, source, content, file_content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id, source) DO UPDATE
            SET content = EXCLUDED.content,
                file_content = EXCLUDED.file_content,
                embedding = EXCLUDED.embedding
            RETURNING id
            """, 
            (user_id, source, content, file_bytes, embedding))
            doc_id = cur.fetchone()[0]
            conn.commit()
        
        st.session_state.current_document = source
        st.session_state.document_name = source
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Database error: {e}")
        return False

def get_document_content(user_id, source):
    with conn.cursor() as cur:
        cur.execute("""
        SELECT id, content, file_content FROM documents 
        WHERE user_id = %s AND source = %s
        """, (user_id, source))
        result = cur.fetchone()
        if result:
            file_content = result[2]
            if isinstance(file_content, memoryview):
                file_content = file_content.tobytes()
            return (result[0], result[1], file_content)
        return (None, None, None)

def get_document_history(user_id, document_id=None):
    with conn.cursor() as cur:
        if document_id:
            cur.execute("""
            SELECT id, document_name, question, answer, created_at
            FROM chat_history
            WHERE user_id = %s AND document_id = %s
            ORDER BY created_at DESC
            """, (user_id, document_id))
        else:
            cur.execute("""
            SELECT id, document_name, question, answer, created_at
            FROM chat_history
            WHERE user_id = %s
            ORDER BY created_at DESC
            """, (user_id,))
        return cur.fetchall()

def store_chat(user_id, document_id, document_name, question, answer):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO chat_history
        (user_id, document_id, document_name, question, answer)
        VALUES (%s, %s, %s, %s, %s)
        """, 
        (user_id, document_id, document_name, question, answer))
        conn.commit()

def query_gemini(context, question):
    try:
        chat = genai.GenerativeModel("gemini-2.0-flash").start_chat()
        response = chat.send_message(f"Context:\n{context}\n\nQuestion:\n{question}")
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return None

def document_uploader():
    option = st.radio("Input method:", ("Upload PDF", "Enter URL"))
    
    if option == "Upload PDF":
        uploaded_file = st.file_uploader("Choose PDF", type=["pdf"])
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            text = extract_text_from_pdf(io.BytesIO(file_bytes))
            if text and store_document(st.session_state.user_id, uploaded_file.name, text, file_bytes):
                st.success(f"{uploaded_file.name} processed successfully!")
    
    elif option == "Enter URL":
        url = st.text_input("Enter webpage URL")
        if st.button("Process URL"):
            text = scrape_text_from_url(url)
            if text and store_document(st.session_state.user_id, url, text):
                st.success("URL content stored!")

def document_selector():
    with conn.cursor() as cur:
        cur.execute("""
        SELECT source FROM documents 
        WHERE user_id = %s ORDER BY created_at DESC
        """, (st.session_state.user_id,))
        sources = [row[0] for row in cur.fetchall()]
    
    if sources:
        selected = st.selectbox(
            "Select document", 
            sources,
            index=0
        )
        st.session_state.current_document = selected
        st.session_state.document_name = selected

def chat_interface():
    if not st.session_state.current_document:
        st.warning("Please select a document first")
        return
    
    doc_id, content, file_bytes = get_document_content(
        st.session_state.user_id, 
        st.session_state.current_document
    )
    
    if not content:
        st.error("Document content not found")
        return
    
    question = st.text_input("Ask about the document")
    if question and st.button("Get Answer"):
        answer = query_gemini(content, question)
        if answer:
            store_chat(
                st.session_state.user_id,
                doc_id,
                st.session_state.document_name,
                question,
                answer
            )
            st.write("**Answer:**")
            st.write(answer)

def history_viewer():
    if st.checkbox("Show Chat History"):
        doc_id, _, _ = get_document_content(
            st.session_state.user_id,
            st.session_state.current_document
        ) if st.session_state.current_document else (None, None, None)
        
        history = get_document_history(st.session_state.user_id, doc_id)
        
        if history:
            for i, entry in enumerate(history):
                _, doc_name, question, answer, created_at = entry
                with st.expander(f"Q: {question[:50]}..."):
                    st.write(f"**Document:** {doc_name}")
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")
                    st.write(f"**Date:** {created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Add download button for PDFs
                    if doc_id and st.session_state.current_document.endswith('.pdf'):
                        _, _, file_data = get_document_content(
                            st.session_state.user_id,
                            st.session_state.current_document
                        )
                        if file_data:
                            if isinstance(file_data, memoryview):
                                file_data = file_data.tobytes()
                            
                            st.download_button(
                                label="Download PDF",
                                data=file_data,
                                file_name=st.session_state.current_document,
                                mime="application/pdf",
                                key=f"download_{doc_id}_{i}"  # Unique key for each button
                            )
        else:
            st.info("No chat history yet")

def main():
    document_uploader()
    document_selector()
    st.header("Document Chat")
    chat_interface() 
    st.header("History")
    history_viewer()
if __name__ == "__main__":
    main()
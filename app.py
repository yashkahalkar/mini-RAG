import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict
import tempfile 

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone as PineconeClient, ServerlessSpec

load_dotenv()

st.set_page_config(
    page_title="Mini RAG App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGApp:
    def __init__(self):
        self.index_name = "mini-rag-index-lc"
        self.setup_apis()
        self.initialize_conversation()
        if all([self.gemini_api_key, self.pinecone_api_key, self.cohere_api_key]):
            self.ensure_pinecone_index_exists()

    def initialize_conversation(self):
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

    def setup_apis(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY') or st.sidebar.text_input("Gemini API Key", type="password")
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY') or st.sidebar.text_input("Pinecone API Key", type="password")
        self.cohere_api_key = os.getenv('COHERE_API_KEY') or st.sidebar.text_input("Cohere API Key", type="password")

    def ensure_pinecone_index_exists(self):
        try:
            pc = PineconeClient(api_key=self.pinecone_api_key)
            if self.index_name not in pc.list_indexes().names():
                st.info(f"Index not found. Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(1)
        except Exception as e:
            st.error(f"❌ Error setting up Pinecone: {e}")

    def process_documents(self, uploaded_files: list, pasted_text: str):
        docs = []
        temp_dir = tempfile.gettempdir()
        
        for uploaded_file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())
            else:
                with open(temp_filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": uploaded_file.name}))

        if pasted_text:
            docs.append(Document(page_content=pasted_text, metadata={"source": "pasted_text"}))

        if not docs:
            st.warning("No documents to process.")
            return 0, 0

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=self.gemini_api_key)
        
        Pinecone.from_documents(
            documents=splits, 
            embedding=embeddings, 
            index_name=self.index_name
        )
        
        return len(docs), len(splits)

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs))
    
    def _format_history(self) -> str:
        """
        Safely formats the conversation history.
        """
        history = st.session_state.get("conversation_history", [])
        if not history:
            return ""
        
        history_parts = [f"Previous Q: {history[i]['content']}\nPrevious A: {history[i+1]['content']}" for i in range(0, len(history), 2) if i + 1 < len(history)]
        return f"\nConversation History:\n{chr(10).join(history_parts)}\n" if history_parts else ""

    @st.cache_resource
    def get_rag_chain(_self, top_k: int):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=_self.gemini_api_key)
        vectorstore = Pinecone.from_existing_index(
            index_name=_self.index_name, 
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 20})
        
        reranker = CohereRerank(
            cohere_api_key=_self.cohere_api_key,
            model="rerank-english-v3.0",
            top_n=top_k
        )
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=_self.gemini_api_key)

        prompt_template = """You are an AI assistant answering questions based on provided documents and conversation context. 

{history}
Document Context:
{context}

Current Question: {question}

Instructions:
1. Answer the current question using ONLY the provided document context.
2. Use inline citations [1], [2], etc., to reference the source documents.
3. If you cannot find the answer in the documents, state that clearly.
4. Be conversational and acknowledge if the question is a follow-up.

Answer:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)

        retrieval_and_rerank_chain = (
            RunnablePassthrough.assign(
                documents=(lambda x: x['question']) | retriever
            ).assign(
                reranked_documents=lambda x: reranker.compress_documents(
                    query=x["question"], documents=x["documents"]
                )
            )
        )
        
        rag_chain = (
            {
                "context": retrieval_and_rerank_chain | (lambda x: _self._format_docs(x["reranked_documents"])),
                "question": lambda x: x["question"],
                "history": lambda x: _self._format_history()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        chain_with_sources = RunnableParallel(
            answer=rag_chain,
            documents=retrieval_and_rerank_chain | (lambda x: x["reranked_documents"])
        )
        return chain_with_sources

    def get_answer(self, query: str, top_k: int) -> Dict:
        return self.get_rag_chain(top_k).invoke({"question": query})

    def add_to_conversation(self, role: str, content: str):
        st.session_state.conversation_history.append({'role': role, 'content': content, 'timestamp': datetime.now().isoformat()})

    def clear_conversation(self):
        st.session_state.conversation_history = []


def main():
    st.title("🔍 Mini RAG Application")
    st.markdown("Upload documents, ask questions, and get answers with citations and conversational context.")
    
    app = RAGApp()
    
    st.sidebar.title("Configuration")
    apis_ready = all([app.gemini_api_key, app.pinecone_api_key, app.cohere_api_key])
    
    if not apis_ready:
        st.warning("Please provide all API keys in the sidebar to continue.")
        return

    st.sidebar.success("✅ APIs & Pinecone configured")
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Conversation")

    if st.sidebar.button("🗑️ Clear Conversation"):
        app.clear_conversation()
        st.sidebar.success("Conversation cleared!")
        st.rerun()

    conv_count = len(st.session_state.get("conversation_history", [])) // 2
    st.sidebar.info(f"💭 Current conversation: {conv_count} exchanges")
    
    tab1, tab2, tab3 = st.tabs(["📁 Document Upload", "❓ Ask Questions", "💬 Conversation History"])
    
    with tab1:
        st.header("Upload & Process Documents")
        col1, col2 = st.columns(2)
        with col1:
            pasted_text = st.text_area("Paste Text Here:", height=300)
        with col2:
            uploaded_files = st.file_uploader(
                "Upload Files (PDF, TXT, etc.)",
                type=['pdf', 'txt', 'md', 'py', 'js', 'html', 'css'],
                accept_multiple_files=True
            )
        
        if st.button("Process Documents", type="primary") and (uploaded_files or pasted_text):
            with st.spinner("Processing documents with LangChain... This may take a moment."):
                start_time = time.time()
                doc_count, chunk_count = app.process_documents(uploaded_files, pasted_text)
                processing_time = time.time() - start_time
                st.success(f"✅ Processed {doc_count} document(s) into {chunk_count} chunks in {processing_time:.2f}s.")

    with tab2:
        st.header("Ask Questions")
        if conv_count > 0:
            st.info(f"🧠 Context-aware mode: I remember our last {conv_count} exchanges.")
        else:
            st.info("💡 Fresh start: Ask your first question!")
            
        query = st.text_input("Enter your question:", placeholder="What would you like to know?")
        top_k = st.slider("Number of source documents to use:", 1, 10, 5)
        
        if st.button("Get Answer", type="primary") and query:
            start_time = time.time()
            with st.spinner("Searching and generating answer..."):
                app.add_to_conversation("user", query)
                result = app.get_answer(query, top_k)
                answer = result['answer']
                source_docs = result['documents']
                app.add_to_conversation("assistant", answer)
            
            total_time = time.time() - start_time
            st.subheader("Answer")
            st.markdown(answer)
            st.info(f"⏱️ Response time: {total_time:.2f}s | 📄 Used {len(source_docs)} documents.")
            
            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(source_docs, 1):
                    relevance_score = doc.metadata.get('relevance_score', 'N/A')
                    score_display = f"{relevance_score:.3f}" if isinstance(relevance_score, float) else "N/A"
                    st.write(f"**[{i}] (Relevance: {score_display}) | Source: `{doc.metadata.get('source', 'N/A')}`**")
                    st.write(doc.page_content)
                    st.write("---")

    with tab3:
        st.header("💬 Conversation History")
        history = st.session_state.get("conversation_history", [])
        if not history:
            st.info("No conversation history yet.")
        else:
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    user_msg = history[i]
                    assistant_msg = history[i + 1]
                    st.chat_message("user").write(user_msg['content'])
                    st.chat_message("assistant").write(assistant_msg['content'])
                    st.markdown("---")

if __name__ == "__main__":
    main()
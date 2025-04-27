import streamlit as st
import os
import tempfile
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter
import chromadb
import base64

# Try to import MarkItDown, with fallback instructions
try:
    from markitdown import MarkItDown
except ImportError:
    st.error("Please install markitdown with: pip install 'markitdown[all]~=0.1.0a1'")

# Set page configuration
st.set_page_config(
    page_title="Professional Learning Intelligence",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recent_queries' not in st.session_state:
    st.session_state.recent_queries = []  # Changed from set to list to maintain order

# App title and description
st.title("📚 Professional Learning Intelligence! Adaptive AI using RAG")
st.markdown("Upload a PDF document and ask questions about its content.")

# Sidebar for API configuration and file upload
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", 
                           placeholder="Enter your OpenAI API Key")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        openai.api_key = api_key
    
    # Model selection
    model_option = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0
    )
    
    # File uploader
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file is not None and not st.session_state.pdf_uploaded:
        st.info("Processing PDF... Please wait.")
        
        # Create persistent ChromaDB client
        db_dir = os.path.join(tempfile.gettempdir(), "chroma_db")
        os.makedirs(db_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=db_dir)
        
        # Use a unique collection name for each upload
        collection_name = f"training_materials_{id(uploaded_file)}"
        collection = client.get_or_create_collection(name=collection_name)
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        try:
            # Convert PDF to markdown
            md = MarkItDown()
            result = md.convert(pdf_path)
            
            if not result.text_content:
                st.error("Failed to extract text from the PDF.")
                st.stop()
            
            # Split markdown by headers
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(result.text_content)
            
            # Create embeddings and vector store
            try:
                embeddings = OpenAIEmbeddings()
                st.session_state.vector_store = Chroma.from_documents(
                    documents=md_header_splits, 
                    embedding=embeddings, 
                    collection_name=collection_name,
                    persist_directory=db_dir
                )
                st.session_state.pdf_uploaded = True
                st.success("PDF processed successfully!")
                
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        
        # Clean up temporary file
        os.unlink(pdf_path)
        
    # Document info once processed
    if st.session_state.pdf_uploaded:
        st.success(f"✅ Document ready for queries")
        
        # Add reset button
        if st.button("Reset Document"):
            # Clean up ChromaDB
            if st.session_state.vector_store:
                try:
                    st.session_state.vector_store._collection.delete()
                except:
                    pass  # Ignore deletion errors
                
            st.session_state.pdf_uploaded = False
            st.session_state.vector_store = None
            st.session_state.chat_history = []
            st.session_state.recent_queries = []
            st.rerun()

# Main content area
if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
    
elif not st.session_state.pdf_uploaded:
    st.info("Please upload a PDF document in the sidebar to get started.")
    
else:
    # Function to get relevant text for a query
    def get_relevant_text(query):
        # Fixed similarity threshold value
        similarity_threshold = 0.7
        
        results = st.session_state.vector_store.similarity_search_with_score(
            query, 
            k=3  # Increased to get more potential matches
        )
        
        # Filter results by similarity score if needed
        filtered_results = [doc for doc, score in results if score <= similarity_threshold]
        
        # If no results meet the threshold, return top result anyway
        if not filtered_results and results:
            filtered_results = [results[0][0]]
            
        # Remove duplicates by content
        unique_contents = set()
        unique_results = []
        
        for doc in filtered_results:
            content = doc.page_content.strip()
            if content not in unique_contents:
                unique_contents.add(content)
                unique_results.append(doc)
        
        return "\n\n".join([doc.page_content for doc in unique_results])

    # Check if a query is similar to recent queries
    def is_similar_to_recent(query, threshold=3):
        # Simple check for very similar questions
        query_lower = query.lower().strip()
        for recent_query in st.session_state.recent_queries:
            if query_lower == recent_query.lower().strip():
                return True
            
            # Check for similarity based on word overlap
            query_words = set(query_lower.split())
            recent_words = set(recent_query.lower().split())
            common_words = query_words.intersection(recent_words)
            
            # If there's significant overlap and the query is similar in length
            if (len(common_words) >= threshold and 
                abs(len(query_words) - len(recent_words)) <= 2):
                return True
        
        return False

    # Function to generate a response
    def generate_response(query):
        # Check if this is a repeated query
        if is_similar_to_recent(query):
            return "You've asked a similar question recently. To avoid redundant answers, please try a different question or rephrase your query."
        
        # Add normalized query to recent queries list
        st.session_state.recent_queries.append(query)
        if len(st.session_state.recent_queries) > 5:  # Keep only recent 5 queries
            st.session_state.recent_queries.pop(0)  # Remove oldest query
            
        with st.spinner("Searching for relevant information..."):
            context = get_relevant_text(query)  # Retrieve relevant content

        if not context.strip():  # If no relevant data found, deny response
            return "I can only answer questions based on the uploaded training material."

        full_prompt = f"""
        Answer the following question **only using** the provided training material.
        If the answer is not found, reply with: 'I can only answer questions based on the uploaded training material.'
        Provide a concise answer without repeating information.

        Training Material:
        {context}

        Question: {query}
        """

        with st.spinner("Generating response..."):
            response = openai.chat.completions.create(
                model=model_option,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise, non-repetitive answers based only on the provided material."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.3  # Lower temperature for more consistent responses
            )

        return response.choices[0].message.content
    
    # Chat interface
    st.header("Ask about your document")
    
    # Add clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.recent_queries = []
        st.rerun()
    
    # Display chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        # Use columns for better display
        message_container = st.container()
        with message_container:
            col1, col2 = st.columns([1, 9])
            with col1:
                st.markdown("🧑")
            with col2:
                st.info(q)
                
            col1, col2 = st.columns([1, 9])
            with col1:
                st.markdown("🤖")
            with col2:
                st.success(a)
    
    # User input for new question with a form to prevent auto-rerun issues
    with st.form(key="question_form"):
        user_question = st.text_input("Ask a question about your PDF", key="user_question")
        submit_button = st.form_submit_button("Submit Question")
    
    # Process the question when submitted through the form
    if submit_button and user_question:
        # Add user question to chat history
        answer = generate_response(user_question)
        st.session_state.chat_history.append((user_question, answer))
        
        # Clear the input field by rerunning
        st.rerun()
    
    # Advanced features section
    with st.expander("Advanced Features"):
        st.subheader("Get Detailed Answer")
        
        with st.form(key="detailed_form"):
            detailed_question = st.text_area("Ask for a comprehensive explanation")
            generate_detailed = st.form_submit_button("Generate Detailed Response")
        
        if generate_detailed:
            if not detailed_question:
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Generating detailed response..."):
                    try:
                        # Function to get detailed answer
                        def get_detailed_answer(query, model=model_option, max_tokens=4000):
                            # Check if similar to recent queries
                            if is_similar_to_recent(query):
                                return "You've asked a similar question recently. To avoid redundant answers, please try a different question or rephrase your query."
                            
                            # Add to recent queries
                            st.session_state.recent_queries.append(query)
                            if len(st.session_state.recent_queries) > 5:
                                st.session_state.recent_queries.pop(0)
                                
                            context = get_relevant_text(query)
                            
                            full_prompt = f"""
                            Answer the following question with a comprehensive, detailed response based on the provided training material.
                            Include examples, explanations, and extensive information when possible.
                            If the answer is not found in the material, acknowledge that limitation.
                            Avoid repeating the same information multiple times.
                            
                            Training Material:
                            {context}
                            
                            Question: {query}
                            """
                            
                            response = openai.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that provides comprehensive, detailed answers without repetition."},
                                    {"role": "user", "content": full_prompt}
                                ],
                                max_tokens=max_tokens,
                                temperature=0.3
                            )
                            
                            return response.choices[0].message.content
                        
                        detailed_answer = get_detailed_answer(detailed_question)
                        
                        # Create a dedicated response container
                        detailed_container = st.container()
                        with detailed_container:
                            st.markdown("### Detailed Answer")
                            st.write(detailed_answer)
                            
                            # Add this to chat history too (optional)
                            st.session_state.chat_history.append((f"[Detailed] {detailed_question}", detailed_answer))
                        
                    except Exception as e:
                        st.error(f"Error generating detailed response: {str(e)}")

# Footer
st.divider()
st.markdown("*This application uses OpenAI's APIs to analyze and answer questions about your documents.*")
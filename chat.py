import json
import streamlit as st
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import docx
from PyPDF2 import PdfReader

load_dotenv()

@st.cache_resource
def init_groq():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

groq = init_groq()

def read_text_file(file_path):
    """Read text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.warning(f"Error reading {file_path}: {e}")
        return ""

def read_pdf_file(file_path):
    """Read PDF file"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.warning(f"Error reading PDF {file_path}: {e}")
        return ""

def read_docx_file(file_path):
    """Read DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.warning(f"Error reading DOCX {file_path}: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    if not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

@st.cache_data
def load_knowledge_base():
    """Load and process documents from docs folder"""
    docs_folder = Path("docs")
    knowledge_base = []
    
    if not docs_folder.exists():
        st.warning("⚠️ 'docs' folder not found. Please create a 'docs' folder and add your documents.")
        return [], None, None
    
    supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
    
    # Process all files in docs folder
    for file_path in docs_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            st.info(f"📄 Loading: {file_path.name}")
            
            if file_path.suffix.lower() == '.txt' or file_path.suffix.lower() == '.md':
                content = read_text_file(file_path)
            elif file_path.suffix.lower() == '.pdf':
                content = read_pdf_file(file_path)
            elif file_path.suffix.lower() == '.docx':
                content = read_docx_file(file_path)
            else:
                continue
            
            if content.strip():
                chunks = chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    knowledge_base.append({
                        "filename": file_path.name,
                        "chunk_id": i,
                        "content": chunk,
                        "source": f"{file_path.name} (chunk {i+1})"
                    })
    
    if not knowledge_base:
        st.warning("⚠️ No valid documents found in 'docs' folder. Supported formats: .txt, .pdf, .docx, .md")
        return [], None, None
    
    # Create TF-IDF vectorizer for document similarity
    texts = [doc["content"] for doc in knowledge_base]
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=1000,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        st.success(f"✅ Loaded {len(knowledge_base)} document chunks from {len(set([doc['filename'] for doc in knowledge_base]))} files")
        return knowledge_base, vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"Error creating TF-IDF matrix: {e}")
        return [], None, None

def retrieve_relevant_docs(query, knowledge_base, vectorizer, tfidf_matrix, top_k=3):
    """Retrieve relevant documents from knowledge base"""
    if not knowledge_base or vectorizer is None or tfidf_matrix is None:
        return []
    
    try:
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                relevant_docs.append({
                    "source": knowledge_base[idx]["source"],
                    "filename": knowledge_base[idx]["filename"],
                    "content": knowledge_base[idx]["content"],
                    "similarity": similarities[idx]
                })
        
        return relevant_docs
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

# Initialize RAG system
knowledge_base, vectorizer, tfidf_matrix = load_knowledge_base()

if 'expense_db' not in st.session_state:
    st.session_state.expense_db = []
if 'income_db' not in st.session_state:
    st.session_state.income_db = []
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            'role': 'system',
            'content': f"""You are Jarvis, a personal finance assistant with access to financial knowledge from uploaded documents. Help users with expenses, balances and financial planning.
            
            Available tools:
            1. getTotalExpense - Get total expenses
            2. addExpense - Add new expense
            3. addIncome - Add new income
            4. getMoneyBalance - Get current balance
            
            When providing financial advice, use the relevant knowledge provided in the context from the uploaded documents. Always be helpful and provide actionable advice based on the available knowledge.
            
            Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        }
    ]
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_total_expense(params):
    expense = sum(item['amount'] for item in st.session_state.expense_db)
    return f"{expense} INR"

def add_expense(params):
    name = params.get('name')
    amount = float(params.get('amount', 0))
    st.session_state.expense_db.append({'name': name, 'amount': amount})
    return 'Added to the database.'

def add_income(params):
    name = params.get('name')
    amount = float(params.get('amount', 0))
    st.session_state.income_db.append({'name': name, 'amount': amount})
    return 'Added to the income database.'

def get_money_balance(params=None):
    total_income = sum(item['amount'] for item in st.session_state.income_db)
    total_expense = sum(item['amount'] for item in st.session_state.expense_db)
    return f"{total_income - total_expense} INR"

# Tool functions mapping
tools_map = {
    'getTotalExpense': get_total_expense,
    'addExpense': add_expense,
    'addIncome': add_income,
    'getMoneyBalance': get_money_balance
}

# Tools definition for Groq API
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'getTotalExpense',
            'description': 'Get total expense from date to date.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'from': {'type': 'string', 'description': 'From date'},
                    'to': {'type': 'string', 'description': 'To date'}
                }
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'addExpense',
            'description': 'Add new expense to database.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Expense name'},
                    'amount': {'type': 'string', 'description': 'Expense amount'}
                }
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'addIncome',
            'description': 'Add new income to database.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Income name'},
                    'amount': {'type': 'string', 'description': 'Income amount'}
                }
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'getMoneyBalance',
            'description': 'Get current money balance.'
        }
    }
]

def get_assistant_response_with_rag(user_message):
    """Get response from Jarvis assistant with RAG"""
    relevant_docs = retrieve_relevant_docs(user_message, knowledge_base, vectorizer, tfidf_matrix)
    context = ""
    if relevant_docs:
        context = "\n\nRelevant Knowledge from Documents:\n"
        for doc in relevant_docs:
            context += f"From {doc['source']}:\n{doc['content']}\n\n"
        context += "Use this knowledge to provide more informed responses when relevant.\n"
    
    enhanced_message = user_message + context
    st.session_state.messages.append({'role': 'user', 'content': enhanced_message})
    
    # Get response from Groq
    while True:
        response = groq.chat.completions.create(
            messages=st.session_state.messages,
            model='llama-3.1-8b-instant',
            tools=tools,
            temperature=0.7
        )

        message = response.choices[0].message
        st.session_state.messages.append(message)

        if not message.tool_calls:
            return message.content, relevant_docs

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            result = tools_map[function_name](function_args)
            
            st.session_state.messages.append({
                'role': 'tool',
                'content': result,
                'tool_call_id': tool_call.id
            })

def main():
    st.set_page_config(page_title="Jarvis - Personal Finance Assistant with RAG", page_icon="💰")
    
    st.title("💰 Jarvis - Personal Finance Assistant")
    st.markdown("Your AI-powered personal finance helper with custom knowledge base")
    
    if knowledge_base:
        st.success(f"🧠 RAG System Active: {len(knowledge_base)} chunks from {len(set([doc['filename'] for doc in knowledge_base]))} documents loaded")
    else:
        st.warning("⚠️ RAG System Inactive: Add documents to 'docs' folder and refresh")
    
    with st.sidebar:
        st.header("📊 Financial Overview")
        
        total_income = sum(item['amount'] for item in st.session_state.income_db)
        total_expense = sum(item['amount'] for item in st.session_state.expense_db)
        balance = total_income - total_expense
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Income", f"₹{total_income}")
            st.metric("Balance", f"₹{balance}", delta=balance)
        with col2:
            st.metric("Expenses", f"₹{total_expense}")
        
        st.divider()
        
        # Recent transactions
        st.subheader("Recent Transactions")
        
        if st.session_state.income_db or st.session_state.expense_db:
            if st.session_state.income_db:
                st.write("**Recent Income:**")
                for item in st.session_state.income_db[-3:]:
                    st.write(f"+ ₹{item['amount']} - {item['name']}")
            
            if st.session_state.expense_db:
                st.write("**Recent Expenses:**")
                for item in st.session_state.expense_db[-3:]:
                    st.write(f"- ₹{item['amount']} - {item['name']}")
        else:
            st.write("No transactions yet")
        
        st.divider()
        
        st.subheader("📚 Knowledge Base")
        if knowledge_base:
            unique_files = set([doc['filename'] for doc in knowledge_base])
            st.write(f"📄 **{len(unique_files)} documents loaded:**")
            for filename in sorted(unique_files):
                chunks_count = len([doc for doc in knowledge_base if doc['filename'] == filename])
                st.write(f"• {filename} ({chunks_count} chunks)")
        else:
            st.write("No documents loaded")
            st.info("💡 Add .txt, .pdf, .docx, or .md files to the 'docs' folder")
        
        if st.button("🔄 Refresh Knowledge Base"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.expense_db = []
            st.session_state.income_db = []
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.chat_history = []
            st.rerun()
    
    st.subheader("💬 Chat with Jarvis")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, chat_item in enumerate(st.session_state.chat_history):
            if len(chat_item) == 2:
                role, message = chat_item
                relevant_docs = []
            else:
                role, message, relevant_docs = chat_item
            
            if role == "user":
                with st.chat_message("user"):
                    st.write(message)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message)
                    if relevant_docs:
                        with st.expander("📚 Knowledge Sources Used"):
                            for doc in relevant_docs:
                                st.write(f"**{doc['source']}** (relevance: {doc['similarity']:.2f})")
                                st.write(f"```\n{doc['content'][:300]}...\n```")
    
    user_input = st.chat_input("Ask Jarvis about your finances...")
    
    if user_input:
        st.session_state.chat_history.append(("user", user_input, []))
        
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get assistant response with RAG
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Jarvis is thinking and searching knowledge base..."):
                response, relevant_docs = get_assistant_response_with_rag(user_input)
            st.write(response)
            
            if relevant_docs:
                with st.expander("📚 Knowledge Sources Used"):
                    for doc in relevant_docs:
                        st.write(f"**{doc['source']}** (relevance: {doc['similarity']:.2f})")
                        st.write(f"```\n{doc['content'][:300]}...\n```")
        
        st.session_state.chat_history.append(("assistant", response, relevant_docs))
        
        st.rerun()
    
    # Quick action buttons
    st.subheader("⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💸 Add Expense", use_container_width=True):
            st.session_state.quick_action = "add_expense"
    
    with col2:
        if st.button("💰 Add Income", use_container_width=True):
            st.session_state.quick_action = "add_income"
    
    with col3:
        if st.button("📊 Check Balance", use_container_width=True):
            balance_response, relevant_docs = get_assistant_response_with_rag("What's my current balance and give me advice based on it?")
            st.session_state.chat_history.append(("user", "What's my current balance and give me advice based on it?", []))
            st.session_state.chat_history.append(("assistant", balance_response, relevant_docs))
            st.rerun()
    
    with col4:
        if st.button("💡 Get Advice", use_container_width=True):
            advice_response, relevant_docs = get_assistant_response_with_rag("Give me personalized financial advice based on my current situation and your knowledge base")
            st.session_state.chat_history.append(("user", "Give me personalized financial advice based on my current situation and your knowledge base", []))
            st.session_state.chat_history.append(("assistant", advice_response, relevant_docs))
            st.rerun()
    
    # Quick action forms
    if hasattr(st.session_state, 'quick_action'):
        if st.session_state.quick_action == "add_expense":
            with st.form("expense_form"):
                st.subheader("Add Expense")
                expense_name = st.text_input("Expense Description")
                expense_amount = st.number_input("Amount (₹)", min_value=0.01, step=0.01)
                
                if st.form_submit_button("Add Expense"):
                    user_message = f"Add expense: {expense_name} for ₹{expense_amount}"
                    response, relevant_docs = get_assistant_response_with_rag(user_message)
                    st.session_state.chat_history.append(("user", user_message, []))
                    st.session_state.chat_history.append(("assistant", response, relevant_docs))
                    del st.session_state.quick_action
                    st.rerun()
        
        elif st.session_state.quick_action == "add_income":
            with st.form("income_form"):
                st.subheader("Add Income")
                income_name = st.text_input("Income Description")
                income_amount = st.number_input("Amount (₹)", min_value=0.01, step=0.01)
                
                if st.form_submit_button("Add Income"):
                    user_message = f"Add income: {income_name} for ₹{income_amount}"
                    response, relevant_docs = get_assistant_response_with_rag(user_message)
                    st.session_state.chat_history.append(("user", user_message, []))
                    st.session_state.chat_history.append(("assistant", response, relevant_docs))
                    del st.session_state.quick_action
                    st.rerun()

if __name__ == "__main__":
    main()
# 💰 Jarvis - Personal Finance Assistant using RAG-based AI Agent

Jarvis is an intelligent personal finance assistant built with **Streamlit**, **Groq's LLM API**, and a **Retrieval-Augmented Generation (RAG)** pipeline powered by **TF-IDF**. It helps users manage their income, expenses, and savings while leveraging contextual knowledge from uploaded documents.

---

## 🚀 Features

- ✅ Add and track **expenses** and **income**
- 💹 Calculate and visualize **current financial balance**
- 🧠 Upload your own knowledge base (PDFs, DOCX, TXT, MD) for contextual financial advice
- 🔍 Uses **RAG (TF-IDF + Cosine Similarity)** for relevant document retrieval
- 🤖 Natural language conversation with Jarvis via **Groq API**
- 📚 Dynamic financial suggestions from your documents
- 📊 Sidebar with summary metrics and recent transactions
- ⚡ Quick Actions: Add expense/income, check balance, get advice

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **LLM**: [Groq API](https://groq.com/)
- **NLP**: Scikit-learn (TF-IDF), Cosine Similarity
- **RAG**: Lightweight TF-IDF document retrieval pipeline
- **Docs Support**: PDF (`PyPDF2`), DOCX (`python-docx`), TXT, MD

---


## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MOHAMMED-ANAS-IQBAL/Personal-Finance-Assistant-using-RAG-based-AI-Agent.git
cd jarvis-finance-assistant
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create `.env` file** and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

4. **Run the app:**
```bash
streamlit run chat.py
```

---

## 🧠 How RAG Works

The RAG (Retrieval-Augmented Generation) component uses:

- **TF-IDF** to vectorize your documents.
- **Cosine Similarity** to match user queries with relevant text chunks.
- The top-k relevant chunks are passed as context to the **Groq LLM**, allowing more intelligent and informed responses.

---

## 📌 Example Use Cases

- Track monthly expenses
- Get advice on saving based on uploaded budget guides
- Calculate current balance instantly
- Use custom finance documents for personalized suggestions

---

## 📄 Requirements

See `requirements.txt`, or install via:

```bash
pip install -r requirements.txt
```

---

## 🙏 Acknowledgments

Thanks to:
- [Groq](https://groq.com/)
- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [python-docx](https://python-docx.readthedocs.io/)

---

## 📬 Contact

For any queries or improvements, feel free to raise an issue or contact me.
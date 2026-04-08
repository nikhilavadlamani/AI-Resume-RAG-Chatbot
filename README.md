# AI Resume LLM RAG Chatbot

A sophisticated chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about resumes and GitHub projects. Built with LangChain, Streamlit, and HuggingFace models.

## Features

- 📄 **PDF Resume Processing**: Extract text from PDF resumes
- 🔍 **Local Vector Search**: TF-IDF based similarity search
- 🤖 **LLM Integration**: Powered by HuggingFace models via LangChain
- 💬 **Interactive Chat Interface**: Streamlit-based web application
- 📊 **GitHub Projects Integration**: Include project descriptions in responses

## Setup

### Prerequisites

- Python 3.8+
- HuggingFace API token (for LLM access)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/nikhilavadlamani/AI-Resume-RAG-Chatbot.git
cd AI-Resume-RAG-Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your HuggingFace API token
```

### Data Setup

1. Place your resume PDF in the `data/` directory as `resume.pdf`
2. Add your GitHub projects information to `data/github_projects.txt`
3. The application will automatically process these files

### Running the Application

```bash
streamlit run app.py
```

## Project Structure

```
AI-Chatbot/
├── app.py                 # Main Streamlit application
├── embeddings.py          # Text processing and similarity functions
├── extract_text.py        # Document loading and text extraction
├── vector_store.py        # Local vector storage implementation
├── rag_pipeline.py        # LangChain RAG pipeline
├── data/                  # Data directory
│   ├── resume.pdf        # Your resume (PDF format)
│   ├── github_projects.txt # GitHub projects description
│   └── README.md         # Data setup instructions
├── vector_store/          # Generated vector store (auto-created)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage

1. Start the application with `streamlit run app.py`
2. Upload or place your resume PDF in the `data/` directory
3. Add your GitHub projects to `data/github_projects.txt`
4. Click "Build/Refresh Index" in the sidebar
5. Ask questions about your resume and projects!

## Sample Questions

- "What projects has this candidate built?"
- "What technical skills are mentioned?"
- "Summarize the candidate's work experience"
- "What programming languages do they know?"

## Technologies Used

- **LangChain**: RAG pipeline and LLM integration
- **Streamlit**: Web interface
- **HuggingFace**: LLM models (Mistral-7B)
- **PyPDF2**: PDF text extraction
- **scikit-learn**: TF-IDF vectorization
- **Python**: Core language

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details
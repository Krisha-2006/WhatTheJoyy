WhatTheJoyy — AI-Powered Book Assistant

WhatTheJoyy is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about a selected book and receive relevant, context-based answers generated from its content. It combines semantic search with a local Large Language Model (LLM) to make reading more interactive and engaging.

Instead of manually searching through hundreds of pages, users can ask questions in natural language and explore the book through conversational AI.

Features

- **Ask Questions About a Book:** Interact with the selected book using natural language.
- **Context-Based Answers:** Generate responses using relevant passages retrieved from the book.
- **Chapter and Page Citations:** Help users locate the information supporting an answer.
- **Semantic Search:** Retrieve relevant passages based on meaning rather than exact keyword matches.
- **Local LLM Integration:** Use Ollama to run the language model locally.
- **PDF Ingestion:** Extract, process, and index the selected book for efficient retrieval.
- **Source-Grounded Responses:** Help reduce unsupported answers by grounding responses in retrieved book content.
- **Interactive Interface:** Access the application through a Streamlit web interface.

 Technology Stack

| Technology | Purpose |

| Python | Core application development |
| Streamlit | Interactive user interface |
| FastAPI | Backend API |
| Ollama | Local language model inference |
| Llama 3.2 (3B) | Answer generation |
| ChromaDB | Vector storage and similarity search |
| Sentence Transformers | Text embeddings |
| Hugging Face | Embedding model distribution |
| Pytest | Automated testing |

Implementation Details

BookWise follows a Retrieval-Augmented Generation architecture.

### 1. PDF Ingestion

The selected book is processed to extract its text and identify chapter boundaries. The extracted content is organized into searchable text chunks.

### 2. Text Embedding

Each text chunk is converted into a numerical vector using the `sentence-transformers/all-MiniLM-L6-v2` embedding model. These embeddings represent the semantic meaning of the text.

### 3. Vector Storage

The generated embeddings and associated text are stored in ChromaDB, allowing the application to retrieve relevant passages efficiently.

### 4. Question Processing

When a user submits a question, the application converts it into an embedding and searches the vector database for relevant passages from the selected book.

### 5. Answer Generation

The retrieved passages are supplied as context to the Llama 3.2 language model running through Ollama. The model uses this context to generate an answer relevant to the user's question.

### 6. Source Citations

The application uses chapter and PDF-page information associated with retrieved passages to help users identify where the answer comes from in the book.

### 7. Application Architecture

The project separates the main responsibilities into three components:

- **Frontend:** Streamlit provides the user-facing interface.
- **Backend:** FastAPI exposes endpoints for application functionality.
- **RAG Pipeline:** The ingestion and retrieval components process the book, search the vector index, and generate answers using the language model.

Project Structure

The project contains the following main components:

```text
BookWise/
├── api.py                  # FastAPI backend
├── app.py                  # Streamlit application
├── src/
│   └── ingestion.py        # Book ingestion and indexing
├── data/
│   ├── books/              # Selected PDF book (local)
│   └── config/             # Chapter configuration
├── chroma_db/              # Local vector database
├── runtime/                # Local runtime state and reports
├── requirements-dev.txt    # Python dependencies
├── .env.example            # Example environment configuration
├── .gitignore              # Files excluded from Git
└── README.md               # Project documentation
```

*Note: This is a simplified overview. Additional modules and files may exist in the actual project.*

## Setup and Installation

### Prerequisites

Install the following software before running BookWise:

- Python 3.11
- Git
- Ollama

Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and project folder name.

Step 2: Create a Virtual Environment

On Windows, run:

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you may need to adjust the execution policy for your current user.

Step 3: Install Dependencies

Upgrade pip and install the required packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Step 4: Install and Configure Ollama

Install Ollama from its official website:

https://ollama.com/download

Download the language model used by the application:

```powershell
ollama pull llama3.2:3b
```

Verify that the model is available:

```powershell
ollama list
```

Make sure the Ollama service is running before starting the application.

### Step 5: Configure Environment Variables

Create a local environment file from the example:

```powershell
Copy-Item .env.example .env
```

Generate a secure secret key:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Add the generated value to the `SECRET_KEY` setting in `.env`, along with any other configuration required by `.env.example`.

**Security note:** Never commit `.env` or expose its secret values publicly.

Step 6: Add the Book

Place a selectable-text PDF in the following location:

```text
data/books/selected_book.pdf
```

The PDF is used as the source material for the application's question-answering system.

Ensure that you have permission to process the book and that you comply with any applicable copyright restrictions.

### Step 7: Detect and Review Chapters

Run the chapter-detection process:

```powershell
python -m src.ingestion --detect-chapters-only
```

Review the generated chapter configuration in:

```text
data/config/chapters.json
```

Check the chapter boundaries and other relevant metadata. Set the `reviewed` field to `true` after reviewing the configuration.

Step 8: Build the Search Index

Run the ingestion process:

```powershell
python -m src.ingestion
```

This processes the selected book and builds the searchable index.

Review the generated ingestion report:

```text
runtime/reports/ingestion_report.md
```

Wait for the process to finish successfully before starting the backend.

Step 9: Start the Backend

In the first terminal, run:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

The API should now be accessible locally.

Useful endpoints:

- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

### Step 10: Start the Frontend

Open a second terminal in the project directory, activate the virtual environment if necessary, and run:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

BookWise is now ready to use locally.

Example Questions

After indexing the selected book, try questions such as:

- What is the main idea of this book?
- Explain a concept discussed in a particular chapter.
- What does the author say about a specific topic?
- Summarize the key points from a chapter.
- Which part of the book supports this explanation?

The answers depend on the content available in the selected book.

Privacy and Security

- Keep `.env` and all secret keys private.
- Do not commit virtual environments or generated runtime files.
- Keep copyrighted PDFs out of public repositories unless you have permission to distribute them.
- Do not publish local authentication databases or private user data.
- Review the deployment configuration before making the application publicly accessible.

## Future Improvements

Potential enhancements include:

- Support for multiple books.
- Book selection and library management.
- Improved conversational memory for follow-up questions.
- More detailed source previews and citation navigation.
- Additional language-model options.
- Cloud deployment with a remotely accessible backend.
- Improved document-processing support for scanned PDFs.

## Project Objective

BookWise aims to make reading more interactive by allowing users to explore a book through natural-language questions. By combining semantic retrieval with language-model generation, it provides a more convenient way to discover, understand, and revisit information contained in a book.

---

**Built with Python, FastAPI, Streamlit, ChromaDB, Sentence Transformers, and Ollama.**

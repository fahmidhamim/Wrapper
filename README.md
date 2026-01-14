# RAG (Retrieval-Augmented Generation) System

A production-ready Retrieval-Augmented Generation system with multi-format document support, speech I/O capabilities, and dual LLM support (local + remote fallback).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
  - [Windows Setup](#windows-setup)
  - [macOS Setup](#macos-setup)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Text Queries](#text-queries)
  - [Speech Input/Output](#speech-inputoutput)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Working with Git](#working-with-git)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [Contributing Guidelines](#contributing-guidelines)

---

## 📌 Overview

This project implements a Retrieval-Augmented Generation (RAG) system that:
- Loads and processes multiple document formats (PDF, Word, Excel, PowerPoint, TXT, HTML, Markdown, JSON, Images with OCR)
- Creates semantic embeddings using FAISS vector database
- Retrieves relevant documents based on user queries
- Generates intelligent responses using LLMs with fallback mechanisms
- Supports speech-to-text input and text-to-speech output
- Works with both local LLMs (Ollama) and remote LLMs (Groq API)

**Faculty Requirement Met:** Local knowledge base + RAG + Local LLM + Remote LLM fallback

---

## ✨ Features

### Document Processing
- **Multiple Formats:** PDF, Word (.docx), Excel, CSV, PowerPoint, HTML, Markdown, JSON, Images (OCR)
- **Smart Chunking:** Recursive text splitting with configurable chunk size and overlap
- **Metadata Tracking:** Source tracking for all documents

### Vector Search
- **FAISS Database:** Fast semantic search with 384-dimensional embeddings
- **SentenceTransformer:** `all-MiniLM-L6-v2` model for efficient embeddings
- **Persistent Storage:** Indexes saved locally for quick reloads

### LLM Integration
- **Groq API:** Free tier support with `llama-3.3-70b-versatile` model
- **Ollama (Local):** Optional local LLM for offline operation
- **Automatic Fallback:** Switches to remote LLM if local is unavailable

### Speech Capabilities
- **Speech-to-Text:** OpenAI Whisper for accurate transcription
- **Text-to-Speech:** pyttsx3 for natural audio output
- **Cross-Platform:** Works on Windows, macOS, and Linux

### Interactive Interface
- **Menu-Driven System:** Easy-to-use command interface
- **Multiple Query Modes:** Text and speech input options
- **Real-Time Feedback:** Progress indicators and status messages

---

## 🔧 System Requirements

### Minimum Requirements
- **Python:** 3.13.2 or later
- **RAM:** 4GB minimum (8GB+ recommended)
- **Storage:** 2GB for models and indexes
- **Internet:** Required for first-time model downloads and Groq API

### Optional (for Local LLM)
- **Ollama:** Download from https://ollama.ai
- **Storage:** 4-8GB additional for local LLM model

---

## 📥 Installation Guide

### Prerequisites for All Systems

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rkrakib11/RAG.git
   cd RAG
   ```

2. **Set Up Python 3.13.2**
   
   We use `uv` for reproducible Python version management. It handles Python version pinning automatically.

---

### Windows Setup

#### Step 1: Install `uv` Package Manager

Download and install from: https://github.com/astral-sh/uv/releases

Or if you have Rust installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install uv
```

Verify installation:
```bash
uv --version
```

#### Step 2: Install Project Dependencies

```bash
# Navigate to project directory
cd RAG

# uv will automatically install Python 3.13.2
uv sync

# Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Or on Windows CMD:
.\.venv\Scripts\activate.bat
```

#### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for Whisper API
```

**Get Groq API Key:**
1. Visit https://console.groq.com
2. Create an account or sign in
3. Copy your API key to `.env`

#### Step 4: Test Installation

```bash
python main.py
```

Type `exit` to verify the system starts correctly.

---

### macOS Setup

#### Step 1: Install `uv` Package Manager

```bash
# Using Homebrew (recommended)
brew install uv

# Or using curl
curl -LsSf https://astral-sh.uv.io/install.sh | sh

# Add to PATH if needed (add to ~/.zshrc or ~/.bash_profile)
export PATH="$HOME/.cargo/bin:$PATH"
```

Verify installation:
```bash
uv --version
```

#### Step 2: Install Project Dependencies

```bash
cd RAG

# uv will automatically install Python 3.13.2
uv sync

# Activate virtual environment
source .venv/bin/activate
```

#### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

#### Step 4: Install Optional Dependencies (Local LLM)

If you want to use Ollama for local LLM:

```bash
# Install Ollama from https://ollama.ai
brew install ollama

# Pull a model (in a separate terminal)
ollama pull mistral

# Start Ollama service (run in separate terminal)
ollama serve
```

#### Step 5: Test Installation

```bash
python main.py
```

Type `exit` to verify the system starts correctly.

---

## 🚀 Getting Started

### Initial Setup Checklist

- [ ] Clone repository
- [ ] Install `uv` and Python 3.13.2
- [ ] Run `uv sync`
- [ ] Create `.env` file with Groq API key
- [ ] Run `python main.py` to verify

### First Run

```bash
python main.py
```

You'll see the menu:
```
==================================================
RAG Search System - Interactive Mode
==================================================
Type 'exit' or 'quit' to stop
Type 'rebuild' to rebuild index from new data

Ask a question:
```

### Getting Latest Code

Before starting work each day, sync with the remote repository:

```bash
# Fetch latest changes (doesn't modify your code)
git fetch origin

# See what changed
git log origin/main --oneline -10

# Update your local code
git pull origin main

# Re-install any new dependencies
uv sync
```

---

## 📖 Usage

### Text Queries

```bash
python main.py
```

Type: `t` (for text input)

**Example:**
```
Ask a question: t
Enter your question: What is machine learning?
```

The system will:
1. Search the vector database for relevant documents
2. Retrieve top matching chunks
3. Generate an answer using the LLM
4. Display the response

### Speech Input/Output

**Prerequisites:**
- Microphone and speakers/headphones
- macOS: Grant microphone permission when prompted

```bash
python main.py
```

Type: `s` (for speech input)

**What happens:**
1. System starts recording (you'll hear a beep)
2. Ask your question out loud
3. System transcribes your speech
4. Generates answer based on documents
5. Speaks the response back to you

### Rebuild Index

After adding new documents:

```bash
python main.py
```

Type: `rebuild`

This will:
- Scan all data directories for new/modified files
- Process and chunk documents
- Generate embeddings
- Create new FAISS index
- Save for future use

---

## 📁 Project Structure

```
RAG/
├── main.py                      # Entry point - interactive menu
├── README.md                    # This file
├── pyproject.toml              # Project dependencies and configuration
├── uv.lock                      # Locked dependency versions (reproducible builds)
├── .env                         # API keys (NEVER commit this)
├── .gitignore                   # Git ignore rules
│
├── data/                        # Document storage
│   ├── pdf/                     # PDF documents
│   ├── text_files/              # Text files (.txt)
│   ├── docx/                    # Word documents
│   ├── pptx/                    # PowerPoint presentations
│   ├── xlsx/                    # Excel files
│   ├── csv/                     # CSV files
│   ├── html/                    # HTML files
│   ├── markdown/                # Markdown files
│   ├── json/                    # JSON files
│   └── images/                  # Images for OCR
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Multi-format document loader
│   ├── embedding.py             # Chunking and embeddings
│   ├── vectorstore.py           # FAISS index management
│   ├── search.py                # RAG search and generation
│   ├── speech_handler.py        # Whisper STT + pyttsx3 TTS
│   ├── llm_wrapper.py           # LLM abstraction (local/remote)
│   └── config.py                # Configuration constants
│
├── faiss_store/                 # Vector database (auto-generated)
│   ├── index.faiss              # FAISS index
│   └── metadata.pkl             # Chunk metadata
│
└── notebook/                    # Jupyter notebooks (optional)
    └── document.ipynb
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Required for Groq LLM (free tier)
GROQ_API_KEY=gsk_xxxxxxxxxxxxx

# Optional - for OpenAI Whisper API
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
```

**How to get API keys:**

1. **Groq:** https://console.groq.com
   - Sign up for free account
   - Copy API key to `.env`
   - Free tier: 30 requests/minute

2. **OpenAI (optional):** https://platform.openai.com
   - Create account and add payment
   - Generate API key
   - Used only if local Whisper is unavailable

### Embedding Configuration

Located in `src/config.py`:

```python
# Vector embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dimensional embeddings

# Chunk settings
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Character overlap between chunks

# Search settings
TOP_K = 5                # Number of chunks to retrieve

# LLM settings
DEFAULT_LLM = "groq"     # "groq" or "ollama"
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_MODEL = "mistral"
```

### Speech Settings

Located in `src/speech_handler.py`:

```python
# Whisper settings
WHISPER_MODEL = "base"    # Options: tiny, base, small, medium, large
SAMPLE_RATE = 16000       # Audio sample rate

# Text-to-speech settings
TTS_RATE = 150            # Speech speed (words per minute)
TTS_VOLUME = 0.9          # Volume level (0.0-1.0)
```

---

## 🔄 Working with Git

### Initial Clone

```bash
# Clone the repository
git clone https://github.com/rkrakib11/RAG.git
cd RAG

# Set up your local identity (one-time)
git config user.name "Your Name"
git config user.email "your.email@university.edu"
```

### Daily Workflow

#### 1. **Start of Day - Get Latest Code**

```bash
# Check current branch
git branch

# See what changed since last update
git fetch origin
git log origin/main --oneline -5

# Update your local code
git pull origin main

# Install any new dependencies
uv sync
```

#### 2. **Create a Feature Branch**

```bash
# NEVER work on main branch
# Create your own feature branch
git checkout -b feature/your-feature-name

# Examples:
git checkout -b feature/improve-embeddings
git checkout -b feature/add-image-support
git checkout -b feature/optimize-search
```

#### 3. **Make Changes**

```bash
# Edit files as needed
nano src/your_file.py

# Check what changed
git status

# See detailed changes
git diff src/your_file.py
```

#### 4. **Commit Changes**

```bash
# Stage your changes
git add src/your_file.py
# or stage everything
git add .

# Commit with descriptive message
git commit -m "Improve chunking algorithm for better semantic units"

# Check commit log
git log --oneline -3
```

#### 5. **Push to Remote**

```bash
# Push your feature branch
git push origin feature/your-feature-name

# Create Pull Request on GitHub
# Go to: https://github.com/rkrakib11/RAG
# You'll see "Create Pull Request" button
```

#### 6. **Code Review & Merge**

- Request review from team members
- Address feedback by making new commits
- Once approved, merge to main via GitHub
- Delete the feature branch

#### 7. **Back to Main**

```bash
# Switch back to main
git checkout main

# Get latest merged code
git pull origin main

# Clean up local branches
git branch -d feature/your-feature-name
```

---

## 💡 Development Workflow

### Adding New Documents

1. **Add files to data directory:**
   ```bash
   # Add PDF
   cp document.pdf data/pdf/
   
   # Add Word document
   cp report.docx data/docx/
   
   # Add text file
   cp notes.txt data/text_files/
   ```

2. **Rebuild the index:**
   ```bash
   python main.py
   # Type: rebuild
   ```

3. **Verify documents loaded:**
   - Check console output for "Loaded X documents"
   - No errors for any files
   - Proper chunk count (should increase)

### Adding New Features

#### Example: Adding a New LLM Provider

1. **Create feature branch:**
   ```bash
   git checkout -b feature/add-anthropic-claude
   ```

2. **Edit relevant files:**
   ```bash
   # Edit src/llm_wrapper.py to add Claude support
   # Edit src/config.py to add configuration
   # Edit main.py to expose new option
   ```

3. **Test locally:**
   ```bash
   python main.py
   # Test your new feature
   ```

4. **Commit and push:**
   ```bash
   git add src/llm_wrapper.py src/config.py main.py
   git commit -m "Add Anthropic Claude LLM support with auto-fallback"
   git push origin feature/add-anthropic-claude
   ```

5. **Create Pull Request on GitHub** for team review

### Handling Merge Conflicts

If multiple people edit the same file:

```bash
# Pull changes
git pull origin main

# You'll see: "CONFLICT in src/file.py"

# Open the file and look for:
# <<<<<<< HEAD       (your changes)
# your code
# =======
# their code
# >>>>>>> origin/main

# Manually resolve (keep what's needed)
# Then:
git add src/file.py
git commit -m "Resolve merge conflict in file.py"
git push origin feature/your-branch
```

---

## 🐛 Troubleshooting

### Installation Issues

**Problem:** `uv: command not found`
```bash
# Solution: Add to PATH
# macOS - add to ~/.zshrc or ~/.bash_profile
export PATH="$HOME/.cargo/bin:$PATH"

# Windows: Reinstall uv and add to system PATH
```

**Problem:** Python version mismatch
```bash
# Solution: Let uv install Python automatically
uv sync  # This installs Python 3.13.2 automatically
```

**Problem:** `ModuleNotFoundError: No module named 'XXX'`
```bash
# Solution: Reinstall dependencies
uv sync
source .venv/bin/activate  # macOS/Linux
# or
.\.venv\Scripts\activate.bat  # Windows
```

### Runtime Issues

**Problem:** `No module named 'docx'` when loading Word documents
```bash
# Solution: The python-docx package is included but verify:
uv sync
python main.py
```

**Problem:** Groq API errors
```bash
# Solution 1: Check .env file exists and has valid key
cat .env  # Check GROQ_API_KEY is present

# Solution 2: Check API rate limits (free tier: 30 requests/min)
# Solution 3: Check internet connection
ping console.groq.com
```

**Problem:** Microphone not working for speech input
```bash
# macOS: Grant microphone permission
# System Preferences > Security & Privacy > Microphone

# Windows: Check Device Manager for recording devices

# All systems: Test microphone independently
```

**Problem:** FAISS index not found
```bash
# Solution: Rebuild the index
python main.py
# Type: rebuild
```

### Performance Issues

**Problem:** Slow document loading
- Solution: Reduce number of documents temporarily for testing
- Check disk I/O speed
- Ensure sufficient RAM

**Problem:** Slow embedding generation
- Solution: Normal first time (models download)
- Subsequent runs use cached models
- Check if GPU is available (would speed up)

**Problem:** Search returning irrelevant results
- Solution: Check chunk overlap settings (src/config.py)
- Try adjusting CHUNK_SIZE
- Rebuild index with new settings

---

## 👥 Contributing Guidelines

### Code Standards

1. **Python Style:** Follow PEP 8
   ```bash
   # Format code before committing
   python -m autopep8 --in-place src/your_file.py
   ```

2. **Naming Conventions:**
   - Functions: `lowercase_with_underscores()`
   - Classes: `PascalCase`
   - Constants: `UPPERCASE_WITH_UNDERSCORES`
   - Private methods: `_leading_underscore()`

3. **Documentation:**
   ```python
   def load_documents(path: str) -> List[Document]:
       """
       Load documents from path.
       
       Args:
           path: Directory containing documents
           
       Returns:
           List of loaded Document objects
       """
       pass
   ```

4. **Error Handling:**
   ```python
   try:
       result = operation()
   except SpecificError as e:
       print(f"[ERROR] Specific error: {e}")
   except Exception as e:
       print(f"[ERROR] Unexpected error: {e}")
   ```

### Testing Before Submission

```bash
# 1. Test basic functionality
python main.py
# Type: exit

# 2. Test new features specifically
python main.py
# Type: (your feature test)

# 3. Rebuild index to ensure no breakage
python main.py
# Type: rebuild

# 4. Check for any error messages
# (All should be clean or expected)
```

### Pull Request Checklist

- [ ] Feature branch created from latest main
- [ ] Code tested locally
- [ ] No error messages in output
- [ ] Documentation updated if needed
- [ ] Commit message is descriptive
- [ ] `.env` file NOT committed
- [ ] No unnecessary files added

### Commit Message Guidelines

```bash
# Format: "[Type] Short description"

# Examples:
git commit -m "Add support for Anthropic Claude LLM"
git commit -m "Fix Word document loading with python-docx"
git commit -m "Improve embedding performance with batch processing"
git commit -m "Update README with team guidelines"

# Types:
# Add = new feature
# Fix = bug fix
# Improve = performance/code quality
# Docs = documentation
# Refactor = code restructuring
```

---

## 📚 Additional Resources

### Documentation
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Guide](https://github.com/facebookresearch/faiss)
- [Groq API Docs](https://console.groq.com/docs)
- [Ollama Documentation](https://github.com/ollama/ollama)

### Tutorials
- [RAG Overview](https://docs.llamaindex.ai/en/stable/modules/rag/)
- [Vector Databases](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)
- [Git Workflow](https://guides.github.com/introduction/flow/)

### Getting Help
- Check [Issues](https://github.com/rkrakib11/RAG/issues) on GitHub
- Ask on team Slack/Discord
- Review code of similar functions

---

## 📝 License

[Add your license here if applicable]

---

## 👨‍💼 Team

**Project Lead:** rkrakib  
**Repository:** [rkrakib11/RAG](https://github.com/rkrakib11/RAG)

---

**Last Updated:** December 11, 2025  
**Python Version:** 3.13.2+  
**Status:** Production Ready

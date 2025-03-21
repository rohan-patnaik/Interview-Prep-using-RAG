
# 📚 InterviewPrep-RAG: AI-Powered Q&A Evaluator

A user-friendly application that uses Retrieval-Augmented Generation (RAG) with generative AI to help you practice interview questions, compare your spoken answers against ideal responses, and receive targeted feedback.

---

This is how the final screen looks like to give you a visual output ![Alt text](<Screenshot (116).png>)

---

## 🚀 Features

- **Audio Recording & Transcription**  
  Record short audio clips (e.g. a 5-second snippet) using PyAudio and transcribe them using Whisper (local or via the OpenAI API).

- **Custom Reference Documents**  
  Upload trusted Q&A documents that serve as your ideal answer database for personalized feedback.

- **Semantic Answer Retrieval**  
  Automatically retrieves the most relevant ideal answer from your reference documents using embeddings and semantic search (via Pinecone).

- **AI-Powered Feedback**  
  Compares your transcribed answer to the ideal response and highlights key missing points, tailored for:
  - **Solo Mode**: Automated, detailed feedback for self-improvement.
  - **Peer Mode**: Conversation prompts for mock-interview practice.

- **Interactive Web App (Streamlit)**  
  A streamlined, web-based interface that integrates recording, transcription, retrieval, and feedback in one place.

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-link>
cd InterviewPrep-RAG
pip install -r requirements.txt
```

---

## ⚡ Quick Start

To launch the Streamlit web app:

```bash
streamlit run app.py --server.fileWatcherType none
```

---

## 📁 Project Structure

```
Interview-Prep-Using-RAG/
├── app.py                          # Streamlit app entry point
├── rag_interview.ipynb             # Notebook for development & experimentation
├── requirements.txt                # Python dependencies
├── streamlit_app/                     # Folder for reference documents
```

---

## 🌐 Tech Stack

- **Streamlit** – Interactive web interface  
- **PyAudio** – Audio recording  
- **Whisper** – Speech-to-text transcription (local & OpenAI API)  
- **Pinecone** – Semantic search and retrieval  
- **OpenAI API** – LLM for evaluation and feedback  
- **LangChain** – Text splitting and document processing


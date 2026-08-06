---
title: YouTube Video RAG AI Assistant
emoji: 🤖
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.19.0
app_file: app.py
pinned: false
---

# YouTube Video RAG AI Assistant & Chrome Extension

A full-stack Chrome Extension and FastAPI RAG (Retrieval-Augmented Generation) Agent backend that automatically detects the YouTube video you are watching, fetches its transcript, indexes it into a **FAISS** vector store using **HuggingFace Embeddings**, and allows real-time interactive QA powered by **LangChain** and **Groq Llama 3.1 8B**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Chrome Browser Extension                    │
│  ┌────────────────────────┐       ┌──────────────────────────┐  │
│  │ Active YouTube Tab     │       │ Chrome Extension UI      │  │
│  │ (watch?v=JeJ4UOUoxZc)  │──────>│ (SidePanel & Popup UI)   │  │
│  └────────────────────────┘       └─────────────┬────────────┘  │
└─────────────────────────────────────────────────│───────────────┘
                                                  │ POST /chat
                                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend Server                     │
│  • YouTubeTranscriptApi fetches transcript                       │
│  • RecursiveCharacterTextSplitter (chunk_size=500, overlap=100) │
│  • HuggingFaceEmbeddings (all-MiniLM-L6-v2)                     │
│  • FAISS In-Memory Vector Store (cached per video_id)          │
│  • ChatGroq (llama-3.1-8b-instant)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Install Backend Dependencies
Open your terminal in `d:\Gen AI Course\GenAI CampusX\Youtub Extension`:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Ensure `.env` contains your **Groq API Key**:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Start the FastAPI Server
Run the FastAPI backend server:

```bash
uvicorn app:app --reload --port 8000
```
- Server will run at: `http://localhost:8000`
- Interactive API Docs: `http://localhost:8000/docs`

---

## 🧩 Installing the Chrome Extension

1. Open Google Chrome and navigate to `chrome://extensions`.
2. Enable **Developer Mode** using the toggle switch in the top-right corner.
3. Click **Load Unpacked**.
4. Select the `extension` folder located at:
   `d:\Gen AI Course\GenAI CampusX\Youtub Extension\extension`
5. Open any YouTube video in Chrome (e.g. `https://www.youtube.com/watch?v=JeJ4UOUoxZc`).
6. Click the Extension icon in Chrome's toolbar or click **Open Side Panel Chat** to open the side-by-side AI assistant!

---

## 📂 Project Structure

```
d:\Gen AI Course\GenAI CampusX\Youtub Extension\
├── app.py                   # FastAPI Backend Server & RAG Pipeline
├── Youtub_RAG_Chatbot.py    # Original standalone RAG script
├── requirements.txt         # Python dependencies
├── .env                     # API keys configuration
├── README.md                # Documentation & Setup Guide
└── extension/               # Chrome Extension (Manifest V3)
    ├── manifest.json        # Extension configuration & permissions
    ├── sidepanel.html       # Sidepanel Chat UI layout
    ├── popup.html           # Toolbar Popup launcher UI
    ├── sidepanel.js         # Chrome extension logic & API caller
    ├── style.css            # Dark mode YouTube aesthetic styles
    └── icons/               # Extension icons (16x16, 48x48, 128x128)
```

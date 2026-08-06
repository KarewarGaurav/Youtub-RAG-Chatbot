import os
import re
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("Warning: GROQ_API_KEY is not set in environment variables.")

app = FastAPI(
    title="YouTube RAG Extension Backend Agent",
    description="FastAPI Backend serving RAG pipeline for YouTube video transcripts using LangChain & Groq LLM",
    version="1.0.0"
)

# Enable CORS for Chrome Extension support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Global in-memory cache for vector stores and metadata
# Format: { video_id: { "vector_store": FAISS, "chunks_count": int, "title": str } }
vector_store_cache: Dict[str, Dict] = {}

# Conversational Chat Checkpoint Memory per video_id
chat_histories: Dict[str, ChatMessageHistory] = {}

def get_session_history(video_id: str) -> ChatMessageHistory:
    if video_id not in chat_histories:
        chat_histories[video_id] = ChatMessageHistory()
    return chat_histories[video_id]

def format_chat_history(history: ChatMessageHistory) -> str:
    if not history.messages:
        return "No prior conversation history."
    formatted = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
    return "\n".join(formatted)

# Daily API usage stats (Groq Free Tier exact limit: 14,400 Requests/Day for llama-3.1-8b-instant)
DAILY_API_LIMIT = 14400  
daily_request_count = 0

# Shared HuggingFace Embedding Model
print("Loading HuggingFace Embedding Model (all-MiniLM-L6-v2)...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Shared LLM Instance
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature="0.2")
output_parser = StrOutputParser()

# Condense / History-Aware Query Rewriter Prompt Template
condense_prompt = PromptTemplate(
    template="""Given the following conversation history and a follow-up user question, rephrase the follow-up question into a clear, standalone search query in English that includes all necessary topic context. If it is already a complete standalone question, return it as is. Do NOT answer the question, only return the rephrased standalone search query.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Search Query:""",
    input_variables=["chat_history", "question"]
)

condense_chain = condense_prompt | llm | output_parser

# Conversational Prompt Template (RAG + Checkpoint Memory)
rag_prompt = PromptTemplate(
    template="""You are a helpful and intelligent YouTube Video AI Assistant.
Answer the user's question accurately, clearly, and in English based on the provided video transcript context AND previous conversation history.
If the provided context does not contain enough information to answer the question, state politely: "I couldn't find information about that in this video transcript."
Note: Regardless of the language of the transcript, always respond in English and translate any foreign terms accurately.

Transcript Context:
{context}

Previous Conversation History:
{chat_history}

Current Question: {question}

Answer:""",
    input_variables=["context", "chat_history", "question"]
)


def extract_video_id(url_or_id: str) -> str:
    """Extract YouTube 11-character video ID from raw input or full YouTube URL."""
    url_or_id = url_or_id.strip()
    # Check if already an 11-char ID
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id
    
    # Common YouTube URL patterns
    patterns = [
        r"(?:v=|\/embed\/|\/1\/|\/v\/|https:\/\/youtu\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
            
    return url_or_id


def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def translate_to_english_fallback(transcript_items, lang_code: str) -> str:
    """Translates non-English transcript items to English using GoogleTranslator in 1500-char batches."""
    raw_text = " ".join(item.text for item in transcript_items)
    if lang_code.startswith("en") or not raw_text.strip():
        return raw_text
        
    print(f"Translating non-English transcript ({lang_code}) to English using GoogleTranslator...")
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='en')
        
        batch_text = ""
        translated_chunks = []
        
        for item in transcript_items:
            batch_text += item.text + " "
            if len(batch_text) >= 1500:
                try:
                    translated_chunks.append(translator.translate(batch_text))
                except Exception:
                    translated_chunks.append(batch_text)
                batch_text = ""
                
        if batch_text:
            try:
                translated_chunks.append(translator.translate(batch_text))
            except Exception:
                translated_chunks.append(batch_text)
                
        translated_final = " ".join(translated_chunks)
        print(f"Translation complete! Converted {len(raw_text)} chars ({lang_code}) -> {len(translated_final)} chars (en).")
        return translated_final
    except Exception as e:
        print(f"Fallback translation error: {e}. Using raw text.")
        return raw_text


def get_or_create_vector_store(video_id: str, custom_transcript_text: Optional[str] = None) -> FAISS:
    """Fetch transcript for a video_id (guaranteeing English translation), chunk, embed, and store in FAISS index (with caching).
    Supports PROXY_URL environment variable and client-extracted transcript text fallback.
    """
    clean_id = extract_video_id(video_id)
    
    if clean_id in vector_store_cache:
        print(f"Using cached vector store for video ID: {clean_id}")
        return vector_store_cache[clean_id]["vector_store"]
    
    transcript_text = ""
    
    # 1. Use client-provided transcript text if supplied (bypasses Cloud Data Center IP blocks!)
    if custom_transcript_text and len(custom_transcript_text.strip()) > 50:
        print(f"Using client-provided transcript for video ID: {clean_id} ({len(custom_transcript_text)} chars)...")
        transcript_text = custom_transcript_text
    else:
        print(f"Fetching transcript for video ID: {clean_id}...")
        proxy_url = os.getenv("PROXY_URL")
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
        
        try:
            if proxies:
                print(f"Using configured proxy for YouTube transcript request: {proxy_url[:30]}...")
                y_api = YouTubeTranscriptApi(proxies=proxies)
            else:
                y_api = YouTubeTranscriptApi()
                
            transcript_list_obj = y_api.list(clean_id)
            
            lang_code = "en"
            # Try finding manual/generated English transcript
            try:
                english_transcript = transcript_list_obj.find_transcript(['en', 'en-US', 'en-GB'])
                fetched = english_transcript.fetch()
                transcript_text = " ".join(chunk.text for chunk in fetched)
                print(f"Loaded native English transcript for video '{clean_id}'.")
            except Exception:
                # If native English is not available, pick first transcript
                first_transcript = next(iter(transcript_list_obj))
                lang_code = first_transcript.language_code
                
                if first_transcript.is_translatable:
                    print(f"Auto-translating transcript ({lang_code}) to English via YouTube API...")
                    try:
                        translated = first_transcript.translate('en')
                        fetched = translated.fetch()
                        transcript_text = " ".join(chunk.text for chunk in fetched)
                    except Exception:
                        fetched = first_transcript.fetch()
                        transcript_text = translate_to_english_fallback(fetched, lang_code)
                else:
                    fetched = first_transcript.fetch()
                    transcript_text = translate_to_english_fallback(fetched, lang_code)
                    
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcript is disabled or unavailable for YouTube video ID '{clean_id}'."
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not fetch transcript for video ID '{clean_id}': {str(e)}"
            )
        
    if not transcript_text or len(transcript_text.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transcript content for video ID '{clean_id}' is empty."
        )
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([transcript_text])
    
    # Vector store creation
    print(f"Creating FAISS vector store with {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    
    # Cache vector store
    vector_store_cache[clean_id] = {
        "vector_store": vector_store,
        "chunks_count": len(chunks)
    }
    
    return vector_store


# Request Schemas
class ProcessVideoRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID or video URL")
    transcript_text: Optional[str] = Field(None, description="Optional raw transcript text extracted by client browser")

class ChatRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID or video URL")
    question: str = Field(..., description="User question about the video content")
    transcript_text: Optional[str] = Field(None, description="Optional raw transcript text extracted by client browser")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "YouTube RAG Extension API is running successfully!",
        "docs": "http://127.0.0.1:8000/docs",
        "health": "http://127.0.0.1:8000/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "cached_videos_count": len(vector_store_cache),
        "cached_video_ids": list(vector_store_cache.keys()),
        "api_usage": {
            "used": daily_request_count,
            "limit": DAILY_API_LIMIT,
            "remaining": max(0, DAILY_API_LIMIT - daily_request_count),
            "percentage_used": round((daily_request_count / DAILY_API_LIMIT) * 100, 1)
        }
    }


@app.post("/process")
def process_video(req: ProcessVideoRequest):
    clean_id = extract_video_id(req.video_id)
    vector_store = get_or_create_vector_store(clean_id, custom_transcript_text=req.transcript_text)
    cached_info = vector_store_cache.get(clean_id, {})
    return {
        "status": "success",
        "video_id": clean_id,
        "chunks_count": cached_info.get("chunks_count", 0),
        "message": f"Transcript indexed successfully into FAISS vector store."
    }


@app.post("/chat")
def chat_with_video(req: ChatRequest):
    global daily_request_count
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    clean_id = extract_video_id(req.video_id)
    vector_store = get_or_create_vector_store(clean_id, custom_transcript_text=req.transcript_text)
    
    # 1. Retrieve session chat history checkpoint
    session_history = get_session_history(clean_id)
    formatted_history = format_chat_history(session_history)
    
    # 2. Rephrase follow-up question into standalone query if history exists
    if session_history.messages:
        try:
            standalone_query = condense_chain.invoke({
                "chat_history": formatted_history,
                "question": req.question
            }).strip()
            print(f"Rephrased follow-up query '{req.question}' -> Standalone query: '{standalone_query}'")
        except Exception as e:
            print(f"Condense query error: {e}")
            standalone_query = req.question
    else:
        standalone_query = req.question
    
    # 3. Setup Retriever using standalone_query (Top 4 relevant chunks)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # 4. Construct LCEL Runnable Chain
    parallel_chain = RunnableParallel({
        'context': lambda x: format_retrieved_docs(retriever.invoke(standalone_query)),
        'chat_history': lambda x: formatted_history,
        'question': RunnablePassthrough()
    })
    
    main_chain = parallel_chain | rag_prompt | llm | output_parser
    
    try:
        answer = main_chain.invoke(req.question)
        
        # 5. Save original question & answer into ChatMessageHistory checkpoint
        session_history.add_user_message(req.question)
        session_history.add_ai_message(answer)
        
        daily_request_count += 1
        return {
            "status": "success",
            "video_id": clean_id,
            "question": req.question,
            "standalone_query": standalone_query,
            "answer": answer,
            "api_usage": {
                "used": daily_request_count,
                "limit": DAILY_API_LIMIT,
                "remaining": max(0, DAILY_API_LIMIT - daily_request_count),
                "percentage_used": round((daily_request_count / DAILY_API_LIMIT) * 100, 1)
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer from Groq LLM: {str(e)}"
        )


@app.post("/clear_history")
def clear_chat_history(req: ProcessVideoRequest):
    clean_id = extract_video_id(req.video_id)
    if clean_id in chat_histories:
        chat_histories[clean_id].clear()
    return {
        "status": "success",
        "video_id": clean_id,
        "message": f"Chat checkpoint memory cleared for video '{clean_id}'."
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI YouTube RAG Backend Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

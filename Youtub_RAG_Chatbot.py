from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

parser = StrOutputParser()

video_api = "JeJ4UOUoxZc"

try:
    y_api = YouTubeTranscriptApi()
    transcript_list = y_api.fetch(video_api, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)

except:
    print("Transcript Disabled")
    

splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap=100)
chunks = splitter.create_documents([transcript])


embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding)


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature="0.2")


prompt = PromptTemplate(
   template="""
    You are a Helpful Assistant.
    Answer only from the Provided transcript Context.
    If the context is insufficient, just say you don't know.
    {context}
    Question: {question}
    """,
    input_variables=["context","question"])


def format_docs(retrive_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrive_docs)
  return  context_text


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
}
)

main_chain = parallel_chain | prompt | llm | parser

question = "What is SME IPO and MainBoard IPO ?"
answer = main_chain.invoke(question)
print("\n--- Answer ---")
print(answer)
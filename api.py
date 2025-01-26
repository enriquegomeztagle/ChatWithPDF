from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/query")
async def query(request: Request):

    query_data = await request.json()

    if not query_data or "question" not in query_data:
        logger.error("Invalid input: %s", query_data)
        return Response(
            status_code=400,
            content='{"error": "Invalid input, \'question\' field is required."}',
        )

    query = query_data["question"]

    prompt_template = f"""
    Humano: Usa las siguientes piezas de contexto para proporcionar una respuesta concisa a la 
    pregunta al final. Por favor, resume con menos de 250 palabras y explicaciones detalladas. 
    Si no sabes la respuesta, solo di que no la sabes; no intentes inventarla. Solo contesta de acuerdo al siguiente fragmento de texto,
    sin agregar información adicional. Porfavor siempre contesta en español.
    Si la pregunta no está relacionada con el contexto, responde "Por favor, haga una pregunta relacionada con el contexto".
    Pregunta: {query}

    Asistente:
    """
    start_time_query = time.time()
    request_uuid = uuid4()

    response_data = qa_system.invoke(prompt_template)
    end_time_query = time.time()
    query_time_seconds = end_time_query - start_time_query
    current_datetime = datetime.now()
    timestamp = current_datetime.isoformat()
    date_field = current_datetime.strftime("%Y-%m-%d")  # Format YYYY-MM-DD
    time_field = current_datetime.strftime("%H:%M:%S")  # Format HH:MM:SS

    logger.info("Time to query: %s seconds", end_time_query - start_time_query)

    return {
        "response": response_data["result"],
        "query_time_seconds": query_time_seconds,
        "model_name": model_name,
        "embeddings_name": embeddings_name,
        "request_uuid": str(request_uuid),
        "timestamp": timestamp,
        "date": date_field,
        "time": time_field,
    }


def data_ingestion(pdf_directory="pdfs"):
    """
    Load PDF documents from the specified directory and split them into smaller chunks.
    """
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs


def initialize_qa_system():
    # Load environment variables
    load_dotenv()

    model_name = os.getenv("MODEL_NAME")
    embeddings_name = os.getenv("EMBEDDINGS_NAME")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")

    # Initialize embeddings
    embeddings = OllamaEmbeddings(
        model=embeddings_name,
        base_url=ollama_base_url,
    )

    # Load and process documents
    docs = data_ingestion()

    # Create vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Initialize LLM
    llm = OllamaLLM(model=model_name, base_url=ollama_base_url)

    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )

    return qa_chain, model_name, embeddings_name


qa_system, model_name, embeddings_name = initialize_qa_system()

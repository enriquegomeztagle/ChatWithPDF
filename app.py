import boto3
import streamlit as st

# Import Titan Embeddings Model to generate embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

# Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector Store
from langchain_community.vectorstores import FAISS

# LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)


# Function for data ingestion from PDF files
def data_ingestion():
    """
    Load PDF documents from the 'pdfs' directory and split them into smaller chunks.

    Returns:
        list: A list of documents split into smaller chunks.
    """
    loader = PyPDFDirectoryLoader("pdfs")
    documents = loader.load()

    # Using Character splitter for better results with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs


# Function to create and save the vector store
def get_vector_store(docs):
    """
    Create a FAISS vector store from the given documents and save it locally.

    Args:
        docs (list): A list of documents to be embedded and stored in the vector store.
    """
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")


# Function to create the Titan LLM
def get_titan_llm():
    """
    Initialize and return the Titan LLM model.

    Returns:
        Bedrock: The initialized Titan LLM model.
    """
    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock,
        model_kwargs={"maxTokenCount": 512},
    )
    return llm


# Function to get response from the LLM
def get_response_llm(llm, vectorstore_faiss, query):
    """
    Generate a response from the LLM based on the provided query and vector store.

    Args:
        llm (Bedrock): The language model to generate the response.
        vectorstore_faiss (FAISS): The FAISS vector store for document retrieval.
        query (str): The user's question or query.

    Returns:
        str: The generated response from the LLM.
    """

    # Prompt template for the LLM
    prompt_template = """
    Human: Use the following pieces of context to provide a concise answer to the question at the end. Please summarize with at least 250 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer["result"]


# Main function to run the Streamlit app
def main():
    """
    Main function to run the Streamlit app, allowing users to ask questions about their PDF documents
    and generate responses using the Titan LLM.
    """
    st.set_page_config(page_title="📄 AI PDF Assistant", layout="wide")

    # Title and Description Section
    st.markdown(
        """
        <h1 style='text-align: center; color: #4A90E2;'>
        📄 AI PDF Assistant
        </h1>
        <h3 style='text-align: center; color: #34495E;'>
        🤖 Unlock the power of your PDF documents with this AI-powered PDF Document Assistant 📁
        </h3>
        <p style='text-align: center; color: #7F8C8D;'>
        Seamlessly extract insights from your PDF documents using cutting-edge AI technology.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # User Interaction Section
    user_question = st.text_input(
        "🔍 Ask a question related to your PDF documents:",
        "",
        help="Type in a question to get insights from the loaded PDF documents.",
    )

    # Sidebar Management
    st.sidebar.markdown(
        """
        <h2 style='color: #4A90E2;'>Manage Vector Store</h2>
        <p style='color: #7F8C8D;'>
        Update the vector representation of your documents to enhance the AI's comprehension.
        </p>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🔄 Update Vectors"):
        with st.spinner("Updating vectors... This may take a few moments."):
            docs = data_ingestion()
            get_vector_store(docs)
            st.sidebar.success("✅ Vectors updated successfully!")

    # Model Selection
    st.sidebar.markdown(
        """
        <h2 style='color: #4A90E2;'>Select AI Model</h2>
        <p style='color: #7F8C8D;'>
        Choose an AI model to generate responses from your documents.
        </p>
        """,
        unsafe_allow_html=True,
    )
    model_choice = st.sidebar.radio("Choose a model:", ("Titan",))

    # Get Response Section
    if st.button("🤖 Get Response"):
        if user_question.strip():
            with st.spinner(f"Generating response using the {model_choice} model..."):
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True,
                )
                llm = get_titan_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.markdown(f"### 🤖 {model_choice}'s Response")
                st.markdown(response, unsafe_allow_html=True)
                st.success("Response generated successfully!")
        else:
            st.warning("⚠️ Please enter a valid question before generating a response.")


if __name__ == "__main__":
    main()

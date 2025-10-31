import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain_classic.chains import ConversationalRetrievalChain

# ✅ Load environment variables
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# 🧠 Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "doc-qa-index"

# Create Pinecone index if not exists
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=1536, metric="cosine")

# 🧾 Load Document
loader = PyPDFLoader("C:/Users/HP/Documents/Document_Q&A_System/sample1.pdf")
docs = loader.load()

# 🔍 Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 🧬 Create Embeddings using Azure OpenAI
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY
)

# 📦 Store embeddings in Pinecone
vectorstore = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

# 🤖 Create GPT-4o model with Azure
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_key=AZURE_API_KEY,
    deployment_name=DEPLOYMENT_NAME,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# 🔄 Create Retrieval-Augmented Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(), 
)

# 💬 Chat loop
chat_history = []

print("\n🧠 Document Q&A System Ready! Type 'exit' to quit.\n")
while True:
    query = input("Ask a question about your document: ")
    if query.lower() in ["exit", "quit"]:
        break
    try:
        result = qa_chain({"question": query, "chat_history": chat_history})
        print("\n🤖 Answer:", result["answer"])
        chat_history.append((query, result["answer"]))
    except Exception as e:
        print(f"⚠️ Error: {e}")

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

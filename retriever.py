# retriever.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)

# ✅ 후보군 넉넉히 뽑고(예: 30), 아래에서 필터/정렬할 거임
retriever = vectorstore.as_retriever(search_kwargs={"k": 30})

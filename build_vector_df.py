



#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
# 수정 전
#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# from langchain.docstore.document import Document
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma

# # 1. CSV 로드
# df = pd.read_csv("final_preview.csv")

# # 2. Document 생성
# documents = []
# for _, row in df.iterrows():
#     text = f"""
# 요리명: {row['menu']}
# 제목: {row['title']}
# 난이도: {row['level']}
# 조회수: {row['views']}
# """
#     documents.append(
#         Document(
#             page_content=text.strip(),
#             metadata={
#                 "menu": row["menu"],
#                 "level": row["level"],
#                 "views": int(row["views"])
#             }
#         )
#     )

# # 3. 임베딩 모델 (vectorstore.py와 동일해야 함)
# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# # 4. Chroma DB 생성 + 저장
# db = Chroma.from_documents(
#     documents=documents,
#     embedding=embedding,
#     persist_directory="./chroma_db"
# )

# db.persist()

# print(f"Vector DB built: {len(documents)} recipes")

#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------
# 수정 후
#----------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------

# build_vector_db.py
import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

CSV_PATH = "final_preview.csv"    
PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def safe_str(x):
    return "" if pd.isna(x) else str(x)

def build_documents(df: pd.DataFrame):
    docs = []
    for _, row in df.iterrows():
        # ===== Vector text (의미 검색에 유리한 필드만) =====
        page_content = (
            f"요리명: {safe_str(row['요리명'])}\n"
            f"레시피제목: {safe_str(row['레시피제목'])}\n"
            f"상황별분류: {safe_str(row['상황별분류'])}\n"
            f"조리방법: {safe_str(row['조리방법'])}\n"
            f"레시피소개: {safe_str(row['레시피소개'])}\n"
            f"재료내용: {safe_str(row['재료내용'])}\n"
        ).strip()

        # ===== Metadata (정렬/필터용) =====
        metadata = {
            "id": int(row["레시피일련번호"]) if not pd.isna(row["레시피일련번호"]) else None,
            "menu": safe_str(row["요리명"]),
            "title": safe_str(row["레시피제목"]),
            "views": int(row["조회수"]) if not pd.isna(row["조회수"]) else 0,
            "level": safe_str(row["난이도"]),
            "method": safe_str(row["조리방법"]),
            "situation": safe_str(row["상황별분류"]),
            "time": safe_str(row["조리시간"]),
            "serving": safe_str(row["인분"]),
        }

        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

def main():
    df = pd.read_csv(CSV_PATH)

 
    required = ["레시피일련번호", "레시피제목", "요리명", "조회수", "조리방법", "상황별분류",
                "레시피소개", "재료내용", "인분", "난이도", "조리시간"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    documents = build_documents(df)
    print(f"Documents ready: {len(documents)}")

    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=PERSIST_DIR
    )
    db.persist()
    print(f"Vector DB built & persisted: {PERSIST_DIR}  (N={len(documents)})")

if __name__ == "__main__":
    main()
















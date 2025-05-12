import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from datetime import datetime

# ========================
# 環境変数と設定
# ========================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
notion_token = os.getenv("NOTION_API_KEY")
notion_db_id = os.getenv("NOTION_DATABASE_ID")

# Notionクライアントの初期化
notion = NotionClient(auth=notion_token)

# ========================
# ベクトルDBとLLMの準備
# ========================
embedding = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)
llm = ChatOpenAI(api_key=openai_api_key, temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="Chat with Vector DB", layout="wide")
st.title("💬 Chat with Knowledge Base")

query = st.text_input("質問を入力してください")

if query:
    with st.spinner("回答中..."):
        result = qa_chain(query)
        answer = result["result"]
        st.markdown("### 📘 回答:")
        st.write(answer)

        sources = result["source_documents"]
        st.markdown("### 🔍 参照元:")
        for doc in sources:
            st.markdown(f"- {doc.metadata.get('source', '不明')}（長さ: {len(doc.page_content)}文字）")

        # Notionに保存
        def save_to_notion(feedback="pending"):
            notion.pages.create(
                parent={"database_id": notion_db_id},
                properties={
                    "質問": {"title": [{"text": {"content": query}}]},
                    "回答": {"rich_text": [{"text": {"content": answer}}]},
                    "評価": {"select": {"name": feedback}},
                    "日時": {"date": {"start": datetime.now().isoformat()}},
                },
            )

        col1, col2, col3 = st.columns(3)
        if col1.button("👍 Good"):
            save_to_notion("good")
            st.success("Notionに Good 評価で保存しました")

        if col2.button("👎 Bad"):
            save_to_notion("bad")
            st.warning("Notionに Bad 評価で保存しました")

        if col3.button("⏳ Pending"):
            save_to_notion("pending")
            st.info("Notionに Pending 評価で保存しました")

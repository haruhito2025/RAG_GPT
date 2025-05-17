import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from datetime import datetime
from langchain.prompts import PromptTemplate

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
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding,
    collection_metadata={"hnsw:space": "cosine"}  # コサイン類似度を使用
)
llm = ChatOpenAI(
    api_key=openai_api_key,
    temperature=0.3,
    model="gpt-3.5-turbo"  # より経済的なモデルに変更
)

# カスタムプロンプトテンプレート
template = """
あなたは正確で詳細な回答を提供するAIアシスタントです。
与えられた情報源のみに基づいて回答してください。
情報源にない内容については「該当する情報がありません」と明確に伝えてください。
質問の文脈を理解し、最も関連性の高い情報を優先して回答に含めてください。

参考情報:
{context}

質問: {question}

回答は以下の形式で作成してください：
1. 直接的な回答（簡潔かつ具体的に）
2. 参考情報からの具体的な引用（該当箇所を「」で囲む）
3. 引用元の情報（ファイル名やページ番号など）
4. 補足説明（必要な場合のみ）
5. 情報が不足している場合は、その旨を明示

回答:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_kwargs={
            "k": 12  # 検索結果の数
        }
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="Chat with Vector DB", layout="wide")
st.title("💬 Chat with Knowledge Base")

# サイドバーに設定を追加
with st.sidebar:
    st.header("設定")
    temperature = st.slider(
        "回答の創造性",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="値を大きくするとより創造的な回答になりますが、一貫性が低下する可能性があります。"
    )
    
    # 温度パラメータを動的に更新
    llm.temperature = temperature

query = st.text_input("質問を入力してください")

if query:
    with st.spinner("回答中..."):
        result = qa_chain(query)
        answer = result["result"]
        
        # 回答の表示
        st.markdown("### 📘 回答:")
        st.write(answer)

        # 参照元の表示を改善
        st.markdown("### 🔍 参照元:")
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get('source', '不明')
            title = doc.metadata.get('title', 'タイトルなし')
            page = doc.metadata.get('page', '不明')
            
            score = doc.metadata.get('score', 0)
            if score == 0 and hasattr(vectorstore, '_collection'):
                score = 0.5  # デフォルト値
            
            with st.expander(f"参照元 {i}: {title} (ページ: {page}, ファイル: {source})"):
                st.markdown(doc.page_content)
                st.caption(f"関連度スコア: {score:.2f}")
                if 'page' in doc.metadata:
                    st.caption(f"ページ番号: {doc.metadata['page']}")
                if 'title' in doc.metadata:
                    st.caption(f"ドキュメント: {doc.metadata['title']}")

        # Notionに保存
        def save_to_notion(feedback="pending"):
            try:
                notion.pages.create(
                    parent={"database_id": notion_db_id},
                    properties={
                        "質問": {
                            "title": [
                                {
                                    "text": {
                                        "content": query
                                    }
                                }
                            ]
                        },
                        "回答": {
                            "rich_text": [
                                {
                                    "text": {
                                        "content": answer
                                    }
                                }
                            ]
                        },
                        "評価": {
                            "multi_select": [
                                {
                                    "name": feedback
                                }
                            ]
                        },
                        "日時": {
                            "date": {
                                "start": datetime.now().isoformat()
                            }
                        }
                    }
                )
                return True
            except Exception as e:
                st.error(f"Notionへの保存に失敗しました: {str(e)}")
                return False

        col1, col2, col3 = st.columns(3)
        if col1.button("👍 Good"):
            if save_to_notion("good"):
                st.success("Notionに Good 評価で保存しました")

        if col2.button("👎 Bad"):
            if save_to_notion("bad"):
                st.warning("Notionに Bad 評価で保存しました")

        if col3.button("⏳ Pending"):
            if save_to_notion("pending"):
                st.info("Notionに Pending 評価で保存しました")

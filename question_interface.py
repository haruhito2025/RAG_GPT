import os
import streamlit as st
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from datetime import datetime
from langchain.prompts import PromptTemplate
import json

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

try:
    print("ベクトルDBを読み込み中...")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("ベクトルDBの読み込みが完了しました")
except Exception as e:
    st.error(f"ベクトルDBの読み込みに失敗しました: {str(e)}")
    st.error("システムを再起動してください")
    st.stop()

# LLMの初期化
try:
    llm = ChatOpenAI(
        api_key=openai_api_key,
        temperature=0.3,
        model="gpt-3.5-turbo"
    )
except Exception as e:
    st.error(f"LLMの初期化に失敗しました: {str(e)}")
    st.error("APIキーを確認してください")
    st.stop()

# 回答生成用のプロンプトテンプレート（create_retrieval_chain用に修正）
answer_template = """
あなたは正確で詳細な回答を提供するAIアシスタントです。

与えられた情報源のみに基づいて回答してください。
情報源にない内容については「該当する情報がありません」と明確に伝えてください。

参考情報:
{context}

質問: {input}

回答は以下の形式で作成してください：
1. 直接的な回答（簡潔かつ具体的に）
2. 参考情報からの具体的な引用（該当箇所を「」で囲む）
3. 補足説明（必要な場合のみ）
4. 情報が不足している場合は、その旨を明示

回答:"""

# RetrievalQAチェーンを作成
def create_qa_chain(k_docs=12):
    """QAチェーンを作成する関数"""
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.prompts import PromptTemplate

        # プロンプトテンプレートの作成
        PROMPT = PromptTemplate(
            template=answer_template,
            input_variables=["context", "input"]
        )

        # ドキュメント結合チェーンの作成
        document_chain = create_stuff_documents_chain(llm, PROMPT)
        
        # 検索チェーンの作成
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_docs})
        qa_chain = create_retrieval_chain(retriever, document_chain)
        
        return qa_chain
    except Exception as e:
        st.error(f"QAチェーンの初期化に失敗しました: {str(e)}")
        st.error("従来の方法で再試行します...")
        
        # フォールバック: 従来のRetrievalQAを使用
        try:
            PROMPT = PromptTemplate(
                template="""参考情報:
{context}

質問: {question}

回答:""",
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": k_docs}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            return qa_chain, "legacy"
        except Exception as e2:
            st.error(f"フォールバックも失敗しました: {str(e2)}")
            st.stop()

# 質問分類用のプロンプトテンプレート
classification_template = """
あなたは質問を分析し、適切な検索キーワードを提案するAIアシスタントです。
与えられた質問を分析し、以下の形式で回答してください：

1. 質問の種類（技術的質問、操作方法、トラブルシューティング、など）
2. 主要なキーワード（3-5個）
3. 検索に使用すべき具体的なフレーズ（2-3個）
4. 追加で確認すべき関連トピック

質問: {input_text}

以下のJSON形式で回答してください。カンマの位置に注意してください：
{{
    "question_type": "質問の種類",
    "keywords": ["キーワード1", "キーワード2", "キーワード3"],
    "search_phrases": ["フレーズ1", "フレーズ2"],
    "related_topics": ["関連トピック1", "関連トピック2"]
}}
"""

def classify_question(question):
    """質問を分類し、検索キーワードを生成する"""
    try:
        # 直接LLMを使用してJSONレスポンスを取得
        response = llm.invoke(
            classification_template.format(input_text=question)
        )
        # レスポンスからJSONを抽出して解析
        try:
            # 余分な空白や改行を削除してJSONを整形
            json_str = response.content.strip()
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            st.error(f"JSON解析エラー: {str(e)}")
            st.error(f"受け取ったレスポンス: {response.content}")
            # JSON解析に失敗した場合のデフォルト値
            return {
                "question_type": "一般的な質問",
                "keywords": ["一般"],
                "search_phrases": [question],
                "related_topics": []
            }
    except Exception as e:
        st.error(f"質問分類中にエラーが発生しました: {str(e)}")
        return {
            "question_type": "一般的な質問",
            "keywords": ["一般"],
            "search_phrases": [question],
            "related_topics": []
        }

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
    
    # 検索パラメータの設定を追加
    k_documents = st.slider(
        "参照ドキュメント数",
        min_value=1,
        max_value=20,
        value=12,
        step=1,
        help="検索時に参照するドキュメントの数。多すぎると処理が遅くなる可能性があります。"
    )
    
    # パラメータが変更された場合にチェーンを再作成
    if 'last_k_documents' not in st.session_state:
        st.session_state.last_k_documents = k_documents
    
    if st.session_state.last_k_documents != k_documents:
        st.session_state.last_k_documents = k_documents
        st.session_state.qa_chain = None  # チェーンをリセット

# QAチェーンの初期化（必要時のみ）
if 'qa_chain' not in st.session_state or st.session_state.qa_chain is None:
    with st.spinner("システムを初期化中..."):
        result = create_qa_chain(k_documents)
        if isinstance(result, tuple):
            st.session_state.qa_chain, chain_type = result
            st.session_state.chain_type = chain_type
        else:
            st.session_state.qa_chain = result
            st.session_state.chain_type = "modern"

# LLMの温度を動的に更新
llm.temperature = temperature

# 質問入力フィールド
query = st.text_input("質問を入力してください")

if query:
    try:
        # @queryで始まる場合はJSONとして解析
        if query.startswith("@query"):
            try:
                # @queryを除去してJSONを解析
                json_str = query[6:].strip()
                query_data = json.loads(json_str)
                # search_phrasesから最初の検索フレーズを使用
                if "search_phrases" in query_data and query_data["search_phrases"]:
                    query = query_data["search_phrases"][0]
                else:
                    st.warning("検索フレーズが見つかりません。質問をそのまま使用します。")
            except json.JSONDecodeError as e:
                st.error(f"JSONの解析に失敗しました: {str(e)}")
                st.error("通常の質問として処理を続行します")

        with st.spinner("質問を分析中..."):
            # Step 1: 質問の分類
            classification = classify_question(query)
            
            st.markdown("### 📋 質問の分析:")
            st.json(classification)
            
            # Step 2: 関連文書の検索と回答生成
            with st.spinner("回答を生成中..."):
                try:
                    # チェーンのタイプに応じて呼び出し方法を変更
                    if st.session_state.chain_type == "legacy":
                        result = st.session_state.qa_chain.invoke({"query": query})
                        answer = result["result"]
                        retrieved_documents = result.get("source_documents", [])
                    else:
                        result = st.session_state.qa_chain.invoke({"input": query})
                        answer = result["answer"]
                        retrieved_documents = result.get("context", [])
                    
                    # 回答の表示
                    st.markdown("### 📘 回答:")
                    st.write(answer)

                    # 参照元の詳細表示
                    if retrieved_documents:
                        st.markdown("### 🔍 参照元詳細:")
                        for i, doc in enumerate(retrieved_documents, 1):
                            source = doc.metadata.get('source', '不明')
                            title = doc.metadata.get('title', 'タイトルなし')
                            page = doc.metadata.get('page', '不明')
                            
                            score = doc.metadata.get('score', 0.5)
                            
                            with st.expander(f"参照元 {i}: {title} (ページ: {page}, ファイル: {source})"):
                                st.markdown(doc.page_content)
                                st.caption(f"関連度スコア: {score:.2f}")
                                if 'page' in doc.metadata:
                                    st.caption(f"ページ番号: {doc.metadata['page']}")
                                if 'title' in doc.metadata:
                                    st.caption(f"ドキュメント: {doc.metadata['title']}")
                    else:
                        st.warning("関連する参照元が見つかりませんでした")
                        
                except Exception as e:
                    st.error(f"回答生成中にエラーが発生しました: {str(e)}")
                    st.error("エラーの詳細情報:")
                    st.exception(e)
                    answer = "エラーが発生したため回答を生成できませんでした。"

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
                                },
                                "質問分類": {
                                    "rich_text": [
                                        {
                                            "text": {
                                                "content": classification["question_type"]
                                            }
                                        }
                                    ]
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
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("エラーの詳細情報:")
        st.exception(e)
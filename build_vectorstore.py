# build_vectorstore.py

import os
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ========================
# 環境変数の読み込み
# ========================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ========================
# テキスト読み込みとチャンク分割
# ========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 200から500に増加
    chunk_overlap=50,  # 20から50に増加
    length_function=len,
    separators=["\n\n", "\n", "。", "、", " ", ""]
)
all_chunks = []

print("📚 テキストファイルの読み込みを開始します...")
for file_name in os.listdir("scraped_text"):
    if file_name.endswith(".txt"):
        file_path = os.path.join("scraped_text", file_name)
        print(f"📖 ファイルを読み込み中: {file_name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                print(f"   - ファイルサイズ: {len(text)} 文字")
                chunks = text_splitter.split_text(text)
                all_chunks.extend(chunks)
                print(f"   - チャンク数: {len(chunks)}")
        except Exception as e:
            print(f"❌ エラー: {file_name} の読み込みに失敗しました: {str(e)}")

if not all_chunks:
    print("⚠️ チャンクが生成されませんでした。ファイルを確認してください。")
    exit(1)

print(f"✅ チャンク化完了。総チャンク数: {len(all_chunks)}")
print(f"📊 チャンクサイズ統計:")
chunk_sizes = [len(c) for c in all_chunks]
print(f"   - 最小: {min(chunk_sizes)} 文字")
print(f"   - 最大: {max(chunk_sizes)} 文字")
print(f"   - 平均: {sum(chunk_sizes)/len(chunk_sizes):.1f} 文字")

# ========================
# ベクトルDBの構築
# ========================
try:
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY が .env に設定されていません")

    print("🔄 Embeddings モデル初期化中...")
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

    if os.path.exists("chroma_db"):
        print("🗑️ 既存のベクトルDBを削除中...")
        shutil.rmtree("chroma_db")

    print("💾 Chroma に保存中...")
    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embedding,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    print("✅ ベクトルDBの構築が完了しました")

except Exception as e:
    print(f"❌ エラー: {str(e)}")
    exit(1)

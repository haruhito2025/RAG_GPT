import os
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from datetime import datetime

# 環境変数の読み込み
load_dotenv()
notion_token = os.getenv("NOTION_API_KEY")
notion_db_id = os.getenv("NOTION_DATABASE_ID")

def check_api_key():
    if not notion_token:
        print("❌ NOTION_API_KEYが設定されていません")
        return False
    print("✅ NOTION_API_KEYが設定されています")
    return True

def check_database_id():
    if not notion_db_id:
        print("❌ NOTION_DATABASE_IDが設定されていません")
        return False
    print("✅ NOTION_DATABASE_IDが設定されています")
    return True

def check_integration_permissions(notion):
    try:
        # インテグレーションの情報を取得
        print("\nインテグレーションの情報を取得中...")
        integration = notion.users.me()
        print(f"✅ インテグレーション名: {integration.get('name', '不明')}")
        
        # データベースの情報を取得
        print("\nデータベースの情報を取得中...")
        database = notion.databases.retrieve(database_id=notion_db_id)
        print(f"✅ データベース名: {database.get('title', [{'plain_text': '不明'}])[0]['plain_text']}")
        
        # データベースのプロパティを確認
        print("\nデータベースのプロパティを確認中...")
        required_properties = {
            "質問": "title",
            "回答": "rich_text",
            "評価": "multi_select",
            "日時": "date"
        }
        
        properties = database.get("properties", {})
        missing_properties = []
        
        for prop_name, prop_type in required_properties.items():
            if prop_name not in properties:
                missing_properties.append(f"{prop_name}（{prop_type}）")
            else:
                actual_type = properties[prop_name].get("type")
                if actual_type != prop_type:
                    print(f"⚠️ {prop_name}の型が異なります（期待: {prop_type}, 実際: {actual_type}）")
                else:
                    print(f"✅ {prop_name}（{prop_type}）")
        
        if missing_properties:
            print("\n❌ 以下のプロパティが不足しています：")
            for prop in missing_properties:
                print(f"  - {prop}")
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        if "unauthorized" in str(e).lower():
            print("\n⚠️ インテグレーションにデータベースへのアクセス権限がありません。")
            print("以下の手順で権限を付与してください：")
            print("1. Notionでデータベースを開く")
            print("2. 右上の「...」をクリック")
            print("3. 「コネクションを追加」を選択")
            print("4. インテグレーションを選択して接続")
        return False

def test_notion_connection():
    try:
        # Notionクライアントの初期化
        notion = NotionClient(auth=notion_token)
        
        # インテグレーションの権限確認
        if not check_integration_permissions(notion):
            return False
        
        # テストページの作成
        print("\nテストページを作成中...")
        new_page = notion.pages.create(
            parent={"database_id": notion_db_id},
            properties={
                "質問": {"title": [{"text": {"content": "テスト接続"}}]},
                "回答": {"rich_text": [{"text": {"content": "これは接続テストです"}}]},
                "評価": {"multi_select": [{"name": "pending"}]},
                "日時": {"date": {"start": datetime.now().isoformat()}},
            },
        )
        print("✅ テストページの作成に成功しました！")
        print(f"作成したページのID: {new_page['id']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    print("Notion接続テストを開始します...")
    print("\n=== 環境変数の確認 ===")
    api_key_ok = check_api_key()
    db_id_ok = check_database_id()
    
    if not (api_key_ok and db_id_ok):
        print("\n❌ 環境変数の設定が不完全です。.envファイルを確認してください。")
        exit(1)
    
    print("\n=== 接続テストの実行 ===")
    success = test_notion_connection()
    
    print("\n=== テスト結果 ===")
    if success:
        print("\n✅ すべてのテストが成功しました！")
    else:
        print("\n❌ テストが失敗しました。")
        print("\n確認事項：")
        print("1. Notionのインテグレーションがデータベースにアクセス権限を持っているか")
        print("2. データベースのプロパティが正しく設定されているか：")
        print("   - 質問（タイトル）")
        print("   - 回答（リッチテキスト）")
        print("   - 評価（複数選択、オプション：good, bad, pending）")
        print("   - 日時（日付）") 
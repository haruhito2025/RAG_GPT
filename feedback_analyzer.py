import os
from notion_client import Client as NotionClient
from dotenv import load_dotenv
import pandas as pd
from collections import Counter

load_dotenv()
notion_token = os.getenv("NOTION_API_KEY")
notion_db_id = os.getenv("NOTION_DATABASE_ID")

def analyze_feedback():
    """Notionからフィードバックを取得して分析"""
    notion = NotionClient(auth=notion_token)
    
    response = notion.databases.query(database_id=notion_db_id)
    
    feedback_data = []
    for page in response["results"]:
        props = page["properties"]
        question = props["質問"]["title"][0]["text"]["content"] if props["質問"]["title"] else ""
        answer = props["回答"]["rich_text"][0]["text"]["content"] if props["回答"]["rich_text"] else ""
        rating = props["評価"]["multi_select"][0]["name"] if props["評価"]["multi_select"] else "pending"
        date = props["日時"]["date"]["start"] if props["日時"]["date"] else ""
        
        feedback_data.append({
            "質問": question,
            "回答": answer,
            "評価": rating,
            "日時": date
        })
    
    df = pd.DataFrame(feedback_data)
    
    ratings = Counter(df["評価"])
    print(f"評価の分布: {dict(ratings)}")
    
    bad_questions = df[df["評価"] == "bad"]["質問"].tolist()
    print(f"悪い評価を受けた質問の例: {bad_questions[:5]}")
    
    return df

if __name__ == "__main__":
    analyze_feedback()

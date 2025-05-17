import os
import shutil
import re
from pathlib import Path
import json

def clean_ocr_text(text):
    """OCRで抽出されたテキストをクリーニング"""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('〇', '○').replace('0', '〇')
    return text

def extract_metadata(json_file):
    """JSONファイルからメタデータを抽出"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            metadata = {
                "source": os.path.basename(json_file),
                "title": os.path.basename(os.path.dirname(json_file))
            }
            
            page_match = re.search(r'page_(\d+)', str(json_file))
            if page_match:
                metadata["page"] = int(page_match.group(1))
            else:
                metadata["page"] = 0
            return metadata
    except Exception as e:
        print(f"メタデータ抽出エラー: {str(e)}")
        return {"source": os.path.basename(json_file)}

def process_ocr_files(ocr_output_dir, rag_input_dir):
    """OCR出力ファイルを処理してRAG入力ディレクトリに配置"""
    ocr_dir = Path(ocr_output_dir)
    rag_dir = Path(rag_input_dir)
    
    rag_dir.mkdir(parents=True, exist_ok=True)
    
    for txt_file in ocr_dir.glob("**/page_*.txt"):
        json_file = txt_file.with_suffix('.json')
        if json_file.exists():
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            clean_text = clean_ocr_text(text)
            
            metadata = extract_metadata(json_file)
            
            output_filename = f"{metadata['title']}_p{metadata['page']}.txt"
            output_path = rag_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            print(f"処理完了: {output_filename}")
        else:
            print(f"警告: {txt_file}に対応するJSONファイルが見つかりません")

if __name__ == "__main__":
    process_ocr_files("layoutlm_ocr_output", "scraped_text")

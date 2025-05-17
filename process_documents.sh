#!/bin/bash


PDF_FOLDER="pdf_folder"
OCR_OUTPUT="layoutlm_ocr_output"

echo "PDFからテキスト抽出を開始..."
cd ~/easyocr_layoutlmv3
python main.py

echo "OCRテキストの前処理を開始..."
cd ~/RAG_GPT
python ocr_preprocessor.py

echo "ベクトルDBの構築を開始..."
python build_vectorstore.py

echo "処理が完了しました。質問インターフェースを起動するには 'streamlit run question_interface.py' を実行してください。"

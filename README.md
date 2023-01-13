# Receipt-OCR

Japanese Receipt OCR

### 使用説明

- 精度を改善する余地はまだあるので、留意してご参考にしてください。
- 識別の結果はレシートにある全文字と特定の言葉二部分があります。※1
- 識別できるレシートの言語は日本語です。※2
- エリア調整について、以下のようにレシートを囲むのほうが識別率が高いです。

![sample](static/images/sample.jpg)

※1 取り出せる特定の言葉は「shop」「date」「time」「product」「price」「total」六種類がある。  
       識別できなかったり、誤認識したり、する場合もあります。ご注意ください。  
※2 英語はある程度識別できますが、お勧めしません。

### 開発環境

- Python
- OpenCV
- Tesseract-OCR
- Spacy
- VSCode
- Flask


from spacy import displacy
import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import string

# load NER model モデルを読み込む
model_ner = spacy.load("trainOutput/model-best/")


def cleanText(txt):

    whitespace = string.whitespace
    punctuation = "!#%&\'()*+;<=>?[\\]^`{|}~"  # "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans("", "", whitespace)
    tablePunctuation = str.maketrans("", "", punctuation)

    text = str(txt)
    # text = text.lower() ##小文字に変換する
    removeWhitespace = text.translate(tableWhitespace)
    removePunctuation = removeWhitespace.translate(tablePunctuation)

    return str(removePunctuation)


# group the labels
class groupId():
    def __init__(self):
        self.id = 0
        self.text = ""

    def getGroupID(self, text):
        if text[0] == "B":
            self.text = text
            self.id += 1
            return self.id
        else:
            if text[1:] != self.text[1:]:
                pass
            else:
                return self.id


grpId = groupId()


def getPredictions(image):
    # extract data using pytesseract  データを取り出す
    # image = cv2.resize(image, dsize=None, fx=0.2, fy=0.2)
    tessData = pytesseract.image_to_data(
        image, lang="receipt+jpn", config="--psm 6")
    tessList = list(map(lambda x: x.split("\t"), tessData.split("\n")))
    df = pd.DataFrame(tessList[1:], columns=tessList[0])
    df.dropna(inplace=True)
    df["text"] = df["text"].apply(cleanText)

    # convert data into content   取り出したデータをcontentに入れる
    df_clean = df.query(" text != '' ")
    content = " ".join([w for w in df_clean["text"]])
    # print(content)

    # get predictions from NER model 変数contentをモデルに入れる
    doc = model_ner(content)

    docJson = doc.to_json()

    doc_text = docJson["text"]

    dataframe_tokens = pd.DataFrame(docJson["tokens"])
    dataframe_tokens["tokens"] = dataframe_tokens[["start", "end"]].apply(
        # doc_text[dataframe_tokens["start"][x]:dataframe_tokens["end"][x]]
        lambda x: doc_text[x[0]:x[1]], axis=1
    )

    right_table = pd.DataFrame(docJson["ents"])[["start", "label"]]
    dataframe_tokens = dataframe_tokens.merge(
        right_table, how="left", on="start")

    dataframe_tokens.fillna("O", inplace=True)

    df_clean["end"] = df_clean["text"].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean["start"] = df_clean[["text", "end"]].apply(
        lambda x: x[1] - len(x[0]), axis=1)

    dataframe_info = df_clean.merge(
        dataframe_tokens[["start", "tokens", "label"]], how="inner", on="start")

    # bounding box
    bb_df = dataframe_info.query("label != 'O'")
    # img = image.copy()

    bb_df["group"] = bb_df["label"].apply(grpId.getGroupID)

    bb_df["label"] = bb_df["label"].apply(lambda x: x[2:])
    # bb_df.fillna("O",inplace=True)

    bb_df[["left", "top", "width", "height"]] = bb_df[[
        "left", "top", "width", "height"]].astype(float)
    bb_df[["left", "top", "width", "height"]] = bb_df[[
        "left", "top", "width", "height"]].astype(int)
    bb_df["right"] = bb_df["left"] + bb_df["width"]
    bb_df["bottom"] = bb_df["top"] + bb_df["height"]

    col_group = ["left", "top", "right", "bottom", "label", "tokens", "group"]
    group_tag_img = bb_df[col_group].groupby(by="group")

    group_tag_img

    img_tagging = group_tag_img.agg({
        "left": min,
        "top": max,
        "right": max,
        "bottom": min,
        "label": np.unique,
        "tokens": lambda x: " ".join(x)
    })

    img_tagging

    img_bb = image.copy()

    for l, t, r, b, label, token in img_tagging.values:
        label = str(label)
        cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(img_bb, label, (l, t),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    info_array = dataframe_info[["tokens", "label"]].values
    entities = dict(shop=[], date=[], time=[], product=[], price=[], total=[])
    previous = "O"
    for token, label in info_array:
        BIO_tag = label[0]
        label_tag = label[2:]
        # text = parser(token, label_tag)
        if BIO_tag in ("B", "I"):  # BI 以外
            if previous != label_tag:  # 前回のlabel_tagと違う時
                entities[label_tag].append(token)
            else:  # 前回のlabel_tagと一緒の時
                if BIO_tag == "B":
                    entities[label_tag].append(token)
                else:
                    entities[label_tag][-1] = entities[label_tag][-1] + token

        previous = label_tag

    result_text = pytesseract.image_to_string(
        image, lang="receipt+jpn", config="--psm 6")

    return img_bb, entities, result_text

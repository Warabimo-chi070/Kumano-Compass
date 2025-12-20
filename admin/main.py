from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import re
import unicodedata
from sudachipy import tokenizer, dictionary
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from supabase import create_client, Client
from typing import Optional, List, Dict
import uuid
from openai import OpenAI
import os

openai_client = OpenAI()  # OPENAI_API_KEY を環境変数から読む
USE_LOCAL_NLP = os.getenv("USE_LOCAL_NLP", "0") == "1"

# ==========================
#  Sudachi 設定
# ==========================
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.C

# ==========================
#  Supabase 設定
# ==========================
BASE_DIR = Path(__file__).resolve().parent

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_KEY が環境変数に設定されていません")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_to_supabase(
    raw_text: str,
    area: Optional[str],
    source: Optional[str],
    category: str,
    sentiment: str,
    sentiment_score: float,
    urgency: int,
    topic: str,
    topic_id: int,
    importance: float,
    age: Optional[str] = None,
    gender: Optional[str] = None,
    occupation: Optional[str] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
):
    """
    Supabase voices テーブルに 1 レコード保存
    DB カラム名は important_score で統一（API では importance_score として返却）

    ※ Supabase 側に age / gender / occupation カラムがまだ無い場合は、
       自動で除外して再試行します（移行期間用のフェイルセーフ）。
    """
    data = {
        "area": area,
        "source": source,
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "raw_text": raw_text,
        "lat": lat,
        "lng": lng,
        "category": category,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "urgency": urgency,
        "topic": topic,
        "topic_id": topic_id,
        "important_score": importance,
    }

    try:
        resp = supabase.table("voices").insert(data).execute()
    except Exception:
        # 追加カラム未作成などで insert が落ちる場合の退避
        fallback = dict(data)
        fallback.pop("age", None)
        fallback.pop("gender", None)
        fallback.pop("occupation", None)
        resp = supabase.table("voices").insert(fallback).execute()

    print("Supabase Response:", resp)
    return resp

# ==========================
#  gpt-4o-mini 呼び出し
# ==========================

def extract_report_body(text: str) -> str:
    """
    念のため、プロンプトの指示文っぽいものをフィルタする保険ロジック。
    GPT-4o-mini はかなり従順なので、基本はそのまま返ってくる想定。
    """
    if not text:
        return ""

    lines = text.splitlines()
    filtered = []
    for line in lines:
        s = line.strip()
        if not s:
            filtered.append(line)
            continue
        # 明らかに「AIへの説明」っぽい行は弾く
        if s.startswith("あなたは、") or s.startswith("あなたは日本の"):
            continue
        if s.startswith("【データ一覧】") or s.startswith("【対象の声】"):
            # 冒頭の説明をゴッソリ載せたくないならここで消す
            continue
        if s.startswith("次の観点で") or s.startswith("上記の情報をもとに"):
            continue
        filtered.append(line)

    result = "\n".join(filtered).strip()
    return result if result else text.strip()


def generate_with_llm(core_prompt: str, max_output_tokens: int = 800) -> str:
    """
    OpenAI gpt-4o-mini を使って日本語テキストを生成する。
    - core_prompt: 指示文＋データ（画面には出さない）
    - 戻り値: 画面に表示するレポート本文だけ
    """
    try:
        resp = openai_client.responses.create(
            model="gpt-4o-mini",
            input=core_prompt,
            max_output_tokens=max_output_tokens,
        )
        # responses API の標準的な取り出し方（SDK バージョンに応じて調整が必要な場合あり）
        raw_text = resp.output[0].content[0].text
        return extract_report_body(raw_text)
    except Exception as e:
        print("OpenAI error:", e)
        return f"AIレポート生成中にエラーが発生しました。（ネットワークや API キーの設定をご確認ください）\n詳細: {e}"


# ==========================
#  カテゴリ辞書 / ルール
# ==========================
CATEGORY_RULES = {
    "交通": [
        "バス", "本数", "減便", "乗り合い", "デマンド",
        "道路", "渋滞", "危険", "交通", "インフラ", "移動",
    ],
    "医療": [
        "病院", "診療所", "医者", "救急", "薬局",
        "通院", "医療", "往診",
    ],
    "高齢化・福祉": [
        "高齢", "年寄り", "介護", "独居", "見守り",
        "買い物弱者", "福祉", "老人", "要支援",
    ],
    "子育て・教育": [
        "保育園", "幼稚園", "小学校", "中学校", "高校",
        "子ども", "児童", "生徒", "部活", "遊び場",
    ],
    "観光": [
        "観光", "滞在", "宿泊", "民泊", "旅行",
        "花火", "祭り", "虫送り", "ホタル", "景勝地", "名所",
    ],
    "商店街・地域経済": [
        "商店街", "店", "買い物", "空き店舗", "雇用",
        "仕事", "地元企業", "経済", "産業",
    ],
    "自然環境": [
        "海", "川", "山", "森", "自然", "生態系",
        "景観", "環境", "動植物",
    ],
    "防災・津波": [
        "津波", "南海トラフ", "避難", "洪水", "浸水",
        "災害", "危険箇所", "ハザード", "地震",
    ],
    "行政サービス": [
        "市役所", "行政", "制度", "申請", "広報",
        "サービス", "支援", "相談",
    ],
}

NEGATIVE_PATTERNS = [
    r"混み",
    r"渋滞",
    r"時間.?が?.*かか",
    r"分かりづら", r"わかりづら",
    r"分かりにく", r"わかりにく",
    r"不便",
    r"困っ",
    r"危険",
    r"怖い",
    r"不足", r"足りない",
    r"問題", r"課題",
    r"迷惑",
    r"大変",
    r"しんど",
    r"獣害",
    r"空き家",
    r"人が減ってきて",
    r"若い人はみんな市外に出てしまって",
    r"担い手がいない",
    r"このままでは.*(続けられない|やっていけない)",
    r"どうにもならない",
]

POSITIVE_PATTERNS = [
    r"うれしい", r"嬉しい",
    r"ありがたい",
    r"感謝している",
    r"助かる",
    r"良かった",
    r"きれい",  # 海はきれい、ホタルがきれい、など
    r"にぎやかで嬉しい",
]

def adjust_sentiment_by_negative_words(sentiment: str, sentiment_score: float, raw_text: str):
    """
    文章中のネガ/ポジ表現と「〜けど／ですが」の構造を見て、
    モデルの出したラベルと極性スコアを微調整する。

    スコアの意味:
      0.0   = 強いネガ
      0.5   = 中立
      1.0   = 強いポジ
    """
    if not raw_text:
        return sentiment, sentiment_score

    text = raw_text

    def count_hits(patterns, t: str) -> int:
        return sum(1 for p in patterns if re.search(p, t))

    neg_hits = count_hits(NEGATIVE_PATTERNS, text)
    pos_hits = count_hits(POSITIVE_PATTERNS, text)
    has_but = bool(re.search(r"(けど|けれど|ですが|けども|のですが)", text))

    # ネガ語もポジ語も無いなら何もしない
    if neg_hits == 0 and pos_hits == 0:
        return sentiment, sentiment_score

    # 1) 「ポジ → けど／ですが → 課題」の典型パターンはネガ寄せ
    if has_but and neg_hits >= 1 and sentiment in ("positive", "neutral"):
        sentiment = "negative"
        if neg_hits >= 2:
            # 強めのネガ
            sentiment_score = min(sentiment_score, 0.2)
        else:
            # 軽め〜中程度のネガ
            sentiment_score = min(sentiment_score, 0.35)

    # 2) けど構文ではないが、ネガ語 > ポジ語 かつポジ/ニュートラル判定ならネガ寄せ
    elif neg_hits > pos_hits and sentiment in ("positive", "neutral"):
        sentiment = "negative"
        sentiment_score = min(sentiment_score, 0.3)

    # 3) もともとネガ判定だがスコアがポジ寄りに寄り過ぎているとき → ネガ側に寄せておく
    elif sentiment == "negative" and sentiment_score > 0.4:
        if neg_hits >= 2:
            sentiment_score = 0.2
        else:
            sentiment_score = 0.35

    # 4) 強いポジ表現のわりにスコアが低いときは少し持ち上げる
    elif sentiment == "positive" and pos_hits > neg_hits and sentiment_score < 0.6:
        sentiment_score = max(sentiment_score, 0.7)

    sentiment_score = float(max(0.0, min(sentiment_score, 1.0)))
    return sentiment, sentiment_score

# ==========================
#  FastAPI 基本設定
# ==========================
app = FastAPI()

# 静的ファイル
app.mount("/admin/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
PUBLIC_DIR = BASE_DIR.parent / "public"
app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public")

@app.get("/")
async def root():
    return FileResponse(PUBLIC_DIR / "index.html")

@app.get("/admin")
async def admin_page():
    return FileResponse(BASE_DIR / "admin.html")


@app.get("/admin/map")
async def admin_map_page():
    return FileResponse(BASE_DIR / "map.html")


@app.get("/admin/api/voices")
def get_latest_voices():
    """
    Supabase の 'voices' テーブルから最新20件を取得
    """
    try:
        response = (
            supabase.table("voices")
            .select("*")
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        rows = response.data or []

        # DB カラム important_score → API 上は importance_score にコピー
        for row in rows:
            if "importance_score" not in row and "important_score" in row:
                row["importance_score"] = row["important_score"]

        return rows
    except Exception as e:
        return {"error": str(e)}


# ==========================
#  AI レポート (全体)
# ==========================
@app.get("/admin/api/report/global")
def get_global_report():
    """
    Supabase の voices をまとめて gpt-4o-mini に投げ、
    全体の課題傾向レポートを生成して返す。
    """
    try:
        resp = (
            supabase.table("voices")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        voices: List[Dict] = resp.data or []

        if not voices:
            return {"text": "まだデータがありません。まず現地で声を集めてください。"}

        lines = []
        for i, v in enumerate(voices, start=1):
            area = v.get("area") or "不明"
            cat = v.get("category") or "未分類"
            imp = v.get("importance_score", v.get("important_score"))
            sent = v.get("sentiment") or "-"
            text = v.get("raw_text") or ""
            snippet = text[:80].replace("\n", " ")
            lines.append(
                f"{i}. 地区: {area} / カテゴリ: {cat} / 重要度: {imp} / 感情: {sent} / 声: 「{snippet}…」"
            )

        voices_block = "\n".join(lines)

        core_prompt = f"""
あなたは、日本の地方自治体のまちづくりを支援する政策アナリストAIです。
以下は、熊野市周辺で集めた住民・学生の声の一覧です。

【データ一覧】
{voices_block}

---

上記のデータを踏まえて、次の観点で「全体AI分析レポート」を日本語で作成してください。
出力は、以下のフォーマットだけを含め、ここまでの説明文は繰り返さないでください。

【出力フォーマット】

全体AI分析レポート

1. 全体像（大きな課題テーマ）
- 全体として見えてくる大きな課題テーマを 3〜5 個挙げ、
  それぞれに 2〜3 行の簡潔な説明を付けてください。

2. 地区ごとの特徴
- 神川・木本・飛鳥・市木・紀和 など各地区について、
  (1) 特徴的に現れている課題
  (2) ポジティブなポイント（強み・活かせる資源）
  を箇条書きで整理してください。

3. 早急に対応が必要そうな論点
- 特に優先度が高いと思われる論点を 3〜5 個挙げ、
  「なぜ優先度が高いと考えられるのか」を 1〜2 行で説明してください。

4. 関係者別の打ち手アイデア
- 行政（市役所など）
- 地域住民・地元団体
- 学生ボランティアチーム
  それぞれについて、具体的な打ち手アイデアを 3〜5 個ずつ箇条書きで提案してください。
  （例：◯◯の情報発信を強化する、◯◯ワークショップを試行する 等）

5. 今後の検討・調査の方向性
- 学生チームが現地で検討を深める際に意識すると良さそうな
  「問いかけ」や「次の調査テーマ」を 3〜5 個挙げてください。

文章全体は、市役所の担当者が読んでも分かりやすいように、
丁寧で簡潔な日本語で書いてください。
"""

        text = generate_with_llm(core_prompt, max_output_tokens=900)
        return {"text": text}

    except Exception as e:
        print("Global report error:", e)
        return {"text": f"分析レポート生成中にエラーが発生しました: {e}"}


# ==========================
#  AI レポート (スポット＝1件)
# ==========================
@app.get("/admin/api/report/voice/{voice_id}")
def get_voice_report(voice_id: str):
    """
    特定の1件の声 + 同じ地区の他の声を参考に、スポット分析レポートを生成
    """
    try:
        resp = (
            supabase.table("voices")
            .select("*")
            .eq("id", voice_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return {"text": "該当するデータが見つかりませんでした。"}

        main_voice = rows[0]
        area = main_voice.get("area") or "不明"
        cat = main_voice.get("category") or "未分類"
        sent = main_voice.get("sentiment") or "-"
        imp = main_voice.get("importance_score", main_voice.get("important_score"))
        text = main_voice.get("raw_text") or ""

        related_resp = (
            supabase.table("voices")
            .select("*")
            .eq("area", area)
            .neq("id", voice_id)
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        related = related_resp.data or []

        related_lines = []
        for i, v in enumerate(related, start=1):
            r_cat = v.get("category") or "未分類"
            r_imp = v.get("importance_score", v.get("important_score"))
            r_text = (v.get("raw_text") or "")[:60].replace("\n", " ")
            related_lines.append(
                f"{i}. カテゴリ: {r_cat} / 重要度: {r_imp} / 声: 「{r_text}…」"
            )
        related_block = "\n".join(related_lines) if related_lines else "（同地区の他の声は少数です）"

        core_prompt = f"""
あなたは、日本の地方自治体のまちづくりを支援する政策アドバイザーAIです。
以下は、ある 1 件の「住民の声／学生の気づき」と、同じ地区から寄せられている他の声の一覧です。

【対象の声】
- 地区: {area}
- カテゴリ: {cat}
- 感情: {sent}
- 重要度スコア(AI推定): {imp}
- 内容:
\"\"\"{text}\"\"\"

【同じ地区から寄せられている他の声の例】
{related_block}

---

上記の情報を踏まえて、次の観点で「スポットAI分析メモ」を日本語で作成してください。
出力には、この説明文そのものや「次の観点で〜」といった指示文は含めず、分析メモだけを書いてください。

【出力フォーマット】

スポットAI分析メモ

1. 顕在化している課題
- この声から読み取れる「顕在化している課題」を、箇条書きで整理してください。

2. 背景にありそうな要因
- 地理的条件・人口構造・生活動線・制度やルールなど、
  背景にありそうな要因を 3〜5 個ほど推測しながら説明してください。

3. 潜在的なリスク・将来の懸念
- 今は言葉にされていないが、将来顕在化しそうな潜在的課題やリスクがあれば、
  箇条書きで指摘してください。

4. 関係者別の打ち手アイデア
- 行政（市役所など）
- 地域住民・事業者
- 学生ボランティアチーム
  それぞれについて、今すぐ・短期的に試せそうな打ち手アイデアを 2〜4 個ずつ提案してください。
  ヒントではなく、「まずこういう取り組みから始めてみるとよい」という具体例を書いてください。

5. 今後のヒアリングで投げかけたい質問例
- 次回以降の現地調査で、住民や関係者に投げかけると良さそうな質問例を 3〜5 個挙げてください。

文章全体は、市役所の担当者と学生チームが一緒に議論するための
「たたき台メモ」になるイメージで、丁寧かつ簡潔な日本語で書いてください。
"""

        text_out = generate_with_llm(core_prompt, max_output_tokens=700)
        return {"text": text_out}

    except Exception as e:
        print("Voice report error:", e)
        return {"text": f"個別レポート生成中にエラーが発生しました: {e}"}

# ==========================
#  日本語感情分析モデル（Sonoisa 系 / v2 相当）
# ==========================
# 以前使っていた「Sonoisa の感情モデル」に近い構成に戻す。
# Hugging Face 上の 3 クラス（negative / neutral / positive）モデルを想定。
SENTIMENT_MODEL = "koheiduck/bert-japanese-finetuned-sentiment"  # Sonoisa 系日本語感情モデル

# ==========================
#  ローカルNLPのON/OFF（RenderではOFF推奨）
# ==========================
USE_LOCAL_NLP = os.getenv("USE_LOCAL_NLP", "0") == "1"  # Renderではデフォルト0推奨

_LOCAL_MODELS = None  # キャッシュ

def get_local_models():
    """
    重いモデル類は import 時にロードしない。
    必要になった時だけロードしてキャッシュする。
    """
    global _LOCAL_MODELS

    if not USE_LOCAL_NLP:
        return None

    if _LOCAL_MODELS is not None:
        return _LOCAL_MODELS

    # ここで初めて重いimport（Renderのポート検知を先に通すため）
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer, util as st_util
    from keybert import KeyBERT
    import torch
    import torch.nn.functional as F

    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    # id2label を揃える
    raw_id2label = sentiment_model.config.id2label
    sentiment_id2label = {int(k): v for k, v in raw_id2label.items()}

if USE_LOCAL_NLP:
    from sentence_transformers import SentenceTransformer, util
    from keybert import KeyBERT
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F

    # ここに元々あった tokenizer/model 初期化を入れる
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    # ここに元々あった id2label の整形があればそのまま
    _raw_id2label = sentiment_model.config.id2label
    SENTIMENT_ID2LABEL = {int(k): v for k, v in _raw_id2label.items()}

    keyword_model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens-v2")
    kw_model = KeyBERT(model=keyword_model)

else:
    # Render(512MB)では基本こっち。落ちないためのダミー
    util = None
    torch = None
    F = None
    sentiment_tokenizer = None
    sentiment_model = None
    SENTIMENT_ID2LABEL = {}
    keyword_model = None
    kw_model = None

# ==========================
#  KeyBERT / Sentence-BERT
# ==========================
keyword_model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens-v2")
kw_model = KeyBERT(model=keyword_model)

# ==========================
#  リクエストモデル
# ==========================
class TextPayload(BaseModel):
    text: str
    area: Optional[str] = None
    source: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None


class FeedbackPayload(BaseModel):
    id: str
    cleaned_text: str
    correct_category: str


# ==========================
#  カテゴリセンター（意味ベクトル）
# ==========================
category_vectors: Dict[str, np.ndarray] = {}
category_counts: Dict[str, int] = {}
category_vectors[category] = center_vec
category_counts[category] = 1

def ensure_category_centers():
    """
    カテゴリ中心ベクトルは必要になったタイミングで作る（起動を軽くする）。
    """
    global category_vectors, category_counts
    if category_vectors:
        return
    models = get_local_models()
    if models is None:
        return

    keyword_model = models["keyword_model"]

    for category, words in CATEGORY_RULES.items():
        if not words:
            continue
        vecs = keyword_model.encode(words)
        center_vec = np.mean(vecs, axis=0)
        category_vectors[category] = center_vec
        category_counts[category] = 1

def classify_category_by_semantic(keywords: list) -> str:
    if not keywords:
        return "未分類"

    models = get_local_models()
    if models is None:
        # ローカルNLP OFF時はルールベースだけで判定（軽量）
        for kw in keywords:
            for category, words in CATEGORY_RULES.items():
                if any(w in kw for w in words):
                    return category
        return "未分類"

    ensure_category_centers()

    keyword_model = models["keyword_model"]
    st_util = models["st_util"]

    kw_vecs = keyword_model.encode(keywords)

    scores = {cat: 0.0 for cat in CATEGORY_RULES.keys()}

    for kw in keywords:
        for category, words in CATEGORY_RULES.items():
            for w in words:
                if w in kw:
                    scores[category] += 1.0

    for kw_vec in kw_vecs:
        for category, cat_vec in category_vectors.items():
            sim = st_util.cos_sim(kw_vec, cat_vec).item()
            if sim > 0:
                scores[category] += sim

    best_category = max(scores, key=scores.get)
    if scores[best_category] < 0.3:
        return "未分類"
    return best_category

def update_category_center(category: str, new_text: str, alpha: float = 0.85):
    global category_vectors, category_counts
    new_vec = keyword_model.encode([new_text])[0]
    old_vec = category_vectors[category]
    updated_vec = alpha * old_vec + (1 - alpha) * new_vec
    category_vectors[category] = updated_vec
    category_counts[category] += 1


# ==========================
#  メイン解析エンドポイント
# ==========================
@app.post("/analyze")
async def analyze(payload: TextPayload):
    text = payload.text
    nlp_id = str(uuid.uuid4())

    cleaned = preprocess_text(text)
    keywords = extract_keywords(cleaned)
    category = classify_category_by_semantic(keywords)
    sentiment, sentiment_score = analyze_sentiment(cleaned_text=cleaned, raw_text=text)
    urgency = estimate_urgency(text, sentiment, sentiment_score)
    topic, topic_id = simple_topic(cleaned)

    importance = calc_importance(sentiment_score, urgency)

    result = {
        "nlp_id": nlp_id,
        "cleaned_text": cleaned,
        "keywords": keywords,
        "category": category,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "urgency": urgency,
        "topic": topic,
        "topic_id": topic_id,
        "importance_score": importance,
        "area": payload.area,
        "source": payload.source,
        "age": payload.age,
        "gender": payload.gender,
        "occupation": payload.occupation,
    }

    save_to_supabase(
        raw_text=text,
        area=payload.area,
        source=payload.source,
        category=category,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        urgency=urgency,
        topic=topic,
        topic_id=topic_id,
        importance=importance,
        age=payload.age,
        gender=payload.gender,
        occupation=payload.occupation,
        lat=payload.lat,
        lng=payload.lng,
    )

    return result


@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    text = payload.cleaned_text
    category = payload.correct_category
    update_category_center(category, text)
    return {"status": "updated", "category": category}


# ==========================
#  前処理・キーワード・感情など
# ==========================
def preprocess_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()

    noise_patterns = [
        r"えーと", r"あのー", r"なんか", r"そのー", r"まぁ",
        r"やっぱり", r"ていうか", r"なんやけど", r"〜やけど",
        r"〜んやけど", r"えっと", r"ええと", r"そうですね",
    ]
    for pat in noise_patterns:
        text = re.sub(pat, "", text)

    tokens = tokenizer_obj.tokenize(text, mode)
    stopwords = {
        "する", "ある", "こと", "これ", "それ", "さん",
        "です", "ます", "よう", "ところ", "もの",
        "ため", "みたい", "感じ", "なる", "いる",
    }

    important_words = []
    for tok in tokens:
        base = tok.dictionary_form()
        pos = tok.part_of_speech()[0]
        if pos in ["名詞", "動詞", "形容詞"] and base not in stopwords:
            important_words.append(base)

    cleaned = " ".join(important_words)
    return cleaned

def extract_keywords(text: str):
    if not text or len(text) < 3:
        return []

    models = get_local_models()
    if models is None:
        # ローカルNLP OFF時は簡易抽出（とりあえず動かす用）
        return re.findall(r"[一-龥ぁ-んァ-ンA-Za-z0-9]{2,}", text)[:5]

    kw_model = models["kw_model"]

    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            use_mmr=True,
            diversity=0.7,
            top_n=5,
        )
        return [kw for kw, score in keywords]
    except Exception as e:
        print("Keyword extraction error:", e)
        return []

def analyze_sentiment(cleaned_text: str, raw_text: Optional[str] = None):
    if not cleaned_text and not raw_text:
        return "neutral", 0.5

    models = get_local_models()
    if models is None:
        return "neutral", 0.5

    torch = models["torch"]
    F = models["F"]
    sentiment_tokenizer = models["sentiment_tokenizer"]
    sentiment_model = models["sentiment_model"]
    SENTIMENT_ID2LABEL = models["SENTIMENT_ID2LABEL"]

    text_for_model = cleaned_text or raw_text

    inputs = sentiment_tokenizer(
        text_for_model,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    max_idx = int(np.argmax(probs))
    raw_label = SENTIMENT_ID2LABEL.get(max_idx, str(max_idx))
    sentiment = raw_label.lower()

    label_to_idx = {v.lower(): k for k, v in SENTIMENT_ID2LABEL.items()}
    neg_idx = label_to_idx.get("negative") or label_to_idx.get("neg")
    pos_idx = label_to_idx.get("positive") or label_to_idx.get("pos")

    if neg_idx is not None and pos_idx is not None and len(probs) >= 2:
        neg_p = float(probs[neg_idx])
        pos_p = float(probs[pos_idx])
        polar = pos_p - neg_p
        sentiment_score = (polar + 1.0) / 2.0
    else:
        if sentiment.startswith("neg"):
            sentiment_score = 0.0
        elif sentiment.startswith("pos"):
            sentiment_score = 1.0
        else:
            sentiment_score = 0.5

    if raw_text:
        sentiment, sentiment_score = adjust_sentiment_by_negative_words(
            sentiment, sentiment_score, raw_text
        )

    sentiment_score = max(0.0, min(1.0, float(sentiment_score)))
    return sentiment, sentiment_score

def estimate_urgency(text: str, sentiment: Optional[str] = None, sentiment_score: float = 0.5):
    """
    緊急度を 1〜5 で推定。
    - 危険・命・事故系の単語 → 自動的に 4〜5
    - ネガティブであればあるほど 4〜5 に寄せる
    - 強いポジティブは 1〜2 に寄せることもある
    """
    if not text:
        return 3

    base = 3  # デフォルト

    high_patterns = [
        r"命にかかわる", r"命に関わる", r"死亡", r"死ぬ",
        r"重大な事故", r"大事故", r"倒壊", r"崩落",
        r"津波", r"南海トラフ",
        r"通学路.*危険", r"避難できない",
    ]

    mid_patterns = [
        r"かなり不便", r"本当に不便", r"とても不便",
        r"困っている", r"かなり困る", r"大変",
        r"不安", r"心配", r"怖い",
        r"このままでは.*よくない",
        r"早く.*(なんとか|対応)してほしい",
    ]

    if any(re.search(p, text) for p in high_patterns):
        base = 5
    elif any(re.search(p, text) for p in mid_patterns):
        base = 4

    # 極性スコアからネガ/ポジ強度を算出
    neg_intensity = max(0.0, (0.5 - sentiment_score) * 2.0)  # 0〜1（ネガ側）
    pos_intensity = max(0.0, (sentiment_score - 0.5) * 2.0)  # 0〜1（ポジ側）

    if sentiment == "negative":
        if neg_intensity > 0.7:
            base = max(base, 5)
        elif neg_intensity > 0.4:
            base = max(base, 4)
    elif sentiment == "positive" and pos_intensity > 0.6:
        base = min(base, 2)

    base = max(1, min(base, 5))
    return base

def simple_topic(text: str):
    # TODO: 必要なら軽量トピック分類を実装
    return "交通インフラ問題", 1

def calc_importance(sent_score: float, urgency: int):
    """
    sent_score: 0〜1（0=ネガ, 0.5=中立, 1=ポジ）
    urgency   : 1〜5

    ネガティブ度 × 緊急度 をベースに重要度スコア(1.0〜5.0)を計算。
    """
    # 緊急度を 0〜1 に
    urgency_norm = (urgency - 1) / 4.0  # 1〜5 → 0〜1

    # ネガティブ度（0=全ポジ, 1=全ネガ）
    neg_intensity = max(0.0, (0.5 - sent_score) * 2.0)
    # 感情の強さ（中立=0, 極端なポジ/ネガ=1）
    strength = 2.0 * abs(sent_score - 0.5)

    # ベース 1.2 に、
    # ・ネガ度を強めに
    # ・緊急度をそこそこ強めに
    # ・感情の強さも少しだけ
    importance = 1.2 + 2.0 * neg_intensity + 1.6 * urgency_norm + 0.8 * strength

    importance = max(1.0, min(5.0, importance))
    return round(importance, 2)

# ==========================
#  アプリ起動
# ==========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
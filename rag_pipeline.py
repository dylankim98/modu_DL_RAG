# rag_pipeline.py
from typing import List, Dict, Tuple
import re

from rag_llm import llm_chat, llm_chat_stream
from retriever import retriever

# ---------------------------
# Language detect
# ---------------------------
def detect_language(text: str) -> str:
    if re.search(r"[가-힣]", text):
        return "Korean"
    return "English"

# ---------------------------
# Persona (LLM 말투/규칙)
# ---------------------------
PERSONA_FOREIGN_BEGINNER = """
You are 'K-recipe', a Korean food guide chatbot for foreigners living in Korea.

Persona:
- User is a foreigner living in Korea
- Beginner at cooking
- Not familiar with Korean ingredients or cooking terms
- Wants simple, short, practical explanations
- Friendly tone, not formal
- Avoid long explanations

Behavior rules:
- Be empathetic first
- Use bullet points
- Max 5 cooking steps
- If Korean terms are used, explain briefly
"""

# ---------------------------
# Helpers: parse user inputs
# ---------------------------
def parse_ingredients(text: str) -> List[str]:
    if not text or text.strip() in ["없음", "none", "None"]:
        return []
    # 쉼표/줄바꿈/슬래시 등 대충 분리
    items = re.split(r"[,/\\|\n]+", text)
    return [i.strip() for i in items if i.strip()]

def normalize_level(level: str) -> str:
    # 데이터에 "초급", "중급", "고급", "아무나" 등
    return (level or "").strip()

def time_to_minutes(time_str: str) -> int:
    """
    '30분이내' -> 30
    '60분이내' -> 60
    '정보 없음' -> 9999
    """
    if not time_str:
        return 9999
    t = time_str.strip()
    if "정보" in t:
        return 9999
    m = re.search(r"(\d+)\s*분", t)
    if m:
        return int(m.group(1))
    return 9999

# ---------------------------
# Scoring (우선순위 엔진)
# ---------------------------
def score_doc(doc, user_ings: List[str], style_hint: str) -> Tuple[float, Dict]:
    md = doc.metadata or {}
    text = doc.page_content or ""

    level = normalize_level(md.get("level", ""))
    views = int(md.get("views", 0) or 0)
    cook_time = time_to_minutes(md.get("time", ""))

    # 1) 재료 매칭 (재료 문자열에 포함 여부로 단순 매칭)
    ing_hit = 0
    for ing in user_ings:
        if ing and ing in text:
            ing_hit += 1

    # 2) 난이도 점수 (초보 가정)
    level_score = 0
    if level in ["초급", "아무나", "쉬움", "Easy"]:
        level_score = 5
    elif level in ["중급"]:
        level_score = 2
    else:
        level_score = 0

    # 3) 조회수 점수 (log 느낌으로 완만하게)
    #   0~5 사이로 압축
    pop_score = min(5.0, views / 5000.0)  # 25k면 5점

    # 4) 스타일 힌트 (텍스트 포함 시 보너스)
    style_score = 0
    if style_hint and style_hint != "상관없음":
        if style_hint in text or style_hint in str(md.get("situation", "")) or style_hint in str(md.get("method", "")):
            style_score = 1.5

    # 5) 조리시간 페널티 (너무 오래 걸리면 감점)
    time_penalty = 0.0
    if cook_time <= 30:
        time_penalty = 0.0
    elif cook_time <= 60:
        time_penalty = 0.5
    else:
        time_penalty = 1.5

    # 최종 점수: (재료가 있으면 강하게) + (쉬우면 강하게) + (인기도) + (스타일) - (시간)
    final = (ing_hit * 3.0) + (level_score * 1.5) + (pop_score * 1.0) + style_score - time_penalty

    debug = {
        "ing_hit": ing_hit,
        "level": level,
        "views": views,
        "cook_time": cook_time,
        "final": final
    }
    return final, debug

# ---------------------------
# LLM title rewriting (선택)
# ---------------------------
def make_witty_title(raw_title: str, user_story: str, language: str) -> str:
    # 너무 과하면 UI 깨져서 짧게 제한
    prompt = f"""
You rename Korean dish titles into short, witty but clear titles.
Rules:
- Keep the original dish recognizable
- Max 12 words
- No clickbait, no insult
- Output ONLY the title

Language: {language}
Original dish: {raw_title}
User mood: {user_story}
"""
    try:
        out = llm_chat(prompt).strip()
        return out if out else raw_title
    except Exception:
        return raw_title

# ---------------------------
# Menu suggestion (RAG + ranking)
# ---------------------------
def suggest_menus(user_story: str, ingredients: str, style_hint: str = "") -> List[Dict]:
    user_ings = parse_ingredients(ingredients)

    # ✅ 의미 검색용 Query: 감정/재료/스타일을 자연어로 묶음
    query = f"""
User mood: {user_story}
Ingredients: {ingredients}
Style: {style_hint}
Find suitable Korean recipes.
Beginner friendly.
""".strip()

    docs = retriever.invoke(query)  # Top-N 후보

    # 점수화 후 상위 5개
    scored = []
    for d in docs:
        s, dbg = score_doc(d, user_ings, style_hint)
        scored.append((s, d, dbg))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]

    language = detect_language(user_story)
    menus: List[Dict] = []
    for s, d, dbg in top:
        md = d.metadata or {}
        raw_title = md.get("menu", "") or md.get("title", "") or "Unknown"

        # UI용 위트 제목
        display_title = make_witty_title(raw_title, user_story, language)

        # subtitle은 레시피제목(원문) 우선
        subtitle = md.get("title", "")
        level = md.get("level", "")
        views = int(md.get("views", 0) or 0)

        # 짧은 태그
        tags = []
        if level:
            tags.append(level)
        if md.get("method"):
            tags.append(md["method"])
        if md.get("time") and "정보" not in str(md.get("time")):
            tags.append(md["time"])

        # meme(한 줄) - 인기도/재료 매칭 기반
        if dbg["ing_hit"] >= 2:
            meme = "재료 매칭 꽤 좋다. 오늘은 이걸로 간다."
        elif views >= 5000:
            meme = "검증된 인기 레시피 쪽으로 안전하게."
        else:
            meme = "부담 없는 선택. 실패 확률 낮추자."

        # spice는 데이터에 없으니 임시(필요하면 조리방법/상황으로 추정 가능)
        spice = 3

        menus.append({
            "title": display_title,   # UI 표시용
            "raw_title": raw_title,   # ✅ DB 키(레시피 재검색/정합성)
            "subtitle": subtitle,
            "tags": tags[:3],
            "spice": spice,
            "meme": meme,
            "debug": dbg,  # 필요 없으면 UI에서 안 쓰면 됨
        })

    return menus

# ---------------------------
# Recipe generation (RAG context + LLM)
# ---------------------------
def recipe_stream(user_story: str, ingredients: str, picked_menu_title: str):
    language = detect_language(user_story)

    #  선택된 메뉴로 다시 검색해서 컨텍스트 확보 (정확도↑)
    #    (picked_menu_title이 raw_title일 때 가장 정확)
    query = f"요리명: {picked_menu_title}\nIngredients: {ingredients}\n"
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs[:3]])

    prompt = f"""
{PERSONA_FOREIGN_BEGINNER}

Context (retrieved recipes):
{context}

User mood: {user_story}
Ingredients available: {ingredients}
Selected menu: {picked_menu_title}

Output format (STRICT):
1) One short empathetic sentence (1 line)
2) Core ingredients (bullet list)
3) Cooking steps (max 5 bullets)
4) 2 common mistakes to avoid
5) 2 YouTube Shorts search keywords (TEXT ONLY, no links)

Rules:
- Answer ONLY in {language}
- Simple words only
- No long paragraphs
- If Korean ingredient appears, explain briefly
"""
    return llm_chat_stream(prompt)

# ---------------------------
# Empathy message (LLM)
# ---------------------------
def empathize_story(user_story: str) -> str:
    language = detect_language(user_story)
    prompt = f"""
{PERSONA_FOREIGN_BEGINNER}

Task:
- Respond in 2~3 short sentences
- 1: genuine empathy
- 2: light humor (gentle, no sarcasm)
- 3: ask naturally about fridge ingredients

Rules:
- Answer ONLY in {language}

User situation:
{user_story}
"""
    try:
        return llm_chat(prompt).strip()
    except Exception:
        return "That sounds like a long day. Let’s fix it with food. What ingredients do you have?"



# 6장/streamlit_chat.py
import os
import streamlit as st
from rag_pipeline import suggest_menus, recipe_stream, empathize_story

st.set_page_config(page_title="K-recipe", layout="wide")

# ---- API KEY 체크 (없으면 안내) ----
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY가 설정되지 않았어. .env 또는 환경변수 설정 확인해줘.")

# ---- CSS: 밝고 ‘앱 같은’ 톤 ----
st.markdown("""
<style>
.stApp{
  background:
    radial-gradient(1200px 700px at 10% 0%, rgba(255,217,185,0.55) 0%, rgba(255,217,185,0.0) 60%),
    radial-gradient(900px 600px at 90% 10%, rgba(190,230,255,0.55) 0%, rgba(190,230,255,0.0) 55%),
    linear-gradient(180deg, #ffffff 0%, #fff7ed 100%);
}
.block-container{max-width:1100px; padding-top:2.2rem;}

.hero{
  border-radius:24px;
  padding:22px 22px;
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(17,24,39,0.06);
  box-shadow: 0 18px 40px rgba(17,24,39,0.08);
  display:flex;
  align-items:center;
  gap:18px;
}
.hero h1{ margin:0; font-size:40px; letter-spacing:-1px; color:#111827; }
.hero p{ margin:6px 0 0 0; color:#6b7280; font-size:14px; }
.badge{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  background:#111827;
  color:white;
  font-size:12px;
  font-weight:700;
}
.hr{ height:1px; background: rgba(17,24,39,0.08); margin:18px 0; }

.card{
  background:#fff;
  border:1px solid rgba(17,24,39,0.08);
  border-radius:18px;
  box-shadow: 0 12px 30px rgba(17,24,39,0.08);
  padding:16px 16px 14px 16px;
  min-height:200px;
}
.card-title{font-size:20px;font-weight:900;color:#111827;line-height:1.2;}
.card-sub{margin-top:8px;color:#4b5563;font-size:14px;}
.tag{
  display:inline-block;padding:6px 10px;border-radius:999px;
  background:#f3f4f6;color:#374151;font-size:12px;
  margin-right:6px;border:1px solid rgba(17,24,39,0.06);
}
.meme{margin-top:10px;font-size:13px;color:#6b7280;}
.stButton>button{border-radius:999px !important; font-weight:800 !important;}
.stTextInput>div>div>input, .stTextArea textarea{ border-radius:14px !important; }
</style>
""", unsafe_allow_html=True)

# ---- state ----
if "stage" not in st.session_state:
    st.session_state.stage = "story"   # story -> ingredients -> style -> menus -> recipe
if "story" not in st.session_state:
    st.session_state.story = ""
if "empathy" not in st.session_state:
    st.session_state.empathy = ""
if "ingredients" not in st.session_state:
    st.session_state.ingredients = ""
if "style" not in st.session_state:
    st.session_state.style = "상관없음"
if "menus" not in st.session_state:
    st.session_state.menus = []
if "picked" not in st.session_state:
    st.session_state.picked = None

def reset_all():
    st.session_state.stage = "story"
    st.session_state.story = ""
    st.session_state.empathy = ""
    st.session_state.ingredients = ""
    st.session_state.style = "상관없음"
    st.session_state.menus = []
    st.session_state.picked = None
    st.rerun()

# ---- sidebar ----
with st.sidebar:
    st.header("K-recipe")
    st.caption("감정 + 냉장고 재료 기반 맞춤 한식 추천")
    if st.button("처음으로 돌아가기", use_container_width=True):
        reset_all()

# ---- hero ----
st.markdown("""
<div class="hero">
  <div style="flex:1;">
    <span class="badge"> Team : 응답하RAG </span>
    <h1>K-recipe</h1>
    <p>오늘의 감정 + 냉장고 상황에 맞는 한국요리 추천</p>
  </div>
  <div style="width:220px; text-align:right; opacity:0.9;">
    <img src="https://images.unsplash.com/photo-1555939594-58d7cb561ad1?auto=format&fit=crop&w=520&q=60"
         style="width:210px;height:120px;border-radius:18px;object-fit:cover;border:1px solid rgba(17,24,39,0.08);" />
  </div>
</div>
<div class="hr"></div>
""", unsafe_allow_html=True)

# ---- stage: story ----
if st.session_state.stage == "story":
    st.subheader("오늘은 어땠어?")
    st.caption("메뉴 추천하려면 네 상황부터 듣고 싶어. 짧게 한 줄이면 충분.")

    story = st.text_input(
        "한 줄로 말해줘",
        value=st.session_state.story,
        placeholder="예) 오늘 멘탈 박살… 위로되는 거 먹고 싶다."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("다음", use_container_width=True):
            if not story.strip():
                st.error("한 줄만이라도 적어줘. 그게 추천의 재료야.")
            else:
                st.session_state.story = story.strip()
                with st.spinner("사연 접수 중..."):
                    st.session_state.empathy = empathize_story(st.session_state.story)
                st.session_state.stage = "ingredients"
                st.rerun()

    with col2:
        if story.strip():
            st.info("좋아. 이제 ‘다음’ 누르면 내가 사연 접수하고 냉장고 상황으로 넘어갈게.")
        else:
            st.caption("")

# ---- stage: ingredients ----
elif st.session_state.stage == "ingredients":
    if st.session_state.empathy:
        st.info(st.session_state.empathy)

    st.subheader("냉장고에 뭐 있어?")
    st.caption("쉼표로 적어줘. 없어도 괜찮아(없으면 ‘없음’이라고 써도 됨).")

    ing = st.text_input(
        "보유 재료",
        value=st.session_state.ingredients,
        placeholder="예) 김치, 돼지고기, 대파, 두부"
    )

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("이전", use_container_width=True):
            st.session_state.stage = "story"
            st.rerun()
    with col2:
        if st.button("다음", use_container_width=True):
            st.session_state.ingredients = (ing.strip() if ing.strip() else "없음")
            st.session_state.stage = "style"
            st.rerun()

# ---- stage: style ----
elif st.session_state.stage == "style":
    if st.session_state.empathy:
        st.info(st.session_state.empathy)

    st.subheader("원하는 스타일은?")
    st.caption("취향 한 번만 찍어줘. 그 다음에 메뉴 후보 카드 보여줄게.")

    options = ["상관없음", "초간단", "칼칼/매콤", "든든한 한 끼", "다이어트 느낌", "혼술 안주"]
    style = st.selectbox(
        "스타일",
        options,
        index=options.index(st.session_state.style) if st.session_state.style in options else 0
    )

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("이전", use_container_width=True):
            st.session_state.stage = "ingredients"
            st.rerun()
    with col2:
        if st.button("메뉴 후보 보기", use_container_width=True):
            st.session_state.style = style
            with st.spinner("메뉴 후보 만드는 중..."):
                st.session_state.menus = suggest_menus(
                    st.session_state.story,
                    st.session_state.ingredients,
                    st.session_state.style
                )
            st.session_state.stage = "menus"
            st.rerun()

# ---- stage: menus ----
elif st.session_state.stage == "menus":
    if st.session_state.empathy:
        st.info(st.session_state.empathy)

    st.subheader("이 상황엔… 이 메뉴들이 딱이야")
    st.caption("하나 고르면 레시피는 핵심만 딱 보여줄게.")

    menus = st.session_state.menus or []
    if not menus:
        st.warning("후보를 못 뽑았어. 다시 시도해볼까?")
        if st.button("다시 뽑기", use_container_width=True):
            st.session_state.stage = "style"
            st.rerun()
        st.stop()

    cols = st.columns(2, gap="large")
    for i, m in enumerate(menus):
        c = cols[i % 2]
        with c:
            st.markdown(f"""
            <div class="card">
              <div>
                <div class="card-title">{m.get("title","")}</div>
                <div class="card-sub">{m.get("subtitle","")}</div>
                <div style="margin-top:10px;">
                  {''.join([f'<span class="tag">{t}</span>' for t in (m.get("tags") or [])[:3]])}
                </div>
                <div class="meme">{m.get("meme","")}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            spice = max(1, min(5, int(m.get("spice", 3))))
            spice_bar = "🌶️" * spice
            if st.button(f"{spice_bar}  이 메뉴로 간다", key=f"pick_{i}", use_container_width=True):
                st.session_state.picked = m.get("raw_title") or m.get("title")
                st.session_state.stage = "recipe"
                st.rerun()

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("다른 후보 다시 뽑기", use_container_width=True):
            with st.spinner("다시 추천 중..."):
                st.session_state.menus = suggest_menus(
                    st.session_state.story,
                    st.session_state.ingredients,
                    st.session_state.style
                )
            st.rerun()
    with col2:
        if st.button("처음으로 돌아가기", use_container_width=True):
            reset_all()

# ---- stage: recipe ----
elif st.session_state.stage == "recipe":
    picked = st.session_state.picked
    if not picked:
        st.session_state.stage = "menus"
        st.rerun()

    if st.session_state.empathy:
        st.info(st.session_state.empathy)

    st.subheader(f"선택 메뉴: {picked}")
    st.caption("레시피는 핵심만 보여줄게.")

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("메뉴 다시 고르기", use_container_width=True):
            st.session_state.stage = "menus"
            st.rerun()
    with col2:
        if st.button("처음으로 돌아가기", use_container_width=True):
            reset_all()

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        st.write_stream(
            recipe_stream(
                st.session_state.story,
                st.session_state.ingredients,
                picked
            )
        )

"""
edu_visualizer.py — 학생의 생각을 화면에 보여주는 교육용 래퍼 (Streamlit)
========================================================================
선생님이 큰 화면(프로젝터)에 띄워 학생에게 보여주는 도구.

기능:
  1) 상황을 쉬운 말로 입력 → EasyReadGuardrail(자체 엔진)이 쉬움/어려움 판정(신호등)
  2) 키워드를 격자 맵에 큰 색깔 블록으로 표현
  3) ▶ 한 칸 진행: 사람이 위험(불)을 피해 안전한 곳으로 한 걸음씩 이동 (상호작용)
  4) (옵션) 어려운 말이면 'LLM으로 쉽게 바꾸기' — 판정은 자체 엔진, 생성만 LLM

원칙: 판정의 주인공은 가드레일. LLM은 '쉽게 바꾸기' 생성에만. 키 없으면 기능만 숨김.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import Patch
from easyread_guardrail import EasyReadGuardrail

# ── 한글 폰트 ───────────────────────────────────────────
for _f in ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        rcParams["font.family"] = _f; break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="쉬운 말 그림판", layout="wide")

# code: 0 빈 땅, 1 길, 2 사람, 3 안전한 곳, 4 위험
KEYWORDS = {
    "불": (4, "위험"), "화재": (4, "위험"), "연기": (4, "위험"), "위험": (4, "위험"),
    "대피소": (3, "안전한 곳"), "비상구": (3, "안전한 곳"), "밖": (3, "안전한 곳"),
    "길": (1, "길"), "도로": (1, "길"), "계단": (1, "길"),
    "사람": (2, "사람"), "친구": (2, "사람"), "선생님": (2, "사람"), "건물": (2, "건물"),
}
COLORS = {0: "#EEEEEE", 1: "#FFE08A", 2: "#9AD0F5", 3: "#9BE3A2", 4: "#FF8A80"}
LABELS = {0: "빈 땅", 1: "길", 2: "사람", 3: "안전한 곳", 4: "위험"}
N = 8


@st.cache_resource
def load_guardrail():
    g = EasyReadGuardrail(student_bonus=2.0, student_min_count=1)
    try:
        g.train_base(open("easyread_base_corpus.txt", encoding="utf-8").read())
    except Exception as e:
        st.error(f"베이스 코퍼스를 불러오지 못했어요: {e}")
    return g


def build_scene(text):
    """문장에서 위험/안전/사람 위치를 정해 장면(터레인 + 사람 좌표) 생성."""
    terrain = np.zeros((N, N), dtype=int)
    people = []          # 사람 좌표 리스트
    danger = None        # 위험 중심
    safe = None          # 안전 칸
    found = []
    for kw, (code, label) in KEYWORDS.items():
        if kw in text:
            found.append((kw, label, code))
            if code == 4:
                terrain[N//2-1:N//2+1, N//2-1:N//2+1] = 4
                danger = (N//2, N//2)
            elif code == 3:
                terrain[-2:, -2:] = 3
                safe = (N-1, N-1)
            elif code == 1:
                terrain[N//2, :] = np.where(terrain[N//2, :] == 0, 1, terrain[N//2, :])
            elif code == 2:
                people.append([1, 1])
    # 위험이 있는데 사람이 없으면, 사람 한 명을 위험 근처에 자동 배치(보여주기용)
    if danger and not people:
        people.append([N//2 + 1, N//2 - 2])
    return terrain, people, danger, safe, found


def step(people, danger, safe):
    """한 걸음: 사람은 위험에서 멀어지고 안전한 곳으로 한 칸 이동."""
    if not people:
        return people
    tx, ty = safe if safe else (N-1, N-1)
    new = []
    for (x, y) in people:
        # 안전 칸 방향으로 한 칸
        nx = x + (1 if tx > x else -1 if tx < x else 0)
        ny = y + (1 if ty > y else -1 if ty < y else 0)
        nx = max(0, min(N-1, nx)); ny = max(0, min(N-1, ny))
        new.append([nx, ny])
    return new


def draw(terrain, people, safe):
    fig, ax = plt.subplots(figsize=(7, 7))
    for x in range(N):
        for y in range(N):
            ax.add_patch(plt.Rectangle((y, N-1-x), 1, 1,
                         facecolor=COLORS[terrain[x, y]], edgecolor="white", lw=2))
    # 사람 그리기 (파란 동그라미)
    for (x, y) in people:
        at_safe = safe and (x, y) == safe
        ax.add_patch(plt.Circle((y+0.5, N-1-x+0.5), 0.32,
                     facecolor=("#2E7D32" if at_safe else "#1565C0"),
                     edgecolor="white", lw=2, zorder=5))
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    legend = [Patch(facecolor=COLORS[c], label=LABELS[c]) for c in [1, 2, 3, 4]]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=4, fontsize=15, frameon=False)
    fig.tight_layout()
    return fig


def simplify_with_llm(text, g, api_key, model, max_attempts=4):
    """LLM으로 쉽게 바꾸기. 판정은 자체 엔진(g)으로, PASS까지 반복."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    hint = ", ".join(list(g.student_words)[:40]) if g.student_words else ""
    sys = ("당신은 발달장애 학생을 위한 '쉬운 말' 변환기입니다.\n"
           "규칙: 1) 짧은 문장, 쉬운 단어, 한 번에 한 가지 행동만.\n"
           "2) 아래 학생이 쓰는 단어를 최대한 사용.\n"
           "3) 뜻은 그대로, 표현만 쉽게. 새 정보 더하지 말 것.\n"
           f"[학생이 아는 단어] {hint}")
    cur = text; history = []
    for i in range(1, max_attempts+1):
        r = g.evaluate(cur)
        if r["verdict"] == "PASS":
            break
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.3,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": f"다음 문장을 더 쉽게 바꿔주세요.\n문장: {cur}"}])
            cur = resp.choices[0].message.content.strip()
        except Exception as e:
            return {"text": cur, "error": str(e), "history": history}
        history.append((i, cur, g.evaluate(cur)["verdict"]))
    return {"text": cur, "error": None, "history": history}


# ── 상태 초기화 ─────────────────────────────────────────
if "people" not in st.session_state:
    st.session_state.people = None
    st.session_state.scene_text = ""

# ── 화면 ────────────────────────────────────────────────
st.markdown("<h1 style='font-size:40px'>🗺️ 쉬운 말 그림판</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:22px;color:#555'>상황을 쉬운 말로 적으면 그림으로 보여줘요. "
            "▶ 버튼을 누르면 사람이 안전한 곳으로 움직여요.</p>", unsafe_allow_html=True)

g = load_guardrail()

with st.sidebar:
    st.markdown("### 학생이 잘 아는 말 (선택)")
    sw = st.text_area("한 줄에 하나씩", height=100, placeholder="예)\n버스\n정류장")
    if sw.strip():
        g.set_student_words(" ".join(sw.split()))
        st.success(f"학생 단어 {len(g.student_words)}개 반영")
    st.markdown("---")
    st.markdown("### 쉽게 바꾸기 (LLM, 선택)")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.caption("키를 넣으면 '쉽게 바꾸기' 버튼이 켜져요. 판정은 LLM 없이 자체 엔진이 해요.")

text = st.text_input("상황 적기", value="불이 나면 밖으로 천천히 걸어 나가요",
                     label_visibility="collapsed")

# 장면이 바뀌면 사람 위치 리셋
if text != st.session_state.scene_text:
    terrain, people, danger, safe, found = build_scene(text)
    st.session_state.people = people
    st.session_state.scene_text = text
    st.session_state.safe = safe

terrain, _people0, danger, safe, found = build_scene(text)

col1, col2 = st.columns([1, 1])
with col1:
    if text.strip():
        r = g.evaluate(text)
        v = r["verdict"]
        box = {"PASS": ("#9BE3A2", "🟢 쉬운 말이에요!"),
               "WARNING": ("#FFE08A", "🟡 조금 어려울 수 있어요"),
               "FATAL": ("#FF8A80", "🔴 어려운 말이에요. 더 쉽게 바꿔볼까요?")}[v]
        st.markdown(f"<div style='background:{box[0]};padding:24px;border-radius:16px'>"
                    f"<span style='font-size:30px'>{box[1]}</span></div>", unsafe_allow_html=True)
        st.caption(f"점수(logP): {r['avg_logp']:.2f} · 기준 PASS≥{g.pass_thr} / FATAL<{g.fatal_thr}")

        # 쉽게 바꾸기 (LLM 옵션)
        if v != "PASS" and api_key:
            if st.button("✨ 쉽게 바꾸기", use_container_width=True):
                with st.spinner("쉬운 말로 바꾸는 중..."):
                    out = simplify_with_llm(text, g, api_key, model)
                if out["error"]:
                    st.error(f"LLM 오류: {out['error']}")
                else:
                    st.markdown(f"<div style='background:#E3F2FD;padding:20px;border-radius:14px'>"
                                f"<b style='font-size:22px'>바꾼 말:</b><br>"
                                f"<span style='font-size:26px'>{out['text']}</span></div>",
                                unsafe_allow_html=True)
        elif v != "PASS" and not api_key:
            st.caption("← 사이드바에 API Key를 넣으면 '쉽게 바꾸기'를 쓸 수 있어요.")

with col2:
    if found:
        st.markdown("**찾은 그림:** " + "  ".join(f"{l}" for _, l, _ in found))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ 한 칸 움직이기", use_container_width=True):
            st.session_state.people = step(st.session_state.people, danger, safe)
    with c2:
        if st.button("↺ 처음으로", use_container_width=True):
            _, p, _, _, _ = build_scene(text)
            st.session_state.people = p

if text.strip():
    st.pyplot(draw(terrain, st.session_state.people or [], safe))

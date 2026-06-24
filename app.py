"""
app.py — 발달장애 학생 맞춤 인지 가드레일 (교사용 Streamlit UI)
================================================================
학생 한 명의 인지 세계(텍스트 + 그림 키워드)를 코퍼스로 학습하고,
재난 안내 같은 외부 정보가 그 학생이 이해할 수 있는 범위 안인지
도메인 이탈(=너무 어려움/낯섬)로 검증한다.

설계 원칙(연구 버전):
  - 텍스트/그림 각각 선택적. 둘 다 없어도 프로필 성립.
  - 그림은 직접 임베딩하지 않고 '키워드 텍스트'로 흡수(CPU 경량 유지).
  - 판정/가드레일은 자체 엔진(CoreAIv2Engine). 외부 LLM 불필요.
"""
import streamlit as st
import pickle, io, time

st.set_page_config(
    page_title="학생 맞춤 인지 가드레일",
    page_icon="🧩",
    layout="wide",
)

# ── 엔진/프로필 로드 ───────────────────────────────────────────
try:
    from student_profile import StudentProfile
    OK = True
except Exception as e:
    OK = False
    st.error(f"모듈 로드 실패: {e}")
    st.stop()

# ── 스타일 (차분한 톤) ─────────────────────────────────────────
st.markdown("""
<style>
.main-title { font-size:1.6rem; font-weight:700; color:#3b6ea5; }
.sub { color:#6b7280; font-size:0.9rem; }
.pill { display:inline-block; padding:2px 10px; border-radius:12px;
        font-size:0.75rem; font-weight:600; margin-right:4px; }
.pill-ok { background:#e6f4ea; color:#188038; }
.pill-no { background:#f1f3f4; color:#9aa0a6; }
.verdict-pass { color:#188038; font-weight:700; }
.verdict-warn { color:#b06f00; font-weight:700; }
.verdict-fatal{ color:#c5221f; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── 세션 ───────────────────────────────────────────────────────
if "profile" not in st.session_state:
    st.session_state.profile = None

def P():
    return st.session_state.profile

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 사이드바: 프로필 만들기 / 불러오기 / 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🧩 학생 프로필")
    st.caption("학생 한 명 = 프로필 하나")

    st.markdown("### 새 프로필")
    sid = st.text_input("학생 ID", value="", placeholder="예: s001")
    sname = st.text_input("이름(선택)", value="")
    if st.button("➕ 새로 만들기", use_container_width=True, disabled=not sid.strip()):
        st.session_state.profile = StudentProfile(sid.strip(), sname.strip())
        st.rerun()

    st.markdown("---")
    st.markdown("### 저장된 프로필 불러오기")
    up = st.file_uploader("프로필(.pkl)", type=None, label_visibility="collapsed")
    if up and up.name.endswith(".pkl"):
        if st.button("📂 불러오기", use_container_width=True):
            try:
                data = pickle.loads(up.read())
                st.session_state.profile = StudentProfile.from_dict(data)
                st.rerun()
            except Exception as e:
                st.error(f"불러오기 실패: {e}")

    # 저장
    if P() is not None:
        st.markdown("---")
        data = pickle.dumps(P().to_dict())
        st.download_button(
            "💾 프로필 저장",
            data=data,
            file_name=f"student_{P().student_id}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown('<div class="main-title">발달장애 학생 맞춤 인지 가드레일</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub">학생이 이해할 수 있는 범위를 학습하고, '
            '외부 안내문이 그 범위 안인지(=너무 어렵지 않은지) 검증합니다.</div>',
            unsafe_allow_html=True)
st.markdown("")

if P() is None:
    st.info("왼쪽에서 새 프로필을 만들거나, 저장된 프로필을 불러오세요.")
    st.markdown("""
**이 도구가 하는 일**

- 학생이 평소 쓰는 말·반향어(텍스트)와 그림에서 뽑은 키워드를 모아
  그 학생만의 '이해 가능 범위'를 코퍼스로 학습합니다.
- 재난 안내문 같은 외부 문장을 넣으면, 그 범위 안인지(PASS) /
  경계(WARNING) / 너무 벗어났는지(FATAL)를 판정합니다.
- 텍스트와 그림 키워드는 각각 선택입니다. 한쪽만 있어도 됩니다.
    """)
    st.stop()

# 프로필 요약
s = P().summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("학생", s["name"] or s["student_id"])
c2.metric("텍스트 조각", s["텍스트 조각"])
c3.metric("그림 키워드 묶음", s["그림 키워드 묶음"])
c4.metric("학습됨", "예" if s["학습됨"] else "아니오")

t_ok = '<span class="pill pill-ok">텍스트 있음</span>' if s["텍스트 있음"] else '<span class="pill pill-no">텍스트 없음</span>'
i_ok = '<span class="pill pill-ok">그림 있음</span>' if s["그림 있음"] else '<span class="pill pill-no">그림 없음</span>'
st.markdown(t_ok + i_ok, unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["✍️ 입력 추가", "🎓 학습", "🔍 안내문 검증"])

# ── 탭1: 입력 추가 (각각 선택적) ───────────────────────────────
with tab1:
    cL, cR = st.columns(2)
    with cL:
        st.markdown("#### 텍스트 (선택)")
        st.caption("학생이 쓰는 말, 반향어, 교사 기록 등")
        txt = st.text_area("문장", height=100, label_visibility="collapsed",
                           placeholder="예: 불이 나면 선생님 손을 잡아요")
        src1 = st.text_input("출처", value="교사기록", key="src1")
        if st.button("텍스트 추가", use_container_width=True, disabled=not txt.strip()):
            P().add_text(txt, src1); st.rerun()

    with cR:
        st.markdown("#### 그림 키워드 (선택)")
        st.caption("학생 그림에서 뽑은 키워드 (예: 버스 노랑 반복)")
        kw = st.text_area("키워드", height=100, label_visibility="collapsed",
                         placeholder="예: 버스 노랑 반복")
        src2 = st.text_input("출처", value="그림", key="src2")
        if st.button("그림 키워드 추가", use_container_width=True, disabled=not kw.strip()):
            P().add_image_keywords(kw, src2); st.rerun()

    st.markdown("---")
    st.markdown("#### 현재 코퍼스")
    if P().is_empty():
        st.caption("아직 비어 있어요. 텍스트나 그림 키워드를 추가하세요. (둘 다 없어도 프로필은 유지됩니다.)")
    else:
        if P().has_text():
            st.markdown("**텍스트**")
            for e in P().text_entries:
                st.markdown(f"- {e['text']}  <span class='sub'>({e['source']})</span>",
                            unsafe_allow_html=True)
        if P().has_image():
            st.markdown("**그림 키워드**")
            for e in P().image_entries:
                st.markdown(f"- {' '.join(e['keywords'])}  <span class='sub'>({e['source']})</span>",
                            unsafe_allow_html=True)

# ── 탭2: 학습 ─────────────────────────────────────────────────
with tab2:
    st.markdown("#### 코퍼스 학습")
    st.caption("선택한 소스만으로 학습합니다. 코퍼스가 비면 학습되지 않습니다.")
    inc_t = st.checkbox("텍스트 포함", value=P().has_text(), disabled=not P().has_text())
    inc_i = st.checkbox("그림 키워드 포함", value=P().has_image(), disabled=not P().has_image())

    if st.button("🎓 학습 시작", type="primary", use_container_width=True,
                 disabled=P().is_empty()):
        with st.spinner("학습 중..."):
            prog = st.progress(0)
            ok = P().train(include_text=inc_t, include_image=inc_i,
                           on_progress=lambda p, m: prog.progress(int(p)))
        if ok:
            st.success("학습 완료! '안내문 검증' 탭에서 확인하세요.")
        else:
            st.warning("코퍼스가 비어 학습하지 못했어요.")
        st.rerun()

    if s["학습됨"]:
        st.info("이미 학습된 상태입니다. 입력을 추가하면 다시 학습하세요.")

# ── 탭3: 안내문 검증 ──────────────────────────────────────────
with tab3:
    st.markdown("#### 안내문이 이 학생에게 적절한가?")
    st.caption("재난 안내문 등을 넣으면, 학생 이해 범위 기준으로 판정합니다.")
    if not s["학습됨"]:
        st.info("먼저 '학습' 탭에서 학습해주세요.")
    else:
        test = st.text_area("검증할 문장", height=90,
                            placeholder="예: 금일 화재 발생, 비상구로 신속히 대피 바람")
        if st.button("🔍 검증", type="primary", use_container_width=True,
                     disabled=not test.strip()):
            r = P().evaluate(test.strip())
            v = r.get("verdict", "SKIP")
            cls = {"PASS":"verdict-pass","WARNING":"verdict-warn",
                   "FATAL":"verdict-fatal"}.get(v,"")
            label = {
                "PASS":"✅ 이해 범위 안 — 이 학생에게 적절",
                "WARNING":"🟡 경계 — 일부 어려울 수 있음",
                "FATAL":"🔴 범위 밖 — 너무 어렵거나 낯섦, 쉽게 바꿔야 함",
                "SKIP":"⏭ 판정 불가",
            }.get(v, v)
            st.markdown(f"<div class='{cls}' style='font-size:1.2rem;'>{label}</div>",
                        unsafe_allow_html=True)
            if "logp" in r:
                st.caption(f"점수(logP): {r.get('logp', 0):+.2f}")
            st.caption("FATAL/WARNING이면, 이 학생의 쉬운 말로 다시 쓰는 게 좋아요.")

"""
app.py — 발달장애 학생 맞춤 인지 가드레일 (교사용 Streamlit UI)
================================================================
구성:
  1) 학생 프로필 (텍스트/그림 키워드 각각 선택)
  2) 전문 지식 코퍼스 (안전 매뉴얼/특수교육학 개론 등) — 학생과 분리
  3) 이중 가드레일: 전문(정확성) + 학생(이해수준) 각각 판정
  4) 채팅: OpenAI로 전문 기반 응답 + 너무 어려우면 쉬운 말 재작성
     (판정=자체 엔진, 생성=LLM. LLM은 판단을 대신하지 않음)
"""
import streamlit as st
import pickle, time

st.set_page_config(page_title="학생 맞춤 인지 가드레일", page_icon="🧩", layout="wide")

try:
    from student_profile import StudentProfile
    from dual_guardrail import DualGuardrail
    from llm_bridge import LLMBridge
except Exception as e:
    st.error(f"모듈 로드 실패: {e}")
    st.stop()

st.markdown("""
<style>
.main-title{font-size:1.6rem;font-weight:700;color:#3b6ea5;}
.sub{color:#6b7280;font-size:0.9rem;}
.v-pass{color:#188038;font-weight:700;}
.v-warn{color:#b06f00;font-weight:700;}
.v-fatal{color:#c5221f;font-weight:700;}
.bubble-u{text-align:right;margin:6px 0;}
.bubble-u span{background:#dbe7f3;padding:8px 12px;border-radius:14px 14px 4px 14px;display:inline-block;max-width:80%;}
.bubble-a{text-align:left;margin:6px 0;}
.bubble-a span{background:#f1f3f4;padding:8px 12px;border-radius:14px 14px 14px 4px;display:inline-block;max-width:80%;}
</style>
""", unsafe_allow_html=True)

# ── 세션 ──
ss = st.session_state
ss.setdefault("profile", None)
ss.setdefault("guard", DualGuardrail())
ss.setdefault("api_key", "")
ss.setdefault("model", "gpt-4o-mini")
ss.setdefault("chat", [])

# ── 사이드바 ──
with st.sidebar:
    st.markdown("## 🧩 설정")
    ss.api_key = st.text_input("OpenAI API Key", value=ss.api_key, type="password",
                               help="채팅 응답/쉬운말 변환에 사용. 검증만 할 땐 없어도 됩니다.")
    ss.model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o"],
                            index=["gpt-4o-mini","gpt-4o"].index(ss.model))
    st.markdown("---")
    st.markdown("### 학생 프로필")
    sid = st.text_input("학생 ID", placeholder="예: s001")
    sname = st.text_input("이름(선택)")
    if st.button("➕ 새 프로필", use_container_width=True, disabled=not sid.strip()):
        ss.profile = StudentProfile(sid.strip(), sname.strip()); st.rerun()
    up = st.file_uploader("학생 프로필(.pkl)", type=None, label_visibility="collapsed")
    if up and up.name.endswith(".pkl") and st.button("📂 프로필 불러오기", use_container_width=True):
        try:
            ss.profile = StudentProfile.from_dict(pickle.loads(up.read())); st.rerun()
        except Exception as e: st.error(e)
    if ss.profile is not None:
        st.download_button("💾 프로필 저장", data=pickle.dumps(ss.profile.to_dict()),
                           file_name=f"student_{ss.profile.student_id}.pkl",
                           use_container_width=True)
    st.markdown("---")
    gup = st.file_uploader("가드레일 전체(.pkl)", type=None, key="gpkl", label_visibility="collapsed")
    if gup and gup.name.endswith(".pkl") and st.button("📂 가드레일 불러오기", use_container_width=True):
        try:
            ss.guard = DualGuardrail.from_dict(pickle.loads(gup.read())); st.rerun()
        except Exception as e: st.error(e)
    st.download_button("💾 가드레일 저장", data=pickle.dumps(ss.guard.to_dict()),
                       file_name="dual_guardrail.pkl", use_container_width=True)

st.markdown('<div class="main-title">발달장애 학생 맞춤 인지 가드레일</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">전문 지식(정확성) + 학생 이해수준, 두 가드레일로 따로 판정합니다.</div>', unsafe_allow_html=True)

g = ss.guard
gs = g.summary()
c1,c2,c3,c4 = st.columns(4)
c1.metric("전문자료 수", gs["전문자료 수"])
c2.metric("전문 학습", "예" if gs["전문 학습됨"] else "아니오")
c3.metric("학생", gs["학생"])
c4.metric("학생 학습", "예" if gs["학생 학습됨"] else "아니오")
st.markdown("---")

tabs = st.tabs(["✍️ 학생 입력", "📚 전문 코퍼스", "🎓 학습", "💬 채팅", "🔍 검증"])

# ── 탭1: 학생 입력 ──
with tabs[0]:
    if ss.profile is None:
        st.info("왼쪽에서 학생 프로필을 먼저 만들거나 불러오세요.")
    else:
        cL,cR = st.columns(2)
        with cL:
            st.markdown("#### 텍스트 (선택)")
            txt = st.text_area("문장", height=90, label_visibility="collapsed",
                               placeholder="예: 불이 나면 선생님 손을 잡아요")
            if st.button("텍스트 추가", disabled=not txt.strip(), use_container_width=True):
                ss.profile.add_text(txt, "교사기록"); st.rerun()
        with cR:
            st.markdown("#### 그림 키워드 (선택)")
            kw = st.text_area("키워드", height=90, label_visibility="collapsed",
                              placeholder="예: 버스 노랑 반복")
            if st.button("그림 키워드 추가", disabled=not kw.strip(), use_container_width=True):
                ss.profile.add_image_keywords(kw, "그림"); st.rerun()
        st.markdown("---")
        s = ss.profile.summary()
        st.caption(f"텍스트 {s['텍스트 조각']}개 · 그림 키워드 {s['그림 키워드 묶음']}개")
        if ss.profile.has_text():
            for e in ss.profile.text_entries: st.markdown(f"- {e['text']}")
        if ss.profile.has_image():
            for e in ss.profile.image_entries: st.markdown(f"- 🎨 {' '.join(e['keywords'])}")

# ── 탭2: 전문 코퍼스 ──
with tabs[1]:
    st.markdown("#### 전문 지식 자료 (안전 매뉴얼 / 특수교육학 개론 등)")
    st.caption("학생 코퍼스와 분리됩니다. 여기 자료로 '정확성'을 판정합니다.")
    ex_up = st.file_uploader("자료 업로드 (txt)", type=["txt"], key="exup")
    name = st.text_input("자료 이름", value="전문자료")
    if ex_up and st.button("자료 추가", use_container_width=True):
        text = ex_up.read().decode("utf-8", errors="ignore")
        g.add_expert_text(text, name); st.success(f"'{name}' 추가됨"); st.rerun()
    paste = st.text_area("또는 직접 붙여넣기", height=120)
    if paste.strip() and st.button("붙여넣은 자료 추가", use_container_width=True):
        g.add_expert_text(paste, name); st.rerun()
    if g.expert_sources:
        st.markdown("**등록된 전문자료**")
        for e in g.expert_sources:
            st.markdown(f"- {e['name']} ({e['chars']}자)")

# ── 탭3: 학습 ──
with tabs[2]:
    st.markdown("#### 학습")
    cA,cB = st.columns(2)
    with cA:
        st.markdown("**전문 지식**")
        if st.button("📚 전문 코퍼스 학습", use_container_width=True,
                     disabled=not g.expert_corpus.strip()):
            with st.spinner("전문 학습 중..."):
                g.train_expert(); st.success("전문 학습 완료"); st.rerun()
    with cB:
        st.markdown("**학생 프로필**")
        if st.button("🧩 학생 코퍼스 학습", use_container_width=True,
                     disabled=ss.profile is None or ss.profile.is_empty()):
            with st.spinner("학생 학습 중..."):
                g.set_student(ss.profile); g.train_student()
                st.success("학생 학습 완료"); st.rerun()

# ── 탭4: 채팅 ──
with tabs[3]:
    st.markdown("#### 상황을 입력하면, 전문 지식 기반으로 이 학생에 맞게 답합니다")
    if not ss.api_key:
        st.warning("왼쪽에 OpenAI API Key를 입력하세요.")
    elif not g.is_ready():
        st.info("먼저 '학습' 탭에서 전문/학생 코퍼스를 학습하세요.")
    else:
        for m in ss.chat[-10:]:
            if m["role"]=="user":
                st.markdown(f'<div class="bubble-u"><span>{m["text"]}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bubble-a"><span>{m["text"]}</span></div>', unsafe_allow_html=True)
                st.caption(m.get("meta",""))
        msg = st.text_area("상황 입력", height=80, key="chatin",
                           placeholder="예: 지금 불이 났어요. 이 학생에게 어떻게 안내하죠?")
        if st.button("보내기", type="primary", disabled=not msg.strip()):
            bridge = LLMBridge(g, ss.api_key, ss.model)
            with st.spinner("생성 중..."):
                r = bridge.respond(msg.strip())
            ss.chat.append({"role":"user","text":msg.strip()})
            meta = f"정확성:{r.get('expert')} · 이해도:{r.get('student')} · {r.get('attempts')}회"
            ss.chat.append({"role":"assistant","text":r["answer"],"meta":meta})
            st.rerun()

# ── 탭5: 검증 (+쉬운말 변환) ──
with tabs[4]:
    st.markdown("#### 안내문 검증 + 쉬운 말 변환")
    if not g.is_ready():
        st.info("먼저 학습하세요.")
    else:
        t = st.text_area("검증할 안내문", height=80,
                         placeholder="예: 금일 화재 발생, 비상구로 신속히 대피 바람")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("🔍 검증만", use_container_width=True, disabled=not t.strip()):
                v = g.evaluate(t.strip())
                def fmt(x):
                    cls={"PASS":"v-pass","WARNING":"v-warn","FATAL":"v-fatal"}.get(x,"")
                    return f"<span class='{cls}'>{x}</span>"
                st.markdown(f"정확성(전문): {fmt(v['expert']['verdict'])}", unsafe_allow_html=True)
                st.markdown(f"이해도(학생): {fmt(v['student']['verdict'])}", unsafe_allow_html=True)
                st.markdown(f"종합: {fmt(v['overall'])}", unsafe_allow_html=True)
                if v["needs_simplify"]:
                    st.caption("→ 이 학생에겐 어려워요. '쉬운 말로 변환'을 눌러보세요.")
        with c2:
            if st.button("✨ 쉬운 말로 변환", use_container_width=True,
                         disabled=not t.strip() or not ss.api_key):
                bridge = LLMBridge(g, ss.api_key, ss.model)
                with st.spinner("변환 중..."):
                    r = bridge.simplify(t.strip())
                st.success("변환 결과")
                st.markdown(f"**{r['answer']}**")
                st.caption(f"이해도:{r.get('student')} · {r.get('attempts')}회 시도")

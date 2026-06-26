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

tabs = st.tabs(["✍️ 학생 입력", "📚 전문 코퍼스", "🎓 학습", "🧒 학생용 자료 만들기", "👩‍🏫 교사 상담"])

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
        # ── 학생 그림 업로드 → 쉬운 말 변환(GPT-4V) → 선생님 확인/수정 ──
        st.markdown("#### 🎨 학생 그림 올리기 (jpg / png / pdf)")
        st.caption("그림을 올리면 쉬운 말로 바꿔 보여드려요. 선생님이 확인·수정한 뒤 추가하세요. "
                   "(그림 해석에만 GPT-4V를 써요. API Key 필요)")
        img_up = st.file_uploader("그림 파일", type=["jpg", "jpeg", "png", "pdf"],
                                  key="imgup", label_visibility="collapsed")
        if img_up:
            if img_up.name.lower().endswith(("jpg", "jpeg", "png")):
                st.image(img_up, width=260)
            if not ss.api_key:
                st.warning("그림을 쉬운 말로 바꾸려면 왼쪽에 OpenAI API Key를 넣어주세요.")
            elif st.button("🔎 그림을 쉬운 말로 바꾸기", use_container_width=True):
                from file_ingest import image_to_easy_text
                with st.spinner("그림을 읽는 중..."):
                    try:
                        ss["img_draft"] = image_to_easy_text(
                            img_up.name, img_up.getvalue(), ss.api_key, "gpt-4o")
                    except Exception as e:
                        st.error(f"그림 해석 실패: {e}")
        # 변환 결과: 선생님이 확인·수정 후 추가
        if ss.get("img_draft"):
            st.markdown("**확인·수정 (그림 해석이 틀렸으면 고쳐주세요)**")
            edited = st.text_area("쉬운 말 설명", value=ss["img_draft"], height=120,
                                  key="img_edit")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("✅ 이 설명을 학생 코퍼스에 추가", use_container_width=True):
                    ss.profile.add_text(edited, "그림설명")
                    ss["img_draft"] = ""; st.success("추가했어요"); st.rerun()
            with cc2:
                if st.button("🗑️ 버리기", use_container_width=True):
                    ss["img_draft"] = ""; st.rerun()
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
    ex_up = st.file_uploader("자료 업로드 (txt / pdf / docx)",
                             type=["txt", "pdf", "docx"], key="exup")
    name = st.text_input("자료 이름", value="전문자료")
    if ex_up and st.button("자료 추가", use_container_width=True):
        from file_ingest import extract_text
        try:
            text = extract_text(ex_up.name, ex_up.read())
            if not text.strip():
                st.warning("파일에서 글자를 찾지 못했어요(스캔본일 수 있어요).")
            else:
                g.add_expert_text(text, name)
                st.success(f"'{name}' 추가됨 ({len(text)}자)"); st.rerun()
        except Exception as e:
            st.error(f"추출 실패: {e}")
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

# ── 탭4: 학생용 자료 만들기 ──
with tabs[3]:
    st.markdown("#### 수업 내용을 넣으면, 이 학생이 이해할 쉬운 자료로 만들어요")
    st.caption("입력 → 자체 가드레일로 '이 학생에게 쉬운가' 판정 → 어려우면 쉬운 말로 변환")
    if not g.is_ready():
        st.info("먼저 '학습' 탭에서 전문/학생 코퍼스를 학습하세요.")
    else:
        src = st.text_area("수업 내용 / 안내문", height=110,
                           placeholder="예: 화재가 발생하면 비상구를 통해 신속히 대피한다")
        make = st.button("🧒 학생용 자료 만들기", type="primary",
                         disabled=not src.strip(), use_container_width=True)
        if make:
            v = g.evaluate(src.strip())
            def fmt(x):
                cls = {"PASS":"v-pass","WARNING":"v-warn","FATAL":"v-fatal"}.get(x,"")
                return f"<span class='{cls}'>{x}</span>"
            st.markdown(f"정확성(전문): {fmt(v['expert']['verdict'])} · "
                        f"이해도(학생): {fmt(v['student']['verdict'])}", unsafe_allow_html=True)
            if v["student"]["verdict"] == "PASS":
                st.success("이 학생이 이해할 수 있는 쉬운 자료예요.")
                st.markdown(f"<div style='background:#eef7ee;padding:18px;border-radius:12px;"
                            f"font-size:22px'>{src.strip()}</div>", unsafe_allow_html=True)
            elif not ss.api_key:
                st.warning("이 학생에겐 어려워요. 쉬운 말 변환에 OpenAI API Key가 필요해요.")
            else:
                bridge = LLMBridge(g, ss.api_key, ss.model)
                with st.spinner("이 학생에 맞는 쉬운 말로 바꾸는 중..."):
                    r = bridge.simplify(src.strip())
                st.success("학생용 쉬운 자료")
                st.markdown(f"<div style='background:#eef7ee;padding:18px;border-radius:12px;"
                            f"font-size:24px;line-height:1.6'>{r['answer']}</div>",
                            unsafe_allow_html=True)
                st.caption(f"이해도:{r.get('student')} · {r.get('attempts')}회 변환")

# ── 탭5: 교사 상담 ──
with tabs[4]:
    st.markdown("#### 교육학 자료를 근거로, 이 학생 수업에 대해 자세히 상담해요")
    st.caption("전문(교육학) 코퍼스 기반 답변. 교사용이라 분량을 넉넉히 보여줘요.")
    if not ss.api_key:
        st.warning("왼쪽에 OpenAI API Key를 입력하세요.")
    elif g.expert_engine is None:
        st.info("먼저 '학습' 탭에서 전문(교육학) 코퍼스를 학습하세요.")
    else:
        for m in ss.chat[-12:]:
            if m["role"] == "user":
                st.markdown(f'<div class="bubble-u"><span>{m["text"]}</span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bubble-a"><span>{m["text"]}</span></div>',
                            unsafe_allow_html=True)
                if m.get("meta"):
                    st.caption(m["meta"])
        msg = st.text_area("질문", height=90, key="t_chatin",
                           placeholder="예: 이 학생에게 화재 대피를 가르칠 때 어떤 교수법이 좋을까요? "
                                       "단계별로 알려주세요.")
        col_a, col_b = st.columns([3, 1])
        with col_b:
            length = st.selectbox("분량", ["보통", "길게", "아주 길게"], index=1,
                                  label_visibility="collapsed")
        with col_a:
            send = st.button("보내기", type="primary", disabled=not msg.strip(),
                             use_container_width=True)
        if send:
            mt = {"보통": 1000, "길게": 1800, "아주 길게": 2800}[length]
            bridge = LLMBridge(g, ss.api_key, ss.model)
            with st.spinner("교육학 자료를 근거로 정리하는 중..."):
                r = bridge.teacher_answer(msg.strip(), max_tokens=mt)
            ss.chat.append({"role": "user", "text": msg.strip()})
            ss.chat.append({"role": "assistant", "text": r["answer"],
                            "meta": f"정확성(전문자료 근거): {r.get('expert')}"})
            st.rerun()
        if ss.chat and st.button("🗑️ 대화 비우기"):
            ss.chat = []; st.rerun()

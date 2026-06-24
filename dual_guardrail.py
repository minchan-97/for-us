"""
dual_guardrail.py — 이중 코퍼스 가드레일
==========================================
전문 지식 코퍼스와 학생 프로필 코퍼스를 '분리'해서 각각 판정한다.

  - 전문 가드레일(expert) : 안전 매뉴얼/특수교육학 개론 등.
                            "답변이 그 분야의 정확한 내용인가"를 본다.
  - 학생 가드레일(student): 학생 텍스트/그림 키워드.
                            "이 학생이 이해할 수준인가"를 본다.

두 코퍼스를 섞지 않는 이유:
  전문 매뉴얼은 어려운 전문용어투성이라, 학생 코퍼스에 섞으면
  '쉬움' 판정이 망가진다. 분리해야 정확성과 이해수준을 각각 검사 가능.

판정/가드레일은 자체 엔진(CoreAIv2Engine). 생성은 외부 LLM(선택).
이 모듈 자체는 LLM을 호출하지 않는다(판정 전담).
"""
from __future__ import annotations
import pickle, time
from typing import Optional


class DualGuardrail:
    def __init__(self):
        self.expert_engine  = None   # 전문 지식 엔진
        self.student_engine = None   # 학생 프로필 엔진
        self.expert_corpus  = ""     # 합쳐진 전문 코퍼스 텍스트
        self.student_corpus = ""     # 합쳐진 학생 코퍼스 텍스트
        self.expert_sources:  list = []   # [{"name":..., "ts":...}]
        self.student_id = ""
        self.student_name = ""

    # ── 전문 지식 코퍼스 추가 ─────────────────────────────────
    def add_expert_text(self, text: str, source_name: str = "전문자료"):
        """안전 매뉴얼/특수교육학 개론 등 전문 지식 텍스트 추가."""
        text = (text or "").strip()
        if not text:
            return False
        self.expert_corpus += ("\n" if self.expert_corpus else "") + text
        self.expert_sources.append({"name": source_name, "ts": time.time(),
                                    "chars": len(text)})
        return True

    # ── 학생 프로필 연결 ──────────────────────────────────────
    def set_student(self, student_profile):
        """
        student_profile.StudentProfile 객체를 받아 학생 코퍼스로 사용.
        (텍스트/그림 키워드 각각 선택적인 그 구조 그대로)
        """
        self.student_id   = student_profile.student_id
        self.student_name = student_profile.name
        self.student_corpus = student_profile.build_corpus()
        return bool(self.student_corpus.strip())

    def set_student_corpus_text(self, text: str, sid="", name=""):
        """프로필 객체 없이 학생 코퍼스 텍스트만 직접 넣고 싶을 때."""
        self.student_corpus = (text or "").strip()
        self.student_id = sid; self.student_name = name
        return bool(self.student_corpus.strip())

    # ── 학습 ──────────────────────────────────────────────────
    def train_expert(self, on_progress=None):
        if not self.expert_corpus.strip():
            self.expert_engine = None
            return False
        from core_ai_v2_engine import CoreAIv2Engine
        self.expert_engine = CoreAIv2Engine()
        self.expert_engine.corpus_name = "expert"
        self.expert_engine.train(self.expert_corpus, on_progress=on_progress)
        return True

    def train_student(self, on_progress=None):
        if not self.student_corpus.strip():
            self.student_engine = None
            return False
        from core_ai_v2_engine import CoreAIv2Engine
        self.student_engine = CoreAIv2Engine()
        self.student_engine.corpus_name = f"student:{self.student_id}"
        self.student_engine.train(self.student_corpus, on_progress=on_progress)
        return True

    def is_ready(self) -> bool:
        """최소 한 축이라도 학습돼야 판정 가능."""
        return self.expert_engine is not None or self.student_engine is not None

    # ── 이중 판정 ─────────────────────────────────────────────
    def evaluate(self, text: str) -> dict:
        """
        두 가드레일로 각각 판정.
        반환:
          expert  : 전문 정확성 판정 (PASS/WARNING/FATAL/SKIP)
          student : 학생 이해수준 판정 (PASS/WARNING/FATAL/SKIP)
          overall : 둘 중 더 나쁜 쪽 (둘 다 통과해야 진짜 통과)
        """
        order = {"PASS": 0, "WARNING": 1, "FATAL": 2, "SKIP": -1}

        exp = {"verdict": "SKIP"}
        if self.expert_engine is not None:
            r = self.expert_engine.evaluate(text)
            exp = {"verdict": r.get("verdict", "SKIP"),
                   "logp": r.get("logp", 0.0)}

        stu = {"verdict": "SKIP"}
        if self.student_engine is not None:
            r = self.student_engine.evaluate(text)
            stu = {"verdict": r.get("verdict", "SKIP"),
                   "logp": r.get("logp", 0.0)}

        # overall = 둘 중 더 나쁜 판정 (SKIP은 제외)
        cands = [v["verdict"] for v in (exp, stu) if v["verdict"] != "SKIP"]
        if not cands:
            overall = "SKIP"
        else:
            overall = max(cands, key=lambda s: order.get(s, 0))

        return {
            "expert":  exp,     # 정확성: 전문 도메인 안인가
            "student": stu,     # 이해도: 학생이 알아들을 수준인가
            "overall": overall, # 둘 다 통과해야 PASS
            "needs_simplify": stu["verdict"] in ("WARNING", "FATAL"),
            "needs_fix":      exp["verdict"] in ("WARNING", "FATAL"),
        }

    # ── 저장 / 로드 ───────────────────────────────────────────
    def to_dict(self) -> dict:
        def eng_dict(v2):
            if v2 is None: return None
            return {
                "n_clusters": v2.n_clusters, "global_vocab": v2.global_vocab,
                "corpus_name": getattr(v2, "corpus_name", ""),
                "train_stats": getattr(v2, "train_stats", {}),
                "emb_emb": v2.embedder.emb if v2.embedder else None,
                "emb_vocab": v2.embedder.vocab if v2.embedder else None,
                "emb_dim": v2.embedder.dim if v2.embedder else 32,
                "cluster_sentences": dict(v2.decomposer.cluster_sentences),
                "cluster_tokens": dict(v2.decomposer.cluster_tokens),
                "cluster_keywords": v2.decomposer.cluster_keywords,
                "decomp_vocab": v2.decomposer.vocab,
                "decomp_W": v2.decomposer.W,
                "markovs": {k: {"uni": dict(m.uni),
                                "bi": {k2: dict(vv) for k2, vv in m.bi.items()},
                                "tri": {k2: dict(vv) for k2, vv in m.tri.items()},
                                "total": m.total}
                            for k, m in v2.markovs.items()},
            }
        return {
            "expert_corpus":  self.expert_corpus,
            "student_corpus": self.student_corpus,
            "expert_sources": self.expert_sources,
            "student_id": self.student_id, "student_name": self.student_name,
            "expert_engine":  eng_dict(self.expert_engine),
            "student_engine": eng_dict(self.student_engine),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DualGuardrail":
        g = cls()
        g.expert_corpus  = data.get("expert_corpus", "")
        g.student_corpus = data.get("student_corpus", "")
        g.expert_sources = data.get("expert_sources", [])
        g.student_id     = data.get("student_id", "")
        g.student_name   = data.get("student_name", "")
        from core_ai_v2_engine import CoreAIv2Engine
        if data.get("expert_engine"):
            try: g.expert_engine = CoreAIv2Engine.load_from_dict(data["expert_engine"])
            except Exception: g.expert_engine = None
        if data.get("student_engine"):
            try: g.student_engine = CoreAIv2Engine.load_from_dict(data["student_engine"])
            except Exception: g.student_engine = None
        return g

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)
        return path

    @classmethod
    def load(cls, path: str) -> "DualGuardrail":
        with open(path, "rb") as f:
            return cls.from_dict(pickle.load(f))

    def summary(self) -> dict:
        return {
            "전문자료 수": len(self.expert_sources),
            "전문 학습됨": self.expert_engine is not None,
            "학생": self.student_name or self.student_id or "(없음)",
            "학생 학습됨": self.student_engine is not None,
        }

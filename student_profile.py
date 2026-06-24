"""
student_profile.py — 학생 프로필 코퍼스
========================================
한 학생의 인지 세계를 하나의 프로필로 묶는다.

설계 원칙:
  - 텍스트 코퍼스, 그림 코퍼스 각각 '선택적'이다.
  - 둘 다 없어도 프로필은 성립한다(빈 프로필 가능).
  - 한쪽만 있어도 작동한다. 나중에 다른 쪽을 추가할 수 있다.
  - 그림은 직접 임베딩하지 않고 '텍스트 키워드'로 흡수한다.
    (CPU 경량 · 텍스트 코퍼스 중심 정체성 유지)
  - 학습/판정은 기존 GasCore 엔진(CoreAIv2Engine)에 위임한다.

저장 단위: 학생 1명 = 프로필 1개 = pkl 1개
"""
from __future__ import annotations
import pickle, time
from typing import Optional


class StudentProfile:
    """
    학생 한 명의 코퍼스 묶음.
      text_entries  : 글/발화/반향어 등 텍스트 조각 리스트 (선택)
      image_entries : 그림에서 뽑은 텍스트 키워드 묶음 리스트 (선택)
    각 entry는 출처(source)와 함께 보관해 나중에 추적 가능.
    """

    def __init__(self, student_id: str, name: str = ""):
        self.student_id = student_id
        self.name = name
        # 각각 선택적 — 비어 있어도 됨
        self.text_entries:  list = []   # [{"text":..., "source":..., "ts":...}]
        self.image_entries: list = []   # [{"keywords":[...], "source":..., "ts":...}]
        self.created_ts = time.time()
        self.updated_ts = time.time()
        # 학습된 엔진(있으면). 코퍼스가 비면 None.
        self.engine = None
        self.is_trained = False

    # ── 입력 추가 (각각 독립적, 언제든 추가 가능) ──────────────
    def add_text(self, text: str, source: str = "manual"):
        """텍스트 조각 추가 (글/전사 발화/반향어 등)."""
        text = (text or "").strip()
        if not text:
            return False
        self.text_entries.append({
            "text": text, "source": source, "ts": time.time(),
        })
        self.updated_ts = time.time()
        self.is_trained = False   # 코퍼스 바뀌면 재학습 필요 표시
        return True

    def add_image_keywords(self, keywords, source: str = "manual"):
        """
        그림에서 뽑은 키워드 추가.
        keywords: 리스트(["버스","노랑","반복"]) 또는 공백/쉼표 구분 문자열.
        그림 자체가 아니라 '의미를 텍스트로' 흡수한다.
        """
        if isinstance(keywords, str):
            kws = [k.strip() for k in keywords.replace(",", " ").split() if k.strip()]
        else:
            kws = [str(k).strip() for k in keywords if str(k).strip()]
        if not kws:
            return False
        self.image_entries.append({
            "keywords": kws, "source": source, "ts": time.time(),
        })
        self.updated_ts = time.time()
        self.is_trained = False
        return True

    # ── 코퍼스 합성 ───────────────────────────────────────────
    def has_text(self) -> bool:
        return len(self.text_entries) > 0

    def has_image(self) -> bool:
        return len(self.image_entries) > 0

    def is_empty(self) -> bool:
        return not self.has_text() and not self.has_image()

    def build_corpus(self, include_text: bool = True,
                     include_image: bool = True) -> str:
        """
        선택된 소스만 합쳐 하나의 텍스트 코퍼스로 만든다.
        둘 다 비어 있으면 빈 문자열을 돌려준다(에러 아님).
        한쪽만 있으면 그쪽만으로 코퍼스를 만든다.
        """
        lines = []
        if include_text and self.has_text():
            for e in self.text_entries:
                lines.append(e["text"])
        if include_image and self.has_image():
            for e in self.image_entries:
                # 키워드 묶음을 한 줄의 텍스트로 흡수
                lines.append(" ".join(e["keywords"]))
        return "\n".join(lines)

    # ── 학습 (엔진에 위임) ────────────────────────────────────
    def train(self, include_text: bool = True, include_image: bool = True,
              on_progress=None):
        """
        현재 코퍼스로 CoreAIv2Engine 학습.
        코퍼스가 비면 학습하지 않고 False 반환(빈 프로필 허용).
        """
        corpus = self.build_corpus(include_text, include_image)
        if not corpus.strip():
            self.engine = None
            self.is_trained = False
            return False
        try:
            from core_ai_v2_engine import CoreAIv2Engine
        except Exception as e:
            raise RuntimeError(f"엔진 로드 실패: {e}")
        self.engine = CoreAIv2Engine()
        self.engine.corpus_name = f"student:{self.student_id}"
        self.engine.train(corpus, on_progress=on_progress)
        self.is_trained = True
        self.updated_ts = time.time()
        return True

    def evaluate(self, text: str):
        """학습됐을 때만 판정. 아니면 SKIP."""
        if not self.is_trained or self.engine is None:
            return {"verdict": "SKIP", "reason": "학습된 코퍼스 없음"}
        return self.engine.evaluate(text)

    # ── 저장 / 로드 (학생 1명 = pkl 1개) ─────────────────────
    def to_dict(self) -> dict:
        data = {
            "student_id":    self.student_id,
            "name":          self.name,
            "text_entries":  self.text_entries,
            "image_entries": self.image_entries,
            "created_ts":    self.created_ts,
            "updated_ts":    self.updated_ts,
            "is_trained":    self.is_trained,
        }
        # 학습된 엔진이 있으면 같이 저장(없으면 생략 — 빈 프로필 OK)
        if self.is_trained and self.engine is not None:
            try:
                import numpy as np
                v2 = self.engine
                data["engine"] = {
                    "n_clusters":        v2.n_clusters,
                    "global_vocab":      v2.global_vocab,
                    "corpus_name":       getattr(v2, "corpus_name", ""),
                    "train_stats":       getattr(v2, "train_stats", {}),
                    "emb_emb":           v2.embedder.emb if v2.embedder else None,
                    "emb_vocab":         v2.embedder.vocab if v2.embedder else None,
                    "emb_dim":           v2.embedder.dim if v2.embedder else 32,
                    "cluster_sentences": dict(v2.decomposer.cluster_sentences),
                    "cluster_tokens":    dict(v2.decomposer.cluster_tokens),
                    "cluster_keywords":  v2.decomposer.cluster_keywords,
                    "decomp_vocab":      v2.decomposer.vocab,
                    "decomp_W":          v2.decomposer.W,
                    "markovs": {
                        k: {"uni": dict(m.uni),
                            "bi":  {k2: dict(vv) for k2, vv in m.bi.items()},
                            "tri": {k2: dict(vv) for k2, vv in m.tri.items()},
                            "total": m.total}
                        for k, m in v2.markovs.items()
                    },
                }
            except Exception:
                pass  # 엔진 저장 실패해도 프로필(코퍼스)은 보존
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "StudentProfile":
        p = cls(data["student_id"], data.get("name", ""))
        p.text_entries  = data.get("text_entries", [])
        p.image_entries = data.get("image_entries", [])
        p.created_ts    = data.get("created_ts", time.time())
        p.updated_ts    = data.get("updated_ts", time.time())
        p.is_trained    = data.get("is_trained", False)
        # 엔진이 저장돼 있으면 복원, 없으면 코퍼스만 (재학습은 호출측 판단)
        if "engine" in data:
            try:
                from core_ai_v2_engine import CoreAIv2Engine
                p.engine = CoreAIv2Engine.load_from_dict(data["engine"])
                p.is_trained = True
            except Exception:
                p.engine = None
                p.is_trained = False
        return p

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)
        return path

    @classmethod
    def load(cls, path: str) -> "StudentProfile":
        with open(path, "rb") as f:
            return cls.from_dict(pickle.load(f))

    # ── 상태 요약 ─────────────────────────────────────────────
    def summary(self) -> dict:
        return {
            "student_id":  self.student_id,
            "name":        self.name,
            "텍스트 조각": len(self.text_entries),
            "그림 키워드 묶음": len(self.image_entries),
            "텍스트 있음": self.has_text(),
            "그림 있음":   self.has_image(),
            "학습됨":      self.is_trained,
            "빈 프로필":   self.is_empty(),
        }

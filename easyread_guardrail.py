"""
easyread_guardrail.py — easy-read 베이스 + 학생 개인 보정 (방식 B)
==================================================================
데이터 희소성 해결을 위한 2층 구조:

  [베이스 층] easy-read 일반 코퍼스(서울시 읽기쉬운자료 등)
             → "쉬운 말이란 무엇인가"의 판정 토대. 데이터 충분 → 안정.
             → 판정의 주축(마르코프 logP). 당신 시스템 철학 그대로.

  [개인 층]  학생 개인 단어(반향어/그림키워드 등)
             → 베이스 판정에 '보너스'로 얹음. "이 학생한텐 더 쉬움"을 반영.
             → 보조 역할(마르코프가 주축, 보정이 보조).

핵심: 베이스 마르코프가 먼저 판정하고, 학생이 잘 아는 단어가
      들어가면 그 토큰 점수를 올려 PASS 쪽으로 당긴다.
      학생이 모르는 어려운 단어는 베이스 판정 그대로 FATAL 쪽.

[당신이 정할 값 — 임의 확정하지 않고 파라미터로 노출]
  student_bonus    : 학생 단어 1개당 logP 보너스 (기본 2.0, 당신이 조정)
  student_min_count: 학생 단어로 인정할 최소 등장 횟수 (기본 1)
  pass_thr/fatal_thr: 판정 경계 — 당신의 -10/-14 철학 유지(기본값으로 둠)
"""
from __future__ import annotations
import numpy as np
from collections import Counter, defaultdict


class EasyReadGuardrail:
    def __init__(self,
                 student_bonus: float = 2.0,
                 student_min_count: int = 1,
                 pass_thr: float = -10.0,
                 fatal_thr: float = -14.0,
                 alpha: float = 0.001,
                 l1: float = 0.6, l2: float = 0.3, l3: float = 0.1):
        # ── 당신이 조정할 값들 ──
        self.student_bonus = student_bonus          # 학생 단어 보너스 크기
        self.student_min_count = student_min_count  # 학생 단어 인정 기준
        self.pass_thr = pass_thr                    # PASS 경계(당신 -10)
        self.fatal_thr = fatal_thr                  # FATAL 경계(당신 -14)
        # JM smoothing 가중치(당신 기본값)
        self.alpha = alpha
        self.l1, self.l2, self.l3 = l1, l2, l3

        # 베이스 마르코프(easy-read 코퍼스로 학습)
        self.uni = Counter(); self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter); self.total = 0
        # 학생 개인 단어 집합(보너스 대상)
        self.student_words: set = set()
        self.base_trained = False

    # ── 토크나이저(엔진과 동일 규칙) ──────────────────────────
    @staticmethod
    def _tok(text: str) -> list:
        try:
            from korean_tokenizer import tokenize
            return tokenize(text)
        except Exception:
            return text.replace("\n", " ").split()

    # ── 베이스 학습(easy-read 일반 코퍼스) ────────────────────
    def train_base(self, easyread_corpus: str):
        toks = self._tok(easyread_corpus)
        self.uni = Counter(); self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter)
        for i, t in enumerate(toks):
            self.uni[t] += 1
            if i >= 1: self.bi[toks[i-1]][t] += 1
            if i >= 2: self.tri[(toks[i-2], toks[i-1])][t] += 1
        self.total = len(toks)
        self.base_trained = self.total > 0
        return self.base_trained

    # ── 학생 개인 단어 등록(보너스 대상) ──────────────────────
    def set_student_words(self, student_corpus: str):
        """학생 코퍼스에서 일정 빈도 이상 단어를 '잘 아는 단어'로 등록."""
        toks = self._tok(student_corpus)
        cnt = Counter(toks)
        self.student_words = {w for w, c in cnt.items()
                              if c >= self.student_min_count}
        return len(self.student_words)

    # ── 판정: 베이스 logP + 학생 보너스 ───────────────────────
    def evaluate(self, text: str) -> dict:
        if not self.base_trained:
            return {"verdict": "SKIP", "reason": "베이스 미학습", "avg_logp": 0.0}
        toks = self._tok(text)
        if len(toks) < 3:
            return {"verdict": "SKIP", "reason": "문장이 짧음", "avg_logp": 0.0}

        V = len(self.uni)
        total_lp = 0.0
        per = []
        for i in range(2, len(toks)):
            wc, wp, wpp = toks[i], toks[i-1], toks[i-2]
            p1 = (self.uni[wc] + self.alpha) / (self.total + self.alpha * V)
            cp = self.uni[wp]
            p2 = (self.bi[wp][wc] / cp) if cp > 0 else 0.0
            cpp = self.bi[wpp][wp]
            p3 = (self.tri[(wpp, wp)][wc] / cpp) if cpp > 0 else 0.0
            pjm = self.l1 * p3 + self.l2 * p2 + self.l3 * p1
            lp = float(np.log(pjm + 1e-12))
            # ── 학생 보너스: 이 학생이 잘 아는 단어면 점수 올림 ──
            bonus = self.student_bonus if wc in self.student_words else 0.0
            lp_adj = lp + bonus
            total_lp += lp_adj
            per.append({"token": wc, "logp": lp, "bonus": bonus,
                        "logp_adj": lp_adj,
                        "student_known": wc in self.student_words})

        avg = total_lp / max(len(toks) - 2, 1)
        # 판정(당신의 -10/-14 철학)
        if avg >= self.pass_thr:    verdict = "PASS"
        elif avg >= self.fatal_thr: verdict = "WARNING"
        else:                       verdict = "FATAL"

        return {"verdict": verdict, "avg_logp": avg,
                "per_token": per,
                "student_word_count": len(self.student_words)}

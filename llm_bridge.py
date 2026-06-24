"""
llm_bridge.py — OpenAI 연동 (생성 전담, 판정은 가드레일이)
==========================================================
역할 분담(핵심):
  - 판정/분석 기준 = 자체 가드레일(DualGuardrail). LLM 아님.
  - LLM(OpenAI)   = 가드레일을 통과하는 답을 '생성'하고,
                    FATAL이면 학생 쉬운 말로 '재작성'만 한다.
  → LLM이 판단을 대신하지 않으므로 단순 래퍼가 아니다.

두 가지 용도(둘 다 지원):
  1) respond(): 전문 지식 기반으로 상황을 분석/응답 (일반 대화)
  2) simplify(): FATAL(너무 어려움) 안내문을 그 학생 쉬운 말로 재작성
  두 경우 모두 생성 결과를 DualGuardrail로 재검증하고,
  통과할 때까지 정해진 횟수만큼 재생성한다.
"""
from __future__ import annotations
from typing import Optional


def _call_openai(api_key: str, model: str, system: str, user: str,
                 max_tokens: int = 800, temperature: float = 0.4) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


class LLMBridge:
    def __init__(self, guardrail, api_key: str, model: str = "gpt-4o-mini"):
        self.g = guardrail          # DualGuardrail
        self.api_key = api_key
        self.model = model

    # ── 학생 쉬운 말 힌트 (프로필 코퍼스에서 단어 추출) ────────
    def _student_vocab_hint(self, n: int = 40) -> str:
        eng = self.g.student_engine
        if eng is None or not getattr(eng, "global_vocab", None):
            return ""
        words = list(eng.global_vocab.keys())[:n]
        return ", ".join(words)

    def _expert_hint(self, n: int = 1500) -> str:
        return (self.g.expert_corpus or "")[:n]

    # ── 1) 일반 응답: 전문 지식 기반 분석/대화 ────────────────
    def respond(self, user_message: str, max_attempts: int = 3) -> dict:
        """
        전문 코퍼스(매뉴얼/개론)를 근거로 상황을 분석/응답.
        생성 후 이중 가드레일로 검증, 통과까지 재생성.
        """
        expert_hint = self._expert_hint()
        student_hint = self._student_vocab_hint()
        sys = (
            "당신은 특수교육·안전 분야 전문 보조자입니다.\n"
            "[규칙]\n"
            "1. 아래 전문 자료를 근거로 정확하게 답하세요.\n"
            "2. 자료에 없는 내용을 지어내지 마세요.\n"
            "3. 답은 이 학생이 이해할 수 있는 쉬운 말로 쓰세요.\n"
            f"[전문 자료]\n{expert_hint}\n\n"
            f"[학생이 이해하는 쉬운 단어들]\n{student_hint}"
        )
        last = ""
        history = []
        for attempt in range(1, max_attempts + 1):
            prompt = user_message if attempt == 1 else (
                f"앞 답변이 학생에게 너무 어렵거나 전문 내용에서 벗어났어요.\n"
                f"더 쉬운 말로, 전문 자료에 맞게 다시 답해주세요.\n질문: {user_message}"
            )
            try:
                last = _call_openai(self.api_key, self.model, sys, prompt)
            except Exception as e:
                return {"answer": f"[LLM 오류: {e}]", "verdict": "ERROR",
                        "attempts": attempt, "history": history}
            v = self.g.evaluate(last)
            history.append({"attempt": attempt, "overall": v["overall"],
                            "expert": v["expert"]["verdict"],
                            "student": v["student"]["verdict"],
                            "preview": last[:60]})
            if v["overall"] in ("PASS", "SKIP"):
                break
        return {"answer": last, "verdict": v["overall"],
                "expert": v["expert"]["verdict"],
                "student": v["student"]["verdict"],
                "attempts": attempt, "history": history}

    # ── 2) 쉬운 말 재작성: FATAL 안내문 → 학생 수준 ───────────
    def simplify(self, hard_text: str, max_attempts: int = 4) -> dict:
        """
        너무 어려운 안내문을 그 학생이 이해할 쉬운 말로 재작성.
        학생 가드레일이 PASS할 때까지(또는 한도까지) 반복.
        """
        student_hint = self._student_vocab_hint()
        sys = (
            "당신은 발달장애 학생을 위한 '쉬운 말' 변환기입니다.\n"
            "[규칙]\n"
            "1. 짧은 문장, 쉬운 단어, 한 번에 한 가지 행동만.\n"
            "2. 아래 학생이 평소 쓰는 단어를 최대한 사용하세요.\n"
            "3. 뜻은 그대로, 표현만 쉽게. 새 정보를 더하지 마세요.\n"
            f"[학생이 이해하는 쉬운 단어들]\n{student_hint}"
        )
        last = hard_text
        history = []
        v = self.g.evaluate(last)
        for attempt in range(1, max_attempts + 1):
            if v["student"]["verdict"] == "PASS":
                break
            prompt = (
                f"다음 문장을 위 학생이 이해할 수 있게 더 쉽게 바꿔주세요.\n"
                f"문장: {last}"
            )
            try:
                last = _call_openai(self.api_key, self.model, sys, prompt,
                                    temperature=0.3)
            except Exception as e:
                return {"answer": f"[LLM 오류: {e}]", "verdict": "ERROR",
                        "attempts": attempt, "history": history}
            v = self.g.evaluate(last)
            history.append({"attempt": attempt,
                            "student": v["student"]["verdict"],
                            "preview": last[:60]})
        return {"answer": last,
                "student": v["student"]["verdict"],
                "expert": v["expert"]["verdict"],
                "attempts": len(history) or 1, "history": history}

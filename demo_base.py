"""demo_base.py — 실제 easy-read 자료로 베이스 학습 + 학생 보정 데모
※ 베이스: 장애인사회연구소 '안전한 매일매일'(생활안전/사회재난) 쉬운문서에서 추출·정제.
   학생 단어는 예시용 가짜."""
from easyread_guardrail import EasyReadGuardrail

# 1) easy-read 베이스 학습 (데이터 충분 → 판정 안정)
corpus = open("easyread_base_corpus.txt", encoding="utf-8").read()
g = EasyReadGuardrail(student_bonus=2.0, student_min_count=1)
g.train_base(corpus)
print(f"베이스 학습 완료: 어휘 {len(g.uni)}개, 토큰 {g.total}개")

# 2) 한 학생의 개인 단어를 보정으로 얹기 (예시)
g.set_student_words("버스 정류장 버스 노랑 천천히")

# 3) 판정
tests = [
    "불이 나면 젖은 손수건으로 코와 입을 막고 밖으로 나가요",   # 쉬움
    "비상벨을 누르고 도움을 요청해요",                          # 쉬움
    "화재 발생 시 신속히 비상구를 통하여 대피하시기 바랍니다",   # 어려운 공문체
    "분기 영업이익이 전년 대비 대폭 증가하였다",                # 도메인밖
]
for t in tests:
    r = g.evaluate(t)
    print(f"[{r['verdict']:6}] logP={r['avg_logp']:7.2f}  {t}")

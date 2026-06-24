"""demo_easyread.py — easy-read 베이스 + 학생 보정(방식 B) 데모.
※ 베이스/학생 데이터는 전부 예시용 가짜. 보너스·임계값은 파라미터로 조정."""
from easyread_guardrail import EasyReadGuardrail

easyread = ("불이 나면 밖으로 나가요\n불이 나면 선생님을 따라가요\n천천히 걸어서 나가요\n"
            "줄을 서서 걸어요\n선생님 손을 잡아요\n문을 열고 밖으로 나가요\n"
            "큰 소리가 나면 귀를 막아요\n침착하게 걸어요\n계단으로 내려가요\n"
            "밖에 나가서 기다려요\n친구와 함께 나가요\n손을 잡고 걸어요\n") * 6

g = EasyReadGuardrail(student_bonus=2.0, student_min_count=1)
g.train_base(easyread)
g.set_student_words("버스 타고 천천히 버스 타고 천천히")

for t in ["불이 나면 선생님 손을 잡고 밖으로 나가요",
          "금일 화재 발생 비상구로 신속히 대피 바람",
          "불이 나면 버스 타고 천천히 나가요"]:
    r = g.evaluate(t)
    print(f"[{r['verdict']:7}] logP={r['avg_logp']:+6.2f}  {t}")

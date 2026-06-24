"""demo.py — 이중 가드레일 데모 (LLM 없이 판정만). 데이터는 전부 예시용 가짜."""
from student_profile import StudentProfile
from dual_guardrail import DualGuardrail

# 학생 프로필(예시)
p = StudentProfile("demo01","예시학생")
p.add_text("불이 나면 선생님 손을 잡아요")
p.add_text("천천히 걸어서 밖으로 나가요")
p.add_image_keywords("선생님 손 웃음")

# 전문 코퍼스(예시: 아주 짧은 안전 매뉴얼 발췌 흉내)
g = DualGuardrail()
g.add_expert_text("화재 발생 시 대피 경로를 확보하고 침착하게 이동한다 "
                  "교사는 학생의 손을 잡고 비상구로 안내한다 "
                  "연기가 많으면 자세를 낮추고 코와 입을 막는다", "안전매뉴얼")
g.set_student(p)
g.train_expert(); g.train_student()
print("요약:", g.summary())

for t in ["불이 나면 선생님 손을 잡고 천천히 밖으로 나가요",
          "금일 화재 발생 비상구로 신속히 대피 바람",
          "주가가 폭락했다"]:
    v = g.evaluate(t)
    print(f"  정확성={v['expert']['verdict']:7} 이해도={v['student']['verdict']:7} 종합={v['overall']:7}  | {t[:28]}")

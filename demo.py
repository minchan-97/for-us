"""demo.py — 발달장애 학생 프로필 데모 (UI 없이 동작 확인용)
※ 아래 데이터는 전부 예시용 가짜 데이터입니다."""
from student_profile import StudentProfile

p = StudentProfile("demo01", "예시학생")
# 텍스트(선택): 쉬운 말 / 평소 발화 — 예시
p.add_text("불이 나면 선생님 손을 잡아요", source="예시")
p.add_text("천천히 걸어서 밖으로 나가요", source="예시")
p.add_text("문은 손잡이를 잡고 밀어요", source="예시")
# 그림 키워드(선택) — 예시
p.add_image_keywords("버스 노랑 반복", source="예시그림")
p.add_image_keywords("선생님 손 웃음", source="예시그림")

print("프로필:", p.summary())
p.train()
print("학습 후:", p.summary())

for t in ["불이 나면 선생님 손을 잡고 밖으로 나가요",
          "금일 화재 발생 비상구로 신속히 대피 바람",
          "주식 시장이 급락하여 투자자들이 패닉에 빠졌다"]:
    r = p.evaluate(t)
    print(f"  [{r.get('verdict')}] {t[:30]}")

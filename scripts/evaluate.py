
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # путь к rag_bot

import rag_bot as bot

# (вопрос, should_answer): должен ли бот дать ответ
GOLDEN = [
    # --- известные темы (есть в базе) -> ждём ОТВЕТ ---
    ("Who piloted the Stormhawk?", True),
    ("Who is Joran Drellos?", True),
    ("What is the Astral Dominion?", True),
    ("Who is Rourke Caldris?", True),
    ("What is a plasma-brand?", True),
    ("Who are the Sephari?", True),
    ("Who is Wrentak?", True),
    ("What is the Astral Concord?", True),
    # --- удалённые / отсутствующие -> ждём ОТКАЗ ---
    ("What is the Void Core?", False),          # удалён void-core.md
    ("What is the Synth Flux?", False),         # удалён the-synth-flux.md
    ("Who is Oodran?", False),                  # удалён oodran.md
    ("Who is Darth Vader?", False),             # нет в базе (термины вымышлены)
    ("What is the capital of France?", False),  # вне домена
]


def main():
    print(f"Прогон golden set: {len(GOLDEN)} вопросов\n")
    rows = []
    for question, should in GOLDEN:
        resp = bot.answer(question)
        answered = not resp.startswith("Я не знаю")
        correct = (answered == should)
        rows.append((question, should, answered, correct))

        mark = "OK" if correct else "XX"
        exp = "ответ" if should else "отказ"
        got = "ответ" if answered else "отказ"
        print(f"[{mark}] ожид={exp} факт={got}  {question}")

    passed = sum(1 for *_, c in rows if c)
    print(f"\nТочность: {passed}/{len(rows)} = {100 * passed // len(rows)}%")

    leaks = [q for q, should, answered, correct in rows if not should and answered]
    if leaks:
        print("\nОтветил, хотя сущность удалена/отсутствует "
              "(знание «протекает» из чужих чанков):")
        for q in leaks:
            print("  -", q)


if __name__ == "__main__":
    main()

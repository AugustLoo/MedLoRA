"""统一的提示词模板。训练数据转换和评估必须用同一套, 否则对比不公平。"""

SLAKE_CLOSED = "{question}\nAnswer with yes or no only."
SLAKE_OPEN = "{question}\nAnswer with a single word or short phrase."

PUBMEDQA = (
    "You are a biomedical research assistant.\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Based only on the context, answer with exactly one word: yes, no, or maybe."
)

GENERAL_VQA = "{question}\nAnswer with a single word or short phrase."


def slake_prompt(question: str, answer_type: str) -> str:
    tpl = SLAKE_CLOSED if str(answer_type).upper() == "CLOSED" else SLAKE_OPEN
    return tpl.format(question=question.strip())

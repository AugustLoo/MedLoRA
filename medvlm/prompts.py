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

# 第二个通用能力探针 (MMBench 四选一)。hint 为空时占位为空串, 见 eval/eval_mmbench.py。
MMBENCH = (
    "{hint}{question}\n"
    "{options}\n"
    "Answer with the option's letter from the given choices directly."
)


# 外部医学知识测试 (MedQA 四选一, 纯文字), 措辞与 MMBench 一致, 见 eval/eval_medqa.py。
MEDQA = (
    "{question}\n"
    "{options}\n"
    "Answer with the option's letter from the given choices directly."
)


def slake_prompt(question: str, answer_type: str) -> str:
    tpl = SLAKE_CLOSED if str(answer_type).upper() == "CLOSED" else SLAKE_OPEN
    return tpl.format(question=question.strip())


# ---- 35B 三步训练 (docs/PLAN_35B_CPT.md) ----
# SLAKE 中文题 (测试 1,033 道) 的提示词, 与英文版同一结构; 见 eval/eval_slake.py --lang zh。
SLAKE_CLOSED_ZH = "{question}\n请只回答是或否。"
SLAKE_OPEN_ZH = "{question}\n请用一个词或短语回答。"

# 中文医师考试 (CMExam) 选择题, 直接答选项字母; 见 eval/eval_cmexam.py。
CMEXAM = (
    "{question}\n"
    "{options}\n"
    "请直接回答选项字母。"
)

# 第③步推理训练: 先给答案, 再给解析 (不开思考模式); 见 data/convert_sft35_stage3.py。
MCQ_EXPLAIN_ZH = "{question}\n{options}\n请先给出答案选项，再简要解析。"
MCQ_EXPLAIN_EN = "{question}\n{options}\nGive the answer letter first, then a brief explanation."


def slake_prompt_zh(question: str, answer_type: str) -> str:
    tpl = SLAKE_CLOSED_ZH if str(answer_type).upper() == "CLOSED" else SLAKE_OPEN_ZH
    return tpl.format(question=question.strip())

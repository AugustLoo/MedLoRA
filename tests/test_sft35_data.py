import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
import convert_sft35_stage2 as S2  # noqa: E402
import convert_sft35_stage3 as S3  # noqa: E402
import sft35_common as C  # noqa: E402


def chars(s):
    return len(s)


# ---------------------------------------------------------------- 测试集隔离
def make_index():
    idx = C.EvalIndex()
    idx.add("medqa", ["A 45-year-old man presents to the emergency department with crushing chest pain radiating to the "
                      "left arm for two hours. Which of the following is the most likely diagnosis?"])
    idx.add("slake", ["图片中包含肝脏吗?"])
    idx.add("cmexam", ["下列各项，最符合“主诉”书写要求的是"])
    return idx


def test_eval_index_catches_exact_and_light_rewrites():
    idx = make_index()
    assert idx.contains("图片中包含肝脏吗？")                     # 只差标点
    assert idx.contains("下列各项 最符合 主诉 书写要求的是")
    rewrite = ("A 45-year-old man presents to the emergency department with crushing chest pain radiating to the "
               "left arm for two hours. What is the most likely diagnosis?")
    assert idx.contains(rewrite)                                 # 只改了最后一句


def test_eval_index_ignores_shared_case_boilerplate():
    idx = make_index()
    other = ("A 45-year-old man presents to the emergency department with fever, productive cough and right lower "
             "lobe crackles for three days after a long flight. Which antibiotic should be started first?")
    assert not idx.contains(other)
    assert not idx.contains("肝脏")                              # 太短, 只查完全相同


def test_has_pii():
    assert C.has_pii("请拨打13812345678咨询")
    assert C.has_pii("详见 https://example.com")
    assert C.has_pii("加我QQ：123456789")
    assert not C.has_pii("血压 135/85 mmHg, 心率 78 次/分")


# ---------------------------------------------------------------- 第②步
def test_huatuo_strips_prefixes_and_rejects_odd_turns():
    msgs = S2.huatuo_messages({"data": ["问：头痛怎么办？", "答：多休息，必要时就医。"]})
    assert msgs == [{"role": "user", "content": "头痛怎么办？"}, {"role": "assistant", "content": "多休息，必要时就医。"}]
    assert S2.huatuo_messages({"data": ["问：a", "答：b", "问：c"]}) is None
    assert S2.huatuo_messages({"data": ["问：BR>目前疲倦", "答：建议复查肝功能"]})[0]["content"] == "目前疲倦"


def test_disc_requires_alternating_roles():
    ok = {"conversation": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    bad = {"conversation": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]}
    assert S2.disc_messages(ok) and S2.disc_messages(bad) is None


def test_clean_rules():
    good = [{"role": "user", "content": "颈椎病术后怎么睡？"}, {"role": "assistant", "content": "仰卧时可以不用枕头，侧卧时用高一些的枕头。"}]
    assert S2.clean(good, max_rounds=3, image_refs=True)[0] == good
    pic = [{"role": "user", "content": "这是图片，您看一下对位可以吗"}, good[1]]
    assert S2.clean(pic, max_rounds=3, image_refs=True)[0] is None
    assert S2.clean(pic, max_rounds=3, image_refs=False)[0] == pic          # 华佗数据不做这条
    short = [good[0], {"role": "assistant", "content": "好的"}]
    assert S2.clean(short, max_rounds=3, image_refs=True)[0] is None
    long = good * 5
    assert len(S2.clean(long, max_rounds=3, image_refs=True)[0]) == 6


def test_trim_to_fit_drops_rounds_from_the_end():
    msgs = [{"role": "user", "content": "问" * 50}, {"role": "assistant", "content": "答" * 50}] * 3
    out = S2.trim_to_fit(msgs, chars, limit=200)   # 每条 50 字 + 模板 12 = 62; 两轮 248 超了, 一轮 124 放得下
    assert len(out) == 2
    assert S2.trim_to_fit([{"role": "user", "content": "问" * 300}, msgs[1]], chars, limit=200) is None


# ---------------------------------------------------------------- 第③步
def test_cmexam_row_format():
    r = {"Question": "下列除哪项外均符合问诊的要求", "Answer": "D", "Explanation": "医生应避免诱导性提问，即避免在措辞上暗示期望的答案。",
         "Options": [{"key": k, "value": v} for k, v in zip("ABCDE", ["态度和蔼", "去粗取精", "避免术语", "多提诱导性问题", "耐心倾听"])]}
    ex, why = S3.cmexam_row(r)
    user, ans = ex["messages"][0]["content"], ex["messages"][1]["content"]
    assert user.startswith("下列除哪项外均符合问诊的要求\nA. 态度和蔼") and user.endswith("请先给出答案选项，再简要解析。")
    assert ans.startswith("答案：D\n解析：医生应避免")
    assert S3.cmexam_row({**r, "Answer": "AB"})[0] is None
    assert S3.cmexam_row({**r, "Explanation": ""})[0] is None


def test_medmcqa_row_strips_answer_prefix():
    r = {"question": "Which vitamin is supplied from only animal source:", "opa": "Vitamin C", "opb": "Vitamin B7",
         "opc": "Vitamin B12", "opd": "Vitamin D", "cop": 2, "choice_type": "single",
         "exp": "Ans. is 'c' i.e., Vitamin B12 Vitamin B12 is found only in foods of animal origin."}
    ex, _ = S3.medmcqa_row(r)
    assert ex["messages"][1]["content"] == "Answer: C\nExplanation: Vitamin B12 Vitamin B12 is found only in foods of animal origin."
    assert S3.medmcqa_row({**r, "choice_type": "multi"})[0] is None
    assert S3.medmcqa_row({**r, "exp": None})[0] is None


def test_o1_row_uses_final_response_only():
    ex, _ = S3.o1_row({"Question": "What causes paradoxical embolism?", "Complex_CoT": "Okay, let's think…",
                       "Response": "A patent foramen ovale allows a venous clot to reach the arterial side."})
    assert ex["messages"][1]["content"].startswith("A patent foramen ovale") and "Okay" not in ex["messages"][1]["content"]


def test_promo_filter_spares_normal_medical_qa():
    normal = "您好，胎记可以考虑激光治疗，费用因医院而异，建议咨询皮肤科医生并预约挂号。人流对以后怀孕的影响要看具体情况。"
    assert not C.looks_like_promo_qa(normal)
    promo = "我院白癜风专家在线解答，欢迎来院就诊，咨询热线请拨打，本月优惠名额有限。"
    assert C.looks_like_promo_qa(promo)


def test_strip_html():
    assert C.strip_html("BR>目前一般情况:疲倦") == "目前一般情况:疲倦"
    assert C.strip_html("第一行<br/>第二行&nbsp;") == "第一行 第二行"
    assert C.strip_html("血压 <140/90 mmHg") == "血压 <140/90 mmHg"

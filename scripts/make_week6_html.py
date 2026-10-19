"""第 6 周周报网页 (中英对照, 精简版): docs/weekly/week6/Topic6_WeeklyReport6.html
版式 (CSS + 点击翻页) 原样取自 week5 网页; 每节一两句话, 数字用图表; 数字从 results/ 的快照读, 不手抄。
2026-10-19 起含三步训练的评测结果 (results/interns2_3stage_eval_2026-10-19.json 与两份置信区间)。
用法: python scripts/make_week6_html.py
"""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
W5 = (REPO / "docs/weekly/week5/Topic6_WeeklyReport5.html").read_text(encoding="utf-8")
css = re.search(r"<style>\n(.*?)</style>", W5, re.S).group(1)
pager = W5[W5.index('<nav id="pager"'):W5.index("</body>")]

S = json.loads((R / "interns2_3stage_train_2026-10-18.json").read_text(encoding="utf-8"))
EV = json.loads((R / "interns2_3stage_eval_2026-10-19.json").read_text(encoding="utf-8"))["models"]
CI_A = json.loads((R / "interns2_ci_3stage_vsA_2026-10-19.json").read_text(encoding="utf-8"))["tests"]
CI_B = json.loads((R / "interns2_ci_3stage_CvsB_2026-10-19.json").read_text(encoding="utf-8"))["tests"]
EN = json.loads((R / "interns2_cxp_2026-10-13.json").read_text(encoding="utf-8"))["models"]
MD = S["models"]
CPT, A, B, C = (MD[k] for k in MD)
ZM = S["zh_baseline"]["S2-2datasets"]
D1, D2, D3 = S["data"]["stage1"], S["data"]["stage2_zh_dialog"], S["data"]["stage3_reasoning"]
BASE, SIX, M1, MA, MB, MC = (EV[k] for k in ("原版", "S2-6datasets", "只做①", "A (②)", "B (①→②)", "C (①→②→③)"))


def r1(x):
    """一位小数, 四舍五入 (不用银行家舍入)。"""
    return str(Decimal(str(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))


def r3(x):
    return str(Decimal(str(x)).quantize(Decimal("0.001"), ROUND_HALF_UP))


def sg(x):
    s = r1(x)
    return s if s.startswith("-") else "+" + s


def diff(ci, test, tag):
    d = ci[test][tag]
    return d["diff_vs_ref"], d["diff_ci95"], d["significant"]


TOK = D1["_total"]["tokens"]
EVAL1 = CPT["eval_loss"]
SWEEP = S["stage1_lr_sweep"]
CONTAM = {k: D3[k]["dropped"].get("与评测题重复或改写", 0) for k in ("cmexam", "medmcqa")}
N_ZH = D2["huatuo_sft"]["kept"] + D2["disc_med_sft"]["kept"]
N_R = sum(D3[k]["kept"] for k in ("cmexam", "medmcqa", "medical_o1_zh", "medical_o1_en"))

# 「B − A」「C − B」配对置信区间表
CI_ROWS = [("SLAKE 是非", "SLAKE closed"), ("SLAKE 开放", "SLAKE open"), ("SLAKE 中文 开放", "SLAKE 中文 open"),
           ("PathVQA 是非", "PathVQA closed"), ("MedQA", "MedQA acc"), ("儿童肺炎", "PneumoniaMNIST closed")]
ci_html = []
n_sig_ba = 0
for zh, key in CI_ROWS:
    ba, bai, bas = diff(CI_A, key, "interns2_s2b")
    cb, cbi, cbs = diff(CI_B, key, "interns2_s3c")
    n_sig_ba += bas
    cell = lambda v, i, s: (f'<td class="{"sig" if s else "ns"}">{sg(v)}<small> [{sg(i[0])}, {sg(i[1])}]</small>'  # noqa: E731
                            f'{" *" if s else ""}</td>')
    ci_html.append(f"<tr><td>{zh}</td>{cell(ba, bai, bas)}{cell(cb, cbi, cbs)}</tr>")
CI_TABLE = "\n      ".join(ci_html)

CHART = {
    "val": [[s, l] for s, l in EVAL1],
    "sweep": [{"lr": lr, "v": v[-1]} for lr, v in SWEEP.items()],
    # 三步训练的结果: 原版 / 只做① / A / B / C
    "res": [{"zh": zh, "en": en, "v": [float(r1(m[k])) for m in (BASE, M1, MA, MB, MC)]} for zh, en, k in (
        ("SLAKE 是非", "SLAKE closed", "slake_closed"), ("SLAKE 开放", "SLAKE open", "slake_open_em"),
        ("病理图", "PathVQA", "pathvqa_closed"), ("儿童肺炎", "PneumoniaMNIST", "pneumonia_closed"),
        ("医师考试", "MedQA", "medqa"))],
    # 去掉图像: 有图 vs 无图 (A / B / C)
    "noimg": [{"zh": zh, "en": en, "v": [float(r1(m[k])) for m in (MA, MB, MC) for k in ks]} for zh, en, ks in (
        ("是非题", "closed", ("slake_closed", "slake_noimg_closed")), ("开放题", "open EM", ("slake_open_em", "slake_noimg_open_em")))],
    # 成人胸片描述漏报: 原版 / S2-6datasets / 只做① / A / B / C
    "cxr": [float(r1(m["cxr_desc_miss_pct"])) for m in (BASE, SIX, M1, MA, MB, MC)],
    # 中文: 原版 / S2-2datasets / A / B / C
    "zh": [{"zh": zh, "en": en, "v": [float(r1(x)) for x in vals]} for zh, en, vals in (
        ("中文 是非", "closed · ZH", [BASE["slake_zh_closed"], ZM["slake_zh"]["closed_acc"], MA["slake_zh_closed"], MB["slake_zh_closed"], MC["slake_zh_closed"]]),
        ("中文 开放", "open EM · ZH", [BASE["slake_zh_open_em"], ZM["slake_zh"]["open_em"], MA["slake_zh_open_em"], MB["slake_zh_open_em"], MC["slake_zh_open_em"]]),
        ("CMExam", "CMExam", [BASE["cmexam"], ZM["cmexam_acc"], MA["cmexam"], MB["cmexam"], MC["cmexam"]]))],
}
OPEN_GAP = float(r1(EN["S2-2datasets"]["slake_open_em"] - ZM["slake_zh"]["open_em"]))
BA_SLAKE, _, _ = diff(CI_A, "SLAKE closed", "interns2_s2b")

BODY = f"""
<!-- ============================ Cover ============================ -->
<section id="cover">
  <svg id="grid" viewBox="0 0 1400 900" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
    <defs><pattern id="gp" width="56" height="56" patternUnits="userSpaceOnUse"><path d="M56 0H0V56" fill="none" stroke="#D8CFBA" stroke-width="1"/></pattern></defs>
    <rect width="1400" height="900" fill="url(#gp)"/>
  </svg>
  <div class="inner">
    <div class="kicker">Topic 6 · Task 1.3 · 第 6 周进展汇报 · Week 6 progress report · 2026-10-22</div>
    <h1>35B 三步训练评完：<br><span class="hi">预训练没带来提升</span><span class="tr-h">Three-stage training on 35B, evaluated: the gains come from fine-tuning; pretraining added nothing measurable</span></h1>
    <p class="sub">按苏学长的方案训了四个模型，全部评完。微调让看图题大幅变好；在微调前加预训练、在后面加推理训练，都没有可测的提升。</p>
    <p class="sub tr">Four models following Su's plan, all evaluated. Fine-tuning lifts image questions a lot; adding pretraining before it or reasoning training after it gives no measurable gain.</p>
    <div class="asof-line">Intern-S2-Preview 35B · LoRA r16 (0.054%) · one training run each (seed 42) · results as of 2026-10-19</div>
    <div class="stats">
      <div class="stat"><div class="v" style="color:var(--c-good)">{r1(MB['slake_open_em'])}<small>← {r1(M1['slake_open_em'])}</small></div><div class="bar" style="background:var(--c-good)"></div><div class="l">SLAKE 开放题：预训练 + 微调 vs 只做预训练<span class="tr-l">SLAKE open: pretrain + fine-tune vs pretrain only</span></div></div>
      <div class="stat"><div class="v" style="color:var(--c-base)">{sg(BA_SLAKE)}</div><div class="bar" style="background:var(--c-base)"></div><div class="l">加了预训练 (B − A)，SLAKE 是非题，在噪声内<span class="tr-l">B − A on SLAKE closed, within noise</span></div></div>
      <div class="stat"><div class="v" style="color:var(--c-red)">{r1(MB['slake_noimg_closed'])}<small>← {r1(MB['slake_closed'])}</small></div><div class="bar" style="background:var(--c-red)"></div><div class="l">去掉图像后是非题：模型真的在看图<span class="tr-l">closed accuracy without the image: the model really looks</span></div></div>
    </div>
  </div>
  <div class="scrollhint">点击翻页 · CLICK ▶</div>
</section>

<!-- ============================ 01 做了什么 ============================ -->
<section class="section deep" id="done">
  <div class="prose">
    <div class="kicker">01 · 本周 · This week</div>
    <h2>这周做了什么<span class="tr-h">What I did this week</span></h2>
  </div>
  <div class="card next">
    <ol>
      <li><strong>补做预训练：</strong>论文题目有「增量预训练」，所以在 35B 上按三步补做，每步只多加一类数据。<span class="tr-li"><strong>Pretraining added:</strong> the thesis title says incremental pretraining, so the 35B model got three stages, one new kind of data per stage.</span></li>
      <li><strong>三步数据：</strong>清洗、去掉推广和个人信息；和所有测试题比对，剔除重复或改写的题（CMExam {CONTAM['cmexam']} 道、MedMCQA {CONTAM['medmcqa']} 道）。<span class="tr-li"><strong>Data:</strong> cleaned, promotion and personal data removed; items matching any test question dropped (CMExam {CONTAM['cmexam']}, MedMCQA {CONTAM['medmcqa']}).</span></li>
      <li><strong>训练：</strong>四个模型加学习率扫描共约 24 小时显卡；中途修了一个模型代码的问题（纯文字批次沿用了带图批次的位置缓存，A 在第 1,745 步崩溃，修好后从断点续训）。<span class="tr-li"><strong>Training:</strong> four models plus the sweep in about 24 GPU hours; fixed a model-code bug on the way (a text-only batch reused the image batch's position cache; A crashed at step 1,745 and resumed after the fix).</span></li>
      <li><strong>评测：</strong>四个模型全套评完（约 7.5 小时显卡）：英文四张表、中文两项、「去掉图像」对照、四个外部测试、成人胸片漏报，差别都用置信区间检查。<span class="tr-li"><strong>Evaluation:</strong> all four models on the full set (about 7.5 GPU hours): four English tables, two Chinese tests, a no-image control, four external tests and adult chest X-rays, with confidence intervals.</span></li>
    </ol>
  </div>
</section>

<!-- ============================ 02 三步训练 ============================ -->
<section class="section" id="stages">
  <div class="prose">
    <div class="kicker">02 · 方法 · Method</div>
    <h2>三步训练，四个模型<span class="tr-h">Three stages, four models</span></h2>
    <p class="lede">像人学医：先读书（预训练），再练题（微调），最后做带解析的题（推理训练）。B 对比 A 看预训练有没有用，C 对比 B 看推理训练有没有用。</p>
    <p class="lede tr">Like learning medicine: read (pretraining), practise (fine-tuning), then explained exam questions (reasoning). B vs A tests pretraining; C vs B tests reasoning training.</p>
  </div>
  <div class="flow">
    <div class="card st"><span class="tag" style="background:var(--c-base)">第 1 步 · 预训练 · pretraining</span><p>29 本开放医学教材、临床指南、中文医学百科，共 {TOK / 1e6:,.1f}M token；只读文字，整段都算损失。</p><p class="tr">29 open textbooks, clinical guidelines, a Chinese medical encyclopaedia ({TOK / 1e6:,.1f}M tokens); text only, loss on all text.</p></div>
    <div class="card st"><span class="tag" style="background:var(--c-good)">第 2 步 · 微调 · fine-tuning</span><p>六个看图数据集 + 中文问诊对话 {N_ZH:,} 条；练「题目 → 答案」，只算答案的损失。</p><p class="tr">The six image datasets + {N_ZH:,} Chinese medical dialogues; question → answer, loss on the answer only.</p></div>
    <div class="card st"><span class="tag" style="background:var(--c-c1)">第 3 步 · 推理训练 · reasoning</span><p>带解析的考题 {N_R:,} 道（CMExam、MedMCQA、medical-o1）；只算答案 + 解析。</p><p class="tr">{N_R:,} exam questions with explanations; loss on answer + explanation.</p></div>
  </div>
  <table class="mt">
    <thead><tr><th>模型 · model</th><th>预训练</th><th>微调</th><th>推理训练</th><th>步数 · steps</th><th>用时 · time</th></tr></thead>
    <tbody>
      <tr class="dim"><td>只做预训练 · pretrain only</td><td>✓</td><td>—</td><td>—</td><td>{CPT['steps']:,}</td><td>3.1 h</td></tr>
      <tr><td><strong>A</strong> 只做微调（对照）· fine-tune only</td><td>—</td><td>✓</td><td>—</td><td>{A['steps']:,}</td><td>≈ 9.5 h</td></tr>
      <tr><td><strong>B</strong> 预训练 + 微调</td><td>✓</td><td>✓</td><td>—</td><td>{B['steps']:,}</td><td>{r1(B['steps'] * 6.21 / 3600)} h</td></tr>
      <tr><td><strong>C</strong> 三步都做</td><td>✓</td><td>✓</td><td>✓</td><td>{C['steps']:,}</td><td>{r1(C['steps'] * 6.17 / 3600)} h</td></tr>
    </tbody>
  </table>
</section>

<!-- ============================ 03 第 1 步 ============================ -->
<section class="section deep" id="cpt">
  <div class="prose">
    <div class="kicker">03 · 第 1 步 · Stage 1</div>
    <h2>预训练确实在学，学习率先扫了再定<span class="tr-h">Pretraining is learning; the learning rate was swept first</span></h2>
    <p class="lede">留出的教材段落上，损失一路下降、后期变平，没有过拟合。2e-4 只比 1e-4 好 0.8%，按事先定的规则选更稳的 1e-4。</p>
    <p class="lede tr">On held-out text the loss falls and then levels off, with no overfitting. 2e-4 beats 1e-4 by only 0.8%, so the pre-set rule picks the safer 1e-4.</p>
  </div>
  <div class="duo">
    <div class="card"><div class="ttl">验证损失 vs 训练步数（越低越好）· validation loss vs step (lower is better)</div><svg id="val" viewBox="0 0 700 320" role="img" aria-label="Validation loss during stage one"></svg></div>
    <div class="card"><div class="ttl">学习率扫描，120 步后 · LR sweep after 120 steps</div><svg id="sweep" viewBox="0 0 360 320" role="img" aria-label="Validation loss for three learning rates"></svg></div>
  </div>
  <div class="srcline">纵轴不从 0 开始，差别被放大了。学到的是教材文字；下一页看它对考试有没有用。<span class="tr-s">The y-axes do not start at 0. What it learns is textbook text; the next page shows whether that helps on the tests.</span></div>
</section>

<!-- ============================ 04 结果 ============================ -->
<section class="section" id="res">
  <div class="prose">
    <div class="kicker">04 · 结果 · Results</div>
    <h2>分数来自微调，预训练没加分<span class="tr-h">The gains come from fine-tuning; pretraining and reasoning add none</span></h2>
    <p class="lede">只做预训练、不微调，看图题和原版差不多；一微调就大幅变好。A、B、C 三个微调过的模型几乎一样高。</p>
    <p class="lede tr">Pretraining alone stays near the base on image questions; fine-tuning lifts them a lot. The three fine-tuned models A, B, C score almost the same.</p>
  </div>
  <div class="card wide"><div class="ttl">准确率 %（越高越好）· accuracy % (higher is better)</div><svg id="resc" viewBox="0 0 900 330" role="img" aria-label="Accuracy of base, pretrain-only, A, B and C"></svg></div>
  <div class="legend-x duo-legend" id="legend-res"></div>
  <table class="mt ci">
    <thead><tr><th>差别（95% 区间）· difference (95% CI)</th><th>B − A：预训练的作用</th><th>C − B：推理训练的作用</th></tr></thead>
    <tbody>
      {CI_TABLE}
    </tbody>
  </table>
  <div class="srcline">配对重抽样 2,000 次；区间跨 0 就是在噪声内，带 * 的才算真有差别。B − A 六项里 {n_sig_ba} 项有差别：PathVQA 低 0.8 分。每个模型只训练一次。<span class="tr-s">Paired bootstrap, 2,000 resamples; an interval spanning 0 is within noise, * marks a real difference. B − A: {n_sig_ba} of six differ (PathVQA −0.8). One run per model.</span></div>
</section>

<!-- ============================ 05 看图与胸片 ============================ -->
<section class="section deep" id="look">
  <div class="prose">
    <div class="kicker">05 · 可靠性 · Reliability</div>
    <h2>模型在看图；胸片描述变差<span class="tr-h">The models really look at the image; chest X-ray descriptions are worse than S2-6datasets</span></h2>
    <p class="lede">去掉图像只给文字，是非题掉到 50 分左右，等于瞎猜。成人胸片「有病说成正常」20–24%，比原版好，但不如上周的 S2-6datasets（{r1(SIX['cxr_desc_miss_pct'])}%）。</p>
    <p class="lede tr">Without the image, closed accuracy drops to about 50 — chance. Adult chest X-rays described as normal: 20–24%, better than the base but worse than last week's S2-6datasets ({r1(SIX['cxr_desc_miss_pct'])}%).</p>
  </div>
  <div class="duo">
    <div class="card"><div class="ttl">SLAKE：有图 vs 去掉图像 · with vs without the image</div><svg id="noimg" viewBox="0 0 700 320" role="img" aria-label="SLAKE accuracy with and without the image for A, B and C"></svg></div>
    <div class="card"><div class="ttl">成人胸片：病片描述成正常 %（越低越好）· described as normal % (lower is better)</div><svg id="cxr" viewBox="0 0 420 320" role="img" aria-label="Adult chest X-ray description omissions"></svg></div>
  </div>
  <div class="legend-x duo-legend"><span><i style="background:#C25E28"></i>A</span><span><i style="background:#1F8A70"></i>B</span><span><i style="background:#8E44AD"></i>C</span><span>深色 = 有图 · dark = with image</span><span>浅色 = 去掉图像 · light = no image</span><span>6ds = S2-6datasets</span></div>
  <div class="srcline">A 与 S2-6datasets 只差 {N_ZH:,} 条中文对话，所以胸片变差可能来自中文对话，也可能是运气：同一做法换个随机种子，以前能从 9.8% 跳到 43.8%。要用种子 43 重训确认；CheXbert 复核在排队。<span class="tr-s">A differs from S2-6datasets only by the {N_ZH:,} Chinese dialogues, so the drop is either those dialogues or luck: one recipe has moved from 9.8% to 43.8% across seeds. Seed 43 will tell; CheXbert re-check is queued.</span></div>
</section>

<!-- ============================ 06 中文 ============================ -->
<section class="section" id="zh">
  <div class="prose">
    <div class="kicker">06 · 中文 · Chinese</div>
    <h2>中文开放题没补上<span class="tr-h">Chinese open answers did not improve</span></h2>
    <p class="lede">中文是非题都在 93–94；中文开放题 A、B、C 只有 {r1(MA['slake_zh_open_em'])}–{r1(MC['slake_zh_open_em'])}，和只练英文的 S2-2datasets（{r1(ZM['slake_zh']['open_em'])}）差不多，仍比英文低约 {r1(OPEN_GAP)} 分。加的中文对话是纯文字长回答，帮不到看图短答。</p>
    <p class="lede tr">Chinese closed questions sit at 93–94; Chinese open answers for A, B, C are {r1(MA['slake_zh_open_em'])}–{r1(MC['slake_zh_open_em'])}, about the same as English-only S2-2datasets ({r1(ZM['slake_zh']['open_em'])}) and still about {r1(OPEN_GAP)} below English. The added dialogues are text-only long answers and do not help short image answers.</p>
  </div>
  <div class="card wide"><div class="ttl">中文准确率 %（越高越好）· Chinese accuracy % (higher is better)</div><svg id="zhc" viewBox="0 0 900 330" role="img" aria-label="Chinese accuracy of base, S2-2datasets, A, B and C"></svg></div>
  <div class="legend-x duo-legend" id="legend-zh"></div>
  <div class="srcline">要补中文看图，得加中文看图题（SLAKE 中文训练题约 4,900 道，或翻译伪数据）。CMExam 都在 93–94，题太简单，分不出差别。<span class="tr-s">To improve Chinese image answers, add Chinese image questions (about 4,900 SLAKE Chinese training items, or translated pseudo-data). CMExam sits at 93–94 for all: too easy to separate models.</span></div>
</section>

<!-- ============================ 07 文献 ============================ -->
<section class="section deep" id="lit">
  <div class="prose">
    <div class="kicker">07 · 文献 · Literature (2024–2026)</div>
    <h2>文献怎么解释这些结果<span class="tr-h">How the literature explains these results</span></h2>
  </div>
  <div class="card next">
    <ol>
      <li><strong>LoRA Learns Less and Forgets Less</strong>（2024）：LoRA 在预训练里学得少、忘得也少。和我们 3B、35B 两次预训练都没提升一致。<span class="tr-li">LoRA learns less but forgets less in continued pretraining — consistent with no gain at both 3B and 35B.</span></li>
      <li><strong>Lost in the Hype</strong>（2026）：训练视觉编码器只差约 ±0.5 分，支持冻结它；但模型「看得到、说不出」，问题在训练方式。<span class="tr-li">Training the vision encoder changes results by only about ±0.5 points; the model sees more than it says — the training signal is the issue.</span></li>
      <li><strong>Consistent but Dangerous</strong>（CVPR 2026 Workshop）：微调后模型可能不看图，建议做「去掉图像」检查。我们做了：去掉图像后分数掉到瞎猜水平，说明在看图。<span class="tr-li">Fine-tuned models may stop looking; the authors recommend a no-image check. Ours drops to chance without the image, so the models do look.</span></li>
      <li><strong>PU-DPO</strong>（2026）：医生报告会漏写细微发现，模型照着学就学会了少报。这解释了胸片漏报，之后可以用偏好优化来改。<span class="tr-li">Reference reports omit subtle findings and models learn to under-report — an explanation for omissions; preference optimisation is a later fix.</span></li>
    </ol>
  </div>
</section>

<!-- ============================ 08 计划 ============================ -->
<section class="section" id="next">
  <div class="prose">
    <div class="kicker">08 · 计划与讨论 · Plan and discussion</div>
    <h2>接下来的计划<span class="tr-h">Next steps</span></h2>
  </div>
  <div class="card next">
    <ol>
      <li><strong>确认结论：</strong>用种子 43 重训 A 和 B，看「预训练没用」和胸片变差是不是运气；CheXbert 复核 A、B、C（只用 CPU，已排队）。<span class="tr-li"><strong>Confirm:</strong> retrain A and B with seed 43 to check "no gain from pretraining" and the chest X-ray drop; CheXbert re-check of A, B, C (CPU only, queued).</span></li>
      <li><strong>想讨论：</strong>要让预训练起作用，最对症的是「带图的预训练」——把胸片报告改写成带图问答、放开投影层一起训练。要不要做？<span class="tr-li"><strong>To discuss:</strong> the most direct way to make pretraining matter is image-text pretraining — chest X-ray reports rewritten as image Q&amp;A, with the projector unfrozen. Should I?</span></li>
      <li><strong>中文：</strong>加 SLAKE 中文训练题，训一个中英双语版。<span class="tr-li"><strong>Chinese:</strong> add the SLAKE Chinese training items for a bilingual version.</span></li>
      <li><strong>论文：</strong>「两个规模上增量预训练都没有可测提升」写成正式结论；想请老师确认最后能否加一章原型系统。<span class="tr-li"><strong>Thesis:</strong> report "no measurable gain from incremental pretraining at two scales" as a finding; may the last chapter add a prototype system?</span></li>
    </ol>
  </div>
</section>

<footer>
  MedLoRA · <a href="https://github.com/AugustLoo/MedLoRA">github.com/AugustLoo/MedLoRA</a> · 所有数字来自仓库 · every number comes from <span class="en">results/</span>
</footer>
"""

JS = r"""
<script>
(function(){
"use strict";
var NS="http://www.w3.org/2000/svg";
function el(t,a,x){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);if(x!=null)e.textContent=x;return e;}
var C={navy:"#14263B",soft:"#3D4E63",taupe:"#8C8577",rule:"#D8CFBA",base:"#0E67B5",good:"#1F8A70",gold:"#B8860B",purple:"#8E44AD",red:"#C0392B",grey:"#9A9283",orange:"#C25E28"};
var MONO="IBM Plex Mono,monospace";
var D=__DATA__;
function yaxis(svg,L,R,T,B,W,H,lo,hi,ticks,fmt){
  var ph=H-T-B, y=function(v){return T+ph-(v-lo)/(hi-lo)*ph;};
  ticks.forEach(function(t){
    svg.appendChild(el("line",{x1:L,x2:W-R,y1:y(t),y2:y(t),stroke:C.rule}));
    svg.appendChild(el("text",{x:L-8,y:y(t)+4,"text-anchor":"end","font-family":MONO,"font-size":11,fill:C.taupe},fmt(t)));});
  return y;
}
/* 分组柱状图: groups = [{zh,en,v:[...]}], models = [{zh,en,c}] */
function bars(id,W,H,groups,models,opt){
  opt=opt||{};
  var svg=document.getElementById(id),L=40,R=8,T=18,B=52;
  var y=yaxis(svg,L,R,T,B,W,H,0,100,[0,25,50,75,100],function(t){return t;});
  var gw=(W-L-R)/groups.length, inner=gw*(opt.fill||0.8), bw=inner/models.length;
  groups.forEach(function(g,i){
    var x0=L+gw*i+(gw-inner)/2;
    g.v.forEach(function(v,j){
      var m=models[j], x=x0+bw*j, r=el("rect",{x:x+1,y:y(v),width:bw-2,height:y(0)-y(v),fill:m.c,rx:2,opacity:m.op||1});
      r.appendChild(el("title",{},m.zh+" · "+g.zh+": "+v.toFixed(1)));svg.appendChild(r);
      svg.appendChild(el("text",{x:x+bw/2,y:y(v)-5,"text-anchor":"middle","font-family":MONO,"font-size":opt.fs||9.5,fill:C.navy},v.toFixed(1)));
    });
    svg.appendChild(el("text",{x:L+gw*i+gw/2,y:H-B+20,"text-anchor":"middle","font-size":13,"font-weight":700,fill:C.navy},g.zh));
    svg.appendChild(el("text",{x:L+gw*i+gw/2,y:H-B+37,"text-anchor":"middle","font-family":MONO,"font-size":10.5,fill:C.taupe},g.en));
  });
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:C.navy,"stroke-width":1.6}));
}
function legend(id,models,extra){
  document.getElementById(id).innerHTML=models.map(function(m){
    return '<span><i style="background:'+m.c+(m.op?';opacity:'+m.op:'')+'"></i>'+m.zh+' · '+m.en+'</span>';}).join("")+(extra||"");
}
/* 验证损失曲线 */
(function(){
  var svg=document.getElementById("val"),W=700,H=320,L=52,R=16,T=16,B=44,lo=1.58,hi=1.75;
  var y=yaxis(svg,L,R,T,B,W,H,lo,hi,[1.58,1.62,1.66,1.70,1.74],function(t){return t.toFixed(2);});
  var mx=D.val[D.val.length-1][0], x=function(s){return L+(W-L-R)*s/mx;};
  svg.appendChild(el("polyline",{points:D.val.map(function(p){return x(p[0])+","+y(p[1]);}).join(" "),fill:"none",stroke:C.good,"stroke-width":3}));
  D.val.forEach(function(p,i){
    svg.appendChild(el("circle",{cx:x(p[0]),cy:y(p[1]),r:4.5,fill:C.good}));
    if(i===0||i===D.val.length-1||i===1) svg.appendChild(el("text",{x:x(p[0])+(i===D.val.length-1?-6:8),y:y(p[1])-9,"text-anchor":i===D.val.length-1?"end":"start","font-family":MONO,"font-size":12,"font-weight":700,fill:C.navy},p[1].toFixed(3)));
    if(i<D.val.length-2||i===D.val.length-1) svg.appendChild(el("text",{x:x(p[0]),y:H-B+18,"text-anchor":"middle","font-family":MONO,"font-size":10.5,fill:C.taupe},p[0]));
  });
  svg.appendChild(el("text",{x:(L+W-R)/2,y:H-6,"text-anchor":"middle","font-size":12,fill:C.soft},"训练步数 · step"));
})();
/* 学习率扫描 */
(function(){
  var svg=document.getElementById("sweep"),W=360,H=320,L=52,R=10,T=16,B=44,lo=1.64,hi=1.69;
  var y=yaxis(svg,L,R,T,B,W,H,lo,hi,[1.64,1.65,1.66,1.67,1.68,1.69],function(t){return t.toFixed(2);});
  var gw=(W-L-R)/D.sweep.length, bw=gw*0.56;
  D.sweep.forEach(function(s,j){
    var cx=L+gw*j+gw/2, pick=s.lr==="1e-4";
    svg.appendChild(el("rect",{x:cx-bw/2,y:y(s.v),width:bw,height:y(lo)-y(s.v),fill:pick?C.good:C.grey,rx:3}));
    svg.appendChild(el("text",{x:cx,y:y(s.v)-7,"text-anchor":"middle","font-family":MONO,"font-size":12,"font-weight":700,fill:C.navy},s.v.toFixed(3)));
    svg.appendChild(el("text",{x:cx,y:H-B+18,"text-anchor":"middle","font-family":MONO,"font-size":12,"font-weight":pick?700:400,fill:pick?C.good:C.soft},s.lr+(pick?" ✓":"")));
  });
})();
/* 结果 */
var M5=[{zh:"原版",en:"base",c:C.base},{zh:"只做预训练",en:"pretrain only",c:C.grey},{zh:"A 只做微调",en:"fine-tune",c:C.orange},
        {zh:"B 预训练+微调",en:"pretrain+FT",c:C.good},{zh:"C 三步",en:"+ reasoning",c:C.purple}];
bars("resc",900,330,D.res,M5);
legend("legend-res",M5);
/* 去掉图像 */
var MI=[{zh:"A 有图",en:"A",c:C.orange},{zh:"A 无图",en:"A no image",c:C.orange,op:.35},{zh:"B 有图",en:"B",c:C.good},
        {zh:"B 无图",en:"B no image",c:C.good,op:.35},{zh:"C 有图",en:"C",c:C.purple},{zh:"C 无图",en:"C no image",c:C.purple,op:.35}];
bars("noimg",700,320,D.noimg,MI,{fill:.86,fs:9});
/* 成人胸片描述漏报 */
(function(){
  var svg=document.getElementById("cxr"),W=420,H=320,L=40,R=8,T=18,B=52;
  var M=[{zh:"原版",c:C.base},{zh:"6ds",c:C.gold},{zh:"只预训练",c:C.grey},{zh:"A",c:C.orange},{zh:"B",c:C.good},{zh:"C",c:C.purple}];
  var y=yaxis(svg,L,R,T,B,W,H,0,40,[0,10,20,30,40],function(t){return t;});
  var gw=(W-L-R)/M.length, bw=gw*0.62;
  D.cxr.forEach(function(v,j){
    var cx=L+gw*j+gw/2;
    svg.appendChild(el("rect",{x:cx-bw/2,y:y(v),width:bw,height:y(0)-y(v),fill:M[j].c,rx:3}));
    svg.appendChild(el("text",{x:cx,y:y(v)-6,"text-anchor":"middle","font-family":MONO,"font-size":11.5,"font-weight":700,fill:C.navy},v.toFixed(1)));
    svg.appendChild(el("text",{x:cx,y:H-B+20,"text-anchor":"middle","font-size":12,"font-weight":700,fill:M[j].c},M[j].zh));
  });
  svg.appendChild(el("line",{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:C.navy,"stroke-width":1.6}));
})();
/* 中文 */
var MZ=[{zh:"原版",en:"base",c:C.base},{zh:"S2-2datasets（已部署）",en:"deployed",c:C.gold},{zh:"A",en:"fine-tune",c:C.orange},
        {zh:"B",en:"pretrain+FT",c:C.good},{zh:"C",en:"+ reasoning",c:C.purple}];
bars("zhc",900,330,D.zh,MZ,{fill:.7,fs:10.5});
legend("legend-zh",MZ);
})();
</script>
""".replace("__DATA__", json.dumps(CHART, ensure_ascii=False))

EXTRA = """/* ---- week 6 ---- */
.flow{max-width:1160px;margin:36px auto 0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:34px;position:relative}
.flow .st{padding:18px 20px;position:relative}
.flow .st:not(:last-child)::after{content:"→";position:absolute;right:-28px;top:50%;transform:translateY(-50%);font-size:24px;color:var(--navy);font-weight:700}
.flow .tag{font-family:var(--font-num);font-size:11px;letter-spacing:.08em;color:var(--paper);display:inline-block;padding:2px 10px;margin-bottom:10px}
.flow p{font-size:14px;line-height:1.75;color:var(--ink-soft);margin:0}
.flow p.tr{margin-top:8px;font-size:13px}
table.mt{max-width:900px;width:100%;margin:30px auto 0;border-collapse:collapse;font-size:14px;background:var(--card);border:2px solid var(--navy)}
table.mt th,table.mt td{padding:9px 14px;border-bottom:1px solid var(--rule);text-align:center}
table.mt th{font-family:var(--font-num);font-size:12px;letter-spacing:.04em;color:var(--taupe);font-weight:600}
table.mt td:first-child,table.mt th:first-child{text-align:left}
table.mt tr.dim td{color:var(--taupe)}
table.ci td{font-family:var(--font-num);font-size:13px}
table.ci td:first-child{font-family:inherit}
table.ci td small{color:var(--taupe);font-size:11px}
table.ci td.sig{color:var(--c-red);font-weight:700}
table.ci td.ns{color:var(--ink-soft)}
.card.wide{max-width:1000px;margin:36px auto 0;padding:18px 20px 12px}
.card.wide .ttl{font-family:var(--font-num);font-size:12px;letter-spacing:.06em;color:var(--taupe);margin-bottom:8px}
.card.wide svg{width:100%;height:auto;display:block;overflow:visible}
#zh .duo-legend,#res .duo-legend{max-width:1000px}
@media (max-width:900px){.flow{grid-template-columns:1fr;gap:30px}.flow .st:not(:last-child)::after{content:"↓";right:50%;top:auto;bottom:-30px;transform:translateX(50%)}}
"""
html = ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Topic 6 第 6 周 · Week 6</title>\n'
        + re.search(r'<link rel="stylesheet"[^\n]*\n', W5).group(0)
        + "<style>\n" + css + EXTRA + "</style>\n</head>\n<body>\n" + BODY + JS + pager + "</body>\n</html>\n")
out = REPO / "docs/weekly/week6/Topic6_WeeklyReport6.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8", newline="\n")
print("written", out, len(html))

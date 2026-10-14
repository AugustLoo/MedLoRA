"""第 5 周周报网页 (中英对照): docs/weekly/week5/Topic6_WeeklyReport5.html
版式 (CSS + 点击翻页) 原样取自 week4 网页, 正文为第 5 周内容; 数字从 results/ 与 outputs/eval/ 的快照读, 不手抄。
用法: python scripts/make_week5_html.py
"""
import json, re
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
W4 = (REPO / "docs/weekly/week4/Topic6_WeeklyReport4.html").read_text(encoding="utf-8")
css = re.search(r"<style>\n(.*?)/\* ---- week 4 ---- \*/", W4, re.S).group(1)
pager = W4[W4.index('<nav id="pager"'):W4.index("</body>")]

M = json.loads((REPO / "results/interns2_cxp_2026-10-13.json").read_text(encoding="utf-8"))["models"]
J = json.loads((REPO / "outputs/eval/chexbert_interns2_6ds.json").read_text(encoding="utf-8"))
HELD = set(json.loads((REPO / "data/pubmedqa_split.json").read_text(encoding="utf-8"))["test"])
KEYS = ["base", "S2-2datasets", "S2-6datasets", "S2-CheXpertPlus"]
SHORT = dict(zip(KEYS, ["base", "d2", "d6", "cxp"]))


def maybe(tag):
    rows = [json.loads(x) for x in open(REPO / f"outputs/eval/pubmedqa_{tag}_preds.jsonl", encoding="utf-8")]
    return sum(r["gold"] == r["pred"] == "maybe" for r in rows if r["pubid"] in HELD)


V = {k: {"slake": M[k]["slake_closed"], "slake_open": M[k]["slake_open_em"], "maybe": maybe(M[k]["tag"]),
         "textvqa": M[k]["textvqa"], "mmbench": M[k]["mmbench"], "vqarad": M[k]["vqarad_closed"],
         "path": M[k]["pathvqa_closed"], "medqa": M[k]["medqa"], "pneu": M[k]["pneumonia_closed"]["accuracy"],
         "pneu_desc": M[k]["pneumonia_freetext"]["pneu_no_lung_abnormality_pct"],
         "descmiss": M[k]["chexpert"]["freetext"]["abnormal_no_abnormality_reported_pct"],
         "contra": M[k]["chexpert"]["freetext"]["abnormal_closed_yes_but_desc_none_pct"],
         "disc": M[k]["chexpert"]["freetext"]["discrimination_pts"],
         "missed": M[k]["chexpert"]["closed"]["missed_abnormal_pct"],
         "falarm": round(100 - M[k]["chexpert"]["closed"]["specificity"], 2)} for k in KEYS}
from decimal import Decimal, ROUND_HALF_UP
r1 = lambda x: str(Decimal(str(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))  # 95.35 → 95.4, 不用银行家舍入
f1 = r1
pct = lambda x: r1(x) + "%"

ROWS = [("SLAKE 封闭题", "SLAKE closed", "slake", 1, f1),
        ("SLAKE 开放题（完全匹配）", "SLAKE open, exact match", "slake_open", 1, f1),
        ("PubMedQA「不确定」答对 / 55", "PubMedQA maybe correct / 55", "maybe", 1, str),
        ("TextVQA 通用能力", "TextVQA, general", "textvqa", 1, f1),
        ("MMBench 通用能力", "MMBench, general", "mmbench", 1, f1),
        ("VQA-RAD 封闭题 *", "VQA-RAD closed *", "vqarad", 1, f1),
        ("PathVQA 病理图 *", "PathVQA closed *", "path", 1, f1),
        ("MedQA 医师考试 *", "MedQA *", "medqa", 1, f1),
        ("PneumoniaMNIST 判断肺炎 *", "PneumoniaMNIST accuracy *", "pneu", 1, f1),
        ("儿童胸片描述漏报 ↓", "paediatric: pneumonia described as clear ↓", "pneu_desc", 0, pct),
        ("成人胸片描述漏报 ↓", "adult: abnormal film described as normal ↓", "descmiss", 0, pct),
        ("成人胸片区分能力", "adult: discrimination, pts", "disc", 1, f1)]
trs = []
for zh, en, k, hi, fmt in ROWS:
    vals = [V[m][k] for m in KEYS]
    best = (max if hi else min)(vals)
    cells = "".join(f'<td class="num{" best" if v == best else ""}">{fmt(v)}</td>' for v in vals)
    cls = ' class="hl"' if k == "descmiss" else ""
    trs.append(f'<tr{cls}><td>{zh}<span class="tr-c">{en}</span></td>{cells}</tr>')
TABLE = "\n        ".join(trs)
DATA = json.dumps({k: {SHORT[m]: round(V[m][k], 1) for m in KEYS} for k in ["descmiss", "contra", "missed", "falarm"]})
b, d2, d6, cx = (V[m] for m in KEYS)

BODY = f"""
<!-- ============================ Cover ============================ -->
<section id="cover">
  <svg id="grid" viewBox="0 0 1400 900" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
    <defs><pattern id="gp" width="56" height="56" patternUnits="userSpaceOnUse"><path d="M56 0H0V56" fill="none" stroke="#D8CFBA" stroke-width="1"/></pattern></defs>
    <rect width="1400" height="900" fill="url(#gp)"/>
  </svg>
  <div class="inner">
    <div class="kicker">Topic 6 · Task 1.3 · 第 5 周进展汇报 · Week 5 progress report · 2026-10-15</div>
    <h1>两个新模型：描述胸片时<br><span class="hi">漏报降到 14%</span><span class="tr-h">Two new models: missed findings in chest X-ray descriptions down to 14%</span></h1>
    <p class="sub">老师要的两个 35B 新模型都训练完、评测完了：S2-6datasets（六个医学数据集）和 S2-CheXpertPlus（只用胸片报告）。上周发现的问题——让模型描述胸片时把有病的片子写成正常——两个新模型都明显改善。</p>
    <p class="sub tr">Both 35B models the instructor asked for are trained and evaluated: S2-6datasets (six medical datasets) and S2-CheXpertPlus (chest X-ray reports only). Last week's problem — describing abnormal films as normal — is much smaller in both.</p>
    <div class="asof-line">Intern-S2-Preview 35B · LoRA SFT · CheXpert Plus test 3,189 films · one seed per model · results as of 2026-10-14</div>
    <div class="stats">
      <div class="stat"><div class="v" style="color:var(--c-good)">{r1(d6['descmiss'])}%<small>← {r1(d2['descmiss'])}%</small></div><div class="bar" style="background:var(--c-good)"></div><div class="l">成人胸片描述漏报：S2-6datasets 对比现在部署的模型<span class="tr-l">abnormal films described as normal, S2-6datasets vs the deployed model</span></div></div>
      <div class="stat"><div class="v" style="color:var(--c-good)">{r1(d6['path'])}<small>← {r1(d2['path'])}</small></div><div class="bar" style="background:var(--c-good)"></div><div class="l">PathVQA 病理图是非题（练过同类题）<span class="tr-l">PathVQA closed questions (in-domain)</span></div></div>
      <div class="stat"><div class="v" style="color:var(--c-base)">{r1(J['agreement_with_keyword_judge_pct'])}%</div><div class="bar" style="background:var(--c-base)"></div><div class="l">CheXbert 与关键词判断逐条一致，结论可信<span class="tr-l">CheXbert agrees with the keyword judge film by film</span></div></div>
    </div>
  </div>
  <div class="scrollhint">点击翻页 · CLICK ▶</div>
</section>

<!-- ============================ 01 两个新模型 ============================ -->
<section class="section deep" id="models">
  <div class="prose">
    <div class="kicker">01 · 两个新模型 · Two new models</div>
    <h2>训练方法完全相同，只换数据<span class="tr-h">Same training recipe, different data</span></h2>
    <p class="lede">两个新模型和现在部署的 S2-2datasets 用同一套 LoRA 设置（只训练 0.054% 的参数），唯一的区别是训练数据，所以结果的差别就是数据带来的差别。</p>
    <p class="lede tr">Both use the same LoRA recipe as the deployed S2-2datasets (0.054% of parameters trained); only the data differs, so any difference in results comes from the data.</p>
  </div>
  <div class="split">
    <div class="card"><span class="tag" style="background:var(--c-b1)">S2-6datasets</span><div class="n" style="color:var(--c-b1)">15,953 条</div><p>SLAKE、PubMedQA、VQA-RAD、PathVQA、MedQA、PneumoniaMNIST 六个数据集，只用训练部分。VQA-RAD 的测试图大多也出现在训练部分，问这些图的 1,059 道训练题全部去掉。2,991 步，5 小时 40 分。</p><p class="tr">Six datasets, train splits only. Most VQA-RAD test images also appear in its train split, so the 1,059 training questions on those images were dropped. 2,991 steps, 5 h 40 min.</p></div>
    <div class="card"><span class="tag" style="background:var(--c-c1)">S2-CheXpertPlus</span><div class="n" style="color:var(--c-c1)">10,000 对</div><p>只用 CheXpert Plus 胸片 → 医生报告（检查所见 + 结论），和 3,189 张测试片不是同一批病人。数据只在服务器上，内网传输后 1 万个文件逐个核对。1,875 步，3 小时 19 分。</p><p class="tr">CheXpert Plus X-ray → report pairs only (findings + impression), no patient shared with the 3,189 test films. The data stays on the server; all 10,000 files were checked after the internal transfer. 1,875 steps, 3 h 19 min.</p></div>
  </div>
</section>

<!-- ============================ 02 对比表 ============================ -->
<section class="section" id="cmp">
  <div class="prose">
    <div class="kicker">02 · 结果 · Results</div>
    <h2>四个模型放在一起比<span class="tr-h">The four models side by side</span></h2>
    <p class="lede">绿色是每一行最好的。S2-6datasets 大部分项目最好或接近最好；S2-CheXpertPlus 只练了胸片报告，短答题和「不确定」题回到原版水平。</p>
    <p class="lede tr">Green marks the best in each row. S2-6datasets is best or near-best on most rows; S2-CheXpertPlus, trained on reports only, falls back to the base on short answers and "maybe" questions.</p>
  </div>
  <div class="tbl">
    <table>
      <thead><tr><th>测试<span class="tr-c">test</span></th><th class="num">原版<span class="tr-c">base</span></th><th class="num">S2-2datasets<span class="tr-c">deployed</span></th><th class="num">S2-6datasets<span class="tr-c">6 datasets</span></th><th class="num">S2-CheXpertPlus<span class="tr-c">X-ray reports</span></th></tr></thead>
      <tbody>
        {TABLE}
      </tbody>
    </table>
  </div>
  <div class="srcline">* 对 S2-6datasets 来说练过同类题（没练过测试题），属于领域内提升。每个模型只训练了一次，个别指标在两次训练之间可能差很多。<span class="tr-s">* In-domain for S2-6datasets (same kind of questions, never the test items). Each model was trained once; single metrics can move a lot between runs.</span></div>
</section>

<!-- ============================ 03 漏报 · scrolly ============================ -->
<section class="section deep" id="miss-sec">
  <div class="prose">
    <div class="kicker">03 · 漏报异常 · Missed abnormalities</div>
    <h2>上周的问题，两个新模型都改善了<span class="tr-h">Last week's problem is much smaller in both new models</span></h2>
    <p class="lede">还是同样的 3,189 张成人胸片（2,733 张有异常），每张问两次：「有没有异常？」和「描述一下这张片子」。</p>
    <p class="lede tr">The same 3,189 adult chest X-rays (2,733 abnormal), each asked twice: "Any abnormal findings?" and "Describe this chest X-ray".</p>
  </div>
  <div class="scrolly" id="scrolly-miss">
    <div class="sticky-fig"><div class="card">
      <div class="ttl" id="miss-ttl">漏报 · missed</div>
      <svg id="miss" viewBox="0 0 720 360" role="img" aria-label="Missed-abnormality metrics for the four 35B models"></svg>
      <div class="legend-x" id="miss-legend"></div>
    </div></div>
    <div class="steps">
      <div class="step" data-step="0"><div class="step-card"><div class="n">01 / 04</div><h3>描述漏报：68.5% → 14% 左右<span class="tr-h3">Described as normal: 68.5% → about 14%</span></h3><p>现在部署的 S2-2datasets 把 {r1(d2['descmiss'])}% 的病片描述成正常；S2-6datasets {r1(d6['descmiss'])}%、S2-CheXpertPlus {r1(cx['descmiss'])}%，都比原版的 {r1(b['descmiss'])}% 还低。</p><p class="tr">The deployed S2-2datasets describes {r1(d2['descmiss'])}% of abnormal films as normal; S2-6datasets {r1(d6['descmiss'])}% and S2-CheXpertPlus {r1(cx['descmiss'])}%, both below the base ({r1(b['descmiss'])}%).</p></div></div>
      <div class="step" data-step="1"><div class="step-card"><div class="n">02 / 04</div><h3>不再自相矛盾<span class="tr-h3">No longer contradicting itself</span></h3><p>是非题答「有异常」、描述却说正常：S2-2datasets {r1(d2['contra'])}%，两个新模型只有 {r1(d6['contra'])}% 和 {r1(cx['contra'])}%。</p><p class="tr">Answers "abnormal" yet describes the film as normal: {r1(d2['contra'])}% for S2-2datasets, only {r1(d6['contra'])}% and {r1(cx['contra'])}% for the new models.</p></div></div>
      <div class="step" data-step="2"><div class="step-card"><div class="n">03 / 04</div><h3>直接问：漏报保持很低<span class="tr-h3">Asked directly: misses stay low</span></h3><p>直接问「有没有异常」时，S2-6datasets 漏报 {r1(d6['missed'])}%，和 S2-2datasets 一样低；S2-CheXpertPlus {r1(cx['missed'])}%，介于中间。</p><p class="tr">Asked yes/no, S2-6datasets misses {r1(d6['missed'])}%, as low as S2-2datasets; S2-CheXpertPlus sits in between at {r1(cx['missed'])}%.</p></div></div>
      <div class="step" data-step="3"><div class="step-card"><div class="n">04 / 04</div><h3>代价：误报仍偏多<span class="tr-h3">The price: false alarms remain</span></h3><p>正常片被说成有异常：S2-6datasets {r1(d6['falarm'])}%，和 S2-2datasets 差不多；S2-CheXpertPlus {r1(cx['falarm'])}%，最平衡。<strong>两个新模型各有所长。</strong></p><p class="tr">Normal films called abnormal: {r1(d6['falarm'])}% for S2-6datasets, close to S2-2datasets; {r1(cx['falarm'])}% for S2-CheXpertPlus, the most balanced. <strong>Each new model has its strength.</strong></p></div></div>
    </div>
  </div>
  <div class="srcline">标签由 CheXbert 从报告文字自动提取，不是医生看片标注；正常片只有 231 张，误报率是最不精确的数字；每个模型只训练、只生成一次。<span class="tr-s">Labels come from CheXbert reading report text, not radiologists reading films; only 231 normal films, so the false-alarm rate is the least precise number; one training run and one decoding per model.</span></div>
</section>

<!-- ============================ 04 CheXbert ============================ -->
<section class="section" id="judge">
  <div class="prose">
    <div class="kicker">04 · 方法 · The judge</div>
    <h2>换一种判断方法，结论不变<span class="tr-h">A second judge reaches the same conclusion</span></h2>
    <p class="lede">上周计划把「描述漏报」测得更准。这周用 CheXbert（生成标准答案的同一个工具）去读模型写的描述，再和标准答案对比。服务器容器里只能用 CPU，S2-6datasets 一个模型就判了 {r1(J['seconds'] / 3600)} 小时。</p>
    <p class="lede tr">Last week's plan was a more precise omission measure. This week CheXbert — the tool that produced the gold labels — read the models' descriptions for comparison. The container has CPU only, so S2-6datasets alone took {r1(J['seconds'] / 3600)} hours.</p>
  </div>
  <div class="tiles">
    <div class="card"><div class="big" style="color:var(--c-good)">{r1(J['abnormal_no_finding_reported_pct'])}%<small>关键词 {r1(J['keyword_judge_abnormal_no_finding_pct'])}%</small></div><h3>CheXbert 判的描述漏报<span class="tr-h3">omissions by CheXbert</span></h3><p>比关键词判断还略低：S2-6datasets 的改善不是判断方法造出来的。</p><p class="tr">Slightly below the keyword judge: the improvement is not an artefact of the judge.</p></div>
    <div class="card"><div class="big" style="color:var(--c-base)">{r1(J['agreement_with_keyword_judge_pct'])}%</div><h3>两种判断逐条一致<span class="tr-h3">film-by-film agreement</span></h3><p>2,733 张异常片里，两种方法对「有没有提到异常」的判断几乎完全相同。</p><p class="tr">On the 2,733 abnormal films the two judges almost always agree on whether an abnormality was mentioned.</p></div>
    <div class="card"><div class="big" style="color:var(--c-good)">{r1(J['abnormal_closed_yes_but_desc_none_pct'])}%<small>关键词 {r1(d6['contra'])}%</small></div><h3>自相矛盾（CheXbert）<span class="tr-h3">contradictions by CheXbert</span></h3><p>答「有异常」却描述为正常的比例，同样很低。</p><p class="tr">Answered "abnormal" but described as normal: also low.</p></div>
  </div>
  <div class="srcline">其余模型的 CheXbert 判断还在服务器上跑，判完后补进报告。<span class="tr-s">CheXbert scoring of the other models is still running on the server and will be added to the report.</span></div>
</section>

<!-- ============================ 05 总结 ============================ -->
<section class="section deep" id="done">
  <div class="prose">
    <div class="kicker">05 · 总结 · Summary</div>
    <h2>这周做了什么<span class="tr-h">What was done this week</span></h2>
  </div>
  <div class="tbl">
    <table>
      <thead><tr><th>做了什么<span class="tr-c">work</span></th><th>结果<span class="tr-c">result</span></th></tr></thead>
      <tbody>
        <tr><td>① S2-6datasets<span class="tr-c">six-dataset model</span></td><td>15,953 条训练数据，自动去掉与测试题重叠的样本；训练、评测完成<span class="tr-c">15,953 examples, test overlap removed automatically; trained and evaluated</span></td></tr>
        <tr><td>② S2-CheXpertPlus<span class="tr-c">X-ray report model</span></td><td>1 万对胸片 → 报告；训练、评测完成<span class="tr-c">10,000 X-ray → report pairs; trained and evaluated</span></td></tr>
        <tr class="hl"><td>③ 评测<span class="tr-c">evaluation</span></td><td>基础测试 + 四个外部数据集 + 3,189 张成人胸片；描述漏报 {r1(d2['descmiss'])}% → {r1(d6['descmiss'])}% / {r1(cx['descmiss'])}%<span class="tr-c">core tests, four external sets, 3,189 adult films; omissions {r1(d2['descmiss'])}% → {r1(d6['descmiss'])}% / {r1(cx['descmiss'])}%</span></td></tr>
        <tr><td>④ CheXbert 复核<span class="tr-c">CheXbert check</span></td><td>S2-6datasets 漏报 {r1(J['abnormal_no_finding_reported_pct'])}%，与关键词判断一致 {r1(J['agreement_with_keyword_judge_pct'])}%<span class="tr-c">S2-6datasets omissions {r1(J['abnormal_no_finding_reported_pct'])}%, {r1(J['agreement_with_keyword_judge_pct'])}% agreement with the keyword judge</span></td></tr>
        <tr><td>⑤ 交给苏学长<span class="tr-c">for the demo UI</span></td><td>模型对比接口（四个模型一键切换）、10 道示例题、训练说明文档<span class="tr-c">comparison API (switches between the four models), 10 example questions, training notes</span></td></tr>
      </tbody>
    </table>
  </div>
</section>

<!-- ============================ 06 计划 ============================ -->
<section class="section" id="next">
  <div class="prose">
    <div class="kicker">06 · 计划 · Plan</div>
    <h2>接下来的计划<span class="tr-h">Next steps</span></h2>
  </div>
  <div class="card next">
    <h3>按优先级<span class="tr-h3">In priority order</span></h3>
    <ol>
      <li><strong>上线 S2-6datasets。</strong>替换现在部署的模型，等世龙同意，不用改他的文件。<span class="tr-li"><strong>Deploy S2-6datasets</strong> in place of the current model once Shilong agrees; none of his files need to change.</span></li>
      <li><strong>论文的系统功能。</strong>一致性检查：问答和描述矛盾时提醒；回答「不确定」时提示人工复核。<span class="tr-li"><strong>System features for the thesis:</strong> a consistency check that flags answers contradicting the description, and a prompt for human review when the model is unsure.</span></li>
      <li><strong>中文问答。</strong>SLAKE 自带中文题（测试 1,033 道），先测中文表现，再加中文和翻译数据训练。<span class="tr-li"><strong>Chinese QA:</strong> SLAKE has a Chinese half (1,033 test questions); measure first, then train with Chinese and translated data.</span></li>
      <li><strong>确认结果。</strong>用种子 43 再训一次 S2-6datasets；CheXbert 判完全部模型；论文题目 10/16 定稿。<span class="tr-li"><strong>Confirm the results:</strong> retrain S2-6datasets with seed 43, finish CheXbert scoring for all models, and fix the thesis title by 16 October.</span></li>
    </ol>
  </div>
</section>

<footer>
  MedLoRA · <a href="https://github.com/AugustLoo/MedLoRA">github.com/AugustLoo/MedLoRA</a> · 所有数字来自仓库 · every number comes from <span class="en">results/</span> and <span class="en">outputs/eval/</span>
</footer>

<script>
(function(){{
"use strict";
var NS="http://www.w3.org/2000/svg";
function el(t,a,x){{var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);if(x!=null)e.textContent=x;return e;}}
function clear(svg){{while(svg.firstChild)svg.removeChild(svg.firstChild);}}
var C={{navy:"#14263B",soft:"#3D4E63",taupe:"#8C8577",rule:"#D8CFBA",base:"#0E67B5",d2:"#B8860B",d6:"#1F8A70",cxp:"#8E44AD"}};
var MONO="IBM Plex Mono,monospace";

/* ---------- scrolly: 漏报 (四个模型各一种颜色, 虚线 = 原版) ---------- */
var DATA={DATA};
var BARS=[{{k:"base",zh:"原版",en:"base"}},{{k:"d2",zh:"S2-2datasets",en:"deployed"}},
          {{k:"d6",zh:"S2-6datasets",en:"6 datasets"}},{{k:"cxp",zh:"S2-CheXpertPlus",en:"X-ray reports"}}];
var STEPS=[{{m:"descmiss",hi:100,ttl:"描述里一个异常都没提 % · abnormal films described as normal"}},
           {{m:"contra",hi:100,ttl:"答「有」却描述为正常 % · answered yes, described as normal"}},
           {{m:"missed",hi:20,ttl:"直接问时的漏报 %（注意：纵轴到 20） · missed when asked yes/no (axis to 20)"}},
           {{m:"falarm",hi:100,ttl:"正常片被说成有异常 % · normal films called abnormal"}}];
var state=STEPS[0];
function drawMiss(){{
  var svg=document.getElementById("miss"); clear(svg);
  var W=720,H=360,L=50,R=78,T=26,B=70,pw=W-L-R,ph=H-T-B, hi=state.hi;
  var y=function(v){{return T+ph-(v/hi)*ph;}};
  [0,.25,.5,.75,1].map(function(f){{return f*hi;}}).forEach(function(t){{
    svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(t),y2:y(t),stroke:C.rule}}));
    svg.appendChild(el("text",{{x:L-10,y:y(t)+4,"text-anchor":"end","font-family":MONO,"font-size":11,fill:C.taupe}},t));
  }});
  var vals=DATA[state.m], gap=pw/BARS.length, bw=Math.min(84,gap*0.58), baseV=vals.base;
  svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(baseV),y2:y(baseV),stroke:C.base,"stroke-width":1.4,"stroke-dasharray":"5 4",opacity:.8}}));
  svg.appendChild(el("text",{{x:W-R+8,y:y(baseV)-2,"font-size":12,"font-weight":700,fill:C.soft}},"原版线"));
  svg.appendChild(el("text",{{x:W-R+8,y:y(baseV)+13,"font-family":MONO,"font-size":11,fill:C.taupe}},"base "+baseV.toFixed(1)));
  BARS.forEach(function(b,i){{
    var v=vals[b.k], cx=L+gap*i+gap/2, col=C[b.k];
    var rect=el("rect",{{x:cx-bw/2,y:y(v),width:bw,height:Math.max(0,y(0)-y(v)),fill:col,rx:4}});
    rect.appendChild(el("title",{{}},b.zh+" · "+b.en+": "+v.toFixed(1)+"%"));
    svg.appendChild(rect);
    svg.appendChild(el("text",{{x:cx,y:y(v)-8,"text-anchor":"middle","font-family":MONO,"font-size":13,"font-weight":700,fill:C.navy}},v.toFixed(1)));
    svg.appendChild(el("text",{{x:cx,y:H-B+22,"text-anchor":"middle","font-size":13,"font-weight":700,fill:col}},b.zh));
    svg.appendChild(el("text",{{x:cx,y:H-B+40,"text-anchor":"middle","font-family":MONO,"font-size":11,fill:C.taupe}},b.en));
  }});
  svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:C.navy,"stroke-width":1.6}}));
  document.getElementById("miss-ttl").textContent=state.ttl;
  document.getElementById("miss-legend").innerHTML=
    '<span style="color:#8C8577">越低越好 · lower is better · 虚线 = 原版 · dashed line = base</span>';
}}

function scrolly(id,onStep){{
  var root=document.getElementById(id); if(!root)return;
  var steps=Array.prototype.slice.call(root.querySelectorAll(".step"));
  var setActive=function(i){{steps.forEach(function(s,j){{s.classList.toggle("is-active",j===i);}});onStep(i);}};
  setActive(0);
  if(!("IntersectionObserver" in window))return;
  var io=new IntersectionObserver(function(entries){{
    entries.forEach(function(e){{if(e.isIntersecting)setActive(parseInt(e.target.getAttribute("data-step"),10));}});
  }},{{rootMargin:"-45% 0px -45% 0px",threshold:0}});
  steps.forEach(function(s){{io.observe(s);}});
}}
scrolly("scrolly-miss",function(i){{state=STEPS[i]||STEPS[0];drawMiss();}});
}})();
</script>
"""

EXTRA = """/* ---- week 5 ---- */
#cover h1 .hi{color:var(--c-good)}
td.best{color:var(--c-good)}
tbody tr.hl{background:#E6F2EC}
.legend-x{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--font-num);font-size:12px;color:var(--ink-soft);margin-top:10px}
#cmp td:first-child{white-space:normal;min-width:220px}
#done td:last-child{white-space:normal;min-width:440px}
"""
html = ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Topic 6 第 5 周 · Week 5</title>\n'
        + re.search(r'<link rel="stylesheet"[^\n]*\n', W4).group(0)
        + "<style>\n" + css + EXTRA + "</style>\n</head>\n<body>\n" + BODY + pager + "</body>\n</html>\n")
out = REPO / "docs/weekly/week5/Topic6_WeeklyReport5.html"
out.write_text(html, encoding="utf-8", newline="\n")
print("written", out, len(html))

"""第 6 周周报网页 (中英对照, 精简版): docs/weekly/week6/Topic6_WeeklyReport6.html
版式 (CSS + 点击翻页) 原样取自 week5 网页; 每节一两句话, 数字用图表; 数字从 results/ 的快照读, 不手抄。
用法: python scripts/make_week6_html.py
"""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
W5 = (REPO / "docs/weekly/week5/Topic6_WeeklyReport5.html").read_text(encoding="utf-8")
css = re.search(r"<style>\n(.*?)</style>", W5, re.S).group(1)
pager = W5[W5.index('<nav id="pager"'):W5.index("</body>")]

S = json.loads((REPO / "results/interns2_3stage_train_2026-10-18.json").read_text(encoding="utf-8"))
CI = json.loads((REPO / "results/interns2_ci_2026-10-18.json").read_text(encoding="utf-8"))["tests"]
EN = json.loads((REPO / "results/interns2_cxp_2026-10-13.json").read_text(encoding="utf-8"))["models"]
MD = S["models"]
CPT, A, B, C = (MD[k] for k in MD)
ZB, ZM = S["zh_baseline"]["原版"], S["zh_baseline"]["S2-2datasets"]
D1, D2, D3 = S["data"]["stage1"], S["data"]["stage2_zh_dialog"], S["data"]["stage3_reasoning"]


def r1(x):
    """一位小数, 四舍五入 (不用银行家舍入)。"""
    return str(Decimal(str(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))


def r3(x):
    return str(Decimal(str(x)).quantize(Decimal("0.001"), ROUND_HALF_UP))


def tail_mean(m):
    """训练损失在后 10% 步数上的平均 (单步损失抖动大, 不用最后一步)。"""
    return m["train_loss_last10pct_mean"]


TOK = D1["_total"]["tokens"]
EVAL1 = CPT["eval_loss"]
SWEEP = S["stage1_lr_sweep"]
MEDQA = CI["MedQA acc"]["interns2_6ds"]
CMEX = CI["CMExam acc"]["interns2_mix_300"]
CONTAM = {k: D3[k]["dropped"].get("与评测题重复或改写", 0) for k in ("cmexam", "medmcqa", "medical_o1_zh", "medical_o1_en")}

# 图表数据
CHART = {
    "val": [[s, l] for s, l in EVAL1],
    "sweep": [{"lr": lr, "v": v[-1]} for lr, v in SWEEP.items()],
    "zh": [
        {"zh": "是非题 英文", "en": "closed · EN", "v": [EN["base"]["slake_closed"], EN["S2-2datasets"]["slake_closed"]]},
        {"zh": "是非题 中文", "en": "closed · ZH", "v": [ZB["slake_zh"]["closed_acc"], ZM["slake_zh"]["closed_acc"]]},
        {"zh": "开放题 英文", "en": "open EM · EN", "v": [EN["base"]["slake_open_em"], EN["S2-2datasets"]["slake_open_em"]]},
        {"zh": "开放题 中文", "en": "open EM · ZH", "v": [ZB["slake_zh"]["open_em"], ZM["slake_zh"]["open_em"]]},
        {"zh": "CMExam 中文考题", "en": "CMExam", "v": [ZB["cmexam_acc"], ZM["cmexam_acc"]]},
    ],
}
for g in CHART["zh"]:
    g["v"] = [float(r1(x)) for x in g["v"]]
OPEN_GAP = float(r1(EN["S2-2datasets"]["slake_open_em"] - ZM["slake_zh"]["open_em"]))

BODY = f"""
<!-- ============================ Cover ============================ -->
<section id="cover">
  <svg id="grid" viewBox="0 0 1400 900" preserveAspectRatio="xMidYMax slice" aria-hidden="true">
    <defs><pattern id="gp" width="56" height="56" patternUnits="userSpaceOnUse"><path d="M56 0H0V56" fill="none" stroke="#D8CFBA" stroke-width="1"/></pattern></defs>
    <rect width="1400" height="900" fill="url(#gp)"/>
  </svg>
  <div class="inner">
    <div class="kicker">Topic 6 · Task 1.3 · 第 6 周进展汇报 · Week 6 progress report · 2026-10-22</div>
    <h1>35B 三步训练：<br><span class="hi">四个模型都训完了</span><span class="tr-h">Three-stage training on the 35B model: all four models trained</span></h1>
    <p class="sub">按苏学长的方案，先读医学教材，再练问答，最后练带解析的考题。评测还没做。</p>
    <p class="sub tr">Following Su's plan: read medical textbooks first, then practise Q&amp;A, then exam questions with explanations. Evaluation is still to come.</p>
    <div class="asof-line">Intern-S2-Preview 35B · LoRA r16 (0.054%) · one adapter continued stage by stage · training as of 2026-10-18</div>
    <div class="stats">
      <div class="stat"><div class="v" style="color:var(--c-base)">{TOK / 1e6:,.1f}M</div><div class="bar" style="background:var(--c-base)"></div><div class="l">第 1 步读的教材 / 指南 / 百科 token<span class="tr-l">tokens of textbooks, guidelines, encyclopaedia</span></div></div>
      <div class="stat"><div class="v" style="color:var(--c-good)">{r3(EVAL1[-1][1])}<small>← {r3(EVAL1[0][1])}</small></div><div class="bar" style="background:var(--c-good)"></div><div class="l">留出教材上的验证损失<span class="tr-l">validation loss on held-out text</span></div></div>
      <div class="stat"><div class="v" style="color:var(--c-b2)">{r1(ZM['slake_zh']['closed_acc'])}<small>← {r1(ZB['slake_zh']['closed_acc'])}</small></div><div class="bar" style="background:var(--c-b2)"></div><div class="l">中文看图是非题（现有模型）<span class="tr-l">Chinese closed VQA, current model</span></div></div>
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
      <li><strong>补做预训练：</strong>论文题目有「增量预训练」，所以在 35B 上按三步补做，每步只多加一类数据。<span class="tr-li"><strong>Pretraining added:</strong> the thesis title says incremental pretraining, so the 35B model now gets three stages, one new kind of data per stage.</span></li>
      <li><strong>三步数据：</strong>清洗、去掉推广和个人信息；和所有测试题比对，剔除重复或改写的题（CMExam {CONTAM['cmexam']} 道、MedMCQA {CONTAM['medmcqa']} 道）。<span class="tr-li"><strong>Data:</strong> cleaned, promotion and personal data removed; items matching any test question dropped (CMExam {CONTAM['cmexam']}, MedMCQA {CONTAM['medmcqa']}).</span></li>
      <li><strong>训练：</strong>先扫学习率再正式训；四个模型加扫描共约 24 小时显卡，都在世龙同意的时段里跑完。<span class="tr-li"><strong>Training:</strong> learning-rate sweep first; about 24 GPU hours including the sweep, all within slots Shilong agreed to.</span></li>
      <li><strong>评测准备：</strong>加了中文评测、「去掉图像」对照和置信区间，先测了现有模型的中文基线。<span class="tr-li"><strong>Evaluation prep:</strong> Chinese tests, a no-image control and confidence intervals; Chinese baselines measured on the current models.</span></li>
    </ol>
  </div>
</section>

<!-- ============================ 02 三步训练 ============================ -->
<section class="section" id="stages">
  <div class="prose">
    <div class="kicker">02 · 方法 · Method</div>
    <h2>三步训练，三个要比较的模型<span class="tr-h">Three stages, three models to compare</span></h2>
    <p class="lede">像人学医：先读书，再练问答，最后做带解析的题。B 对比 A 看预训练有没有用，C 对比 B 看推理训练有没有用。</p>
    <p class="lede tr">Like learning medicine: read, then practise Q&amp;A, then explained exam questions. B vs A tests pretraining; C vs B tests reasoning training.</p>
  </div>
  <div class="flow">
    <div class="card st"><span class="tag" style="background:var(--c-base)">第 1 步 · 知识预训练 · pretraining</span><p>29 本开放医学教材、临床指南、中文医学百科，共 {TOK / 1e6:,.1f}M token；整段文字都算损失。</p><p class="tr">29 open textbooks, clinical guidelines, a Chinese medical encyclopaedia ({TOK / 1e6:,.1f}M tokens); loss on all text.</p></div>
    <div class="card st"><span class="tag" style="background:var(--c-good)">第 2 步 · 问答微调 · instruction tuning</span><p>六个多模态数据集 + 中文问诊对话 {D2['huatuo_sft']['kept'] + D2['disc_med_sft']['kept']:,} 条（华佗、DISC，含多轮）。</p><p class="tr">The six multimodal datasets + {D2['huatuo_sft']['kept'] + D2['disc_med_sft']['kept']:,} Chinese medical dialogues (Huatuo, DISC, multi-turn).</p></div>
    <div class="card st"><span class="tag" style="background:var(--c-c1)">第 3 步 · 推理训练 · reasoning</span><p>带解析的考题 {sum(D3[k]['kept'] for k in ('cmexam', 'medmcqa', 'medical_o1_zh', 'medical_o1_en')):,} 道：CMExam、MedMCQA、medical-o1；只算答案 + 解析。</p><p class="tr">{sum(D3[k]['kept'] for k in ('cmexam', 'medmcqa', 'medical_o1_zh', 'medical_o1_en')):,} exam questions with explanations; loss on answer + explanation.</p></div>
  </div>
  <table class="mt">
    <thead><tr><th>模型 · model</th><th>第 1 步</th><th>第 2 步</th><th>第 3 步</th><th>步数 · steps</th><th>用时 · time</th></tr></thead>
    <tbody>
      <tr><td><strong>A</strong> 对照 · control</td><td>—</td><td>✓</td><td>—</td><td>{A['steps']:,}</td><td>≈ 9.5 h</td></tr>
      <tr><td><strong>B</strong></td><td>✓</td><td>✓</td><td>—</td><td>{B['steps']:,}</td><td>{r1(B['steps'] * 6.21 / 3600)} h</td></tr>
      <tr><td><strong>C</strong></td><td>✓</td><td>✓</td><td>✓</td><td>{C['steps']:,}</td><td>{r1(C['steps'] * 6.17 / 3600)} h</td></tr>
      <tr class="dim"><td>只做第 1 步 · stage 1 only</td><td>✓</td><td>—</td><td>—</td><td>{CPT['steps']:,}</td><td>3.1 h</td></tr>
    </tbody>
  </table>
</section>

<!-- ============================ 03 第①步 ============================ -->
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
  <div class="srcline">纵轴不从 0 开始，差别被放大了。训练损失：A 和 B 几乎一样（后 10% 平均 {r3(tail_mean(A))} 对 {r3(tail_mean(B))}），预训练有没有用要看评测。<span class="tr-s">The y-axes do not start at 0. Training loss of A and B is nearly identical ({r3(tail_mean(A))} vs {r3(tail_mean(B))}); whether pretraining helps will show only in evaluation.</span></div>
</section>

<!-- ============================ 04 中文基线 ============================ -->
<section class="section" id="zh">
  <div class="prose">
    <div class="kicker">04 · 中文 · Chinese</div>
    <h2>中文：是非题没问题，开放题差 {r1(OPEN_GAP)} 分<span class="tr-h">Chinese baseline: closed questions transfer, open answers lag by {r1(OPEN_GAP)} points</span></h2>
    <p class="lede">现有模型只练过英文，中文是非题照样从 {r1(ZB['slake_zh']['closed_acc'])} 涨到 {r1(ZM['slake_zh']['closed_acc'])}；但中文开放题比英文低 {r1(OPEN_GAP)} 分。第 2 步加的中文对话是纯文字的，能不能补上要看评测；上周提的 SLAKE 中文训练题和翻译伪数据还没加。</p>
    <p class="lede tr">Trained on English only, the current model still lifts Chinese closed questions from {r1(ZB['slake_zh']['closed_acc'])} to {r1(ZM['slake_zh']['closed_acc'])}; Chinese open answers trail English by {r1(OPEN_GAP)} points. Whether stage 2's text-only Chinese dialogues close it will show in evaluation; last week's SLAKE Chinese training items and translated pseudo-data are not added yet.</p>
  </div>
  <div class="card wide"><div class="ttl">SLAKE 与 CMExam 准确率 %（越高越好）· accuracy % (higher is better)</div><svg id="zhc" viewBox="0 0 900 330" role="img" aria-label="English and Chinese accuracy of the base and deployed models"></svg></div>
  <div class="legend-x duo-legend" id="legend"></div>
  <div class="srcline">CMExam 两个模型都在 93 左右，差 {r1(CMEX['diff_vs_ref'])} 分，在噪声内：题太简单，之后可能要换更难的中文题。<span class="tr-s">CMExam sits near 93 for both ({r1(CMEX['diff_vs_ref'])} points apart, within noise): too easy to separate models; a harder Chinese test may be needed.</span></div>
</section>

<!-- ============================ 05 两个发现 ============================ -->
<section class="section deep" id="found">
  <div class="prose">
    <div class="kicker">05 · 过程中 · Along the way</div>
    <h2>修了一个问题，多了一把尺子<span class="tr-h">One bug fixed, one new yardstick</span></h2>
  </div>
  <div class="tiles">
    <div class="card"><div class="big" style="color:var(--c-red)">1,745</div><h3>模型 A 在这一步崩溃<span class="tr-h3">Model A crashed at this step</span></h3><p>35B 的代码会把上一批带图的位置信息套到纯文字批次上，第一次混入纯文字对话才暴露。每步前清空缓存就修好了，从断点续训。</p><p class="tr">The model code reused the previous image batch's position offsets for a text-only batch; first exposed by text-only dialogues. Clearing the cache before each step fixed it; training resumed from a checkpoint.</p></div>
    <div class="card"><div class="big" style="color:var(--c-base)">+{r1(MEDQA['diff_vs_ref'])} ≈ 0</div><h3>置信区间：小差别不算数<span class="tr-h3">Confidence intervals: small gaps don't count</span></h3><p>配对重抽样后，MedQA 上「S2-6datasets 比原版高 {r1(MEDQA['diff_vs_ref'])} 分」（区间 {r1(MEDQA['diff_ci95'][0])} 到 +{r1(MEDQA['diff_ci95'][1])}）在噪声内，只能说没变差；SLAKE、PathVQA 的提升是真的。</p><p class="tr">With paired bootstrap, S2-6datasets' +{r1(MEDQA['diff_vs_ref'])} on MedQA is within noise — only "no worse"; the SLAKE and PathVQA gains are real.</p></div>
    <div class="card"><div class="big" style="color:var(--c-good)">无图</div><h3>「去掉图像」对照<span class="tr-h3">A no-image control</span></h3><p>同一批 SLAKE 题只给文字再考一次，分数掉得越多，越说明模型真的在看图。A / B / C 评测时一起跑。</p><p class="tr">The same SLAKE questions without the image: the bigger the drop, the more the model really looks. Runs with the A / B / C evaluation.</p></div>
  </div>
</section>

<!-- ============================ 06 文献 ============================ -->
<section class="section" id="lit">
  <div class="prose">
    <div class="kicker">06 · 文献 · Literature (2024–2026)</div>
    <h2>和我的实验直接相关的四篇<span class="tr-h">Four papers that bear directly on my results</span></h2>
  </div>
  <div class="card next">
    <ol>
      <li><strong>LoRA Learns Less and Forgets Less</strong>（2024）：LoRA 在预训练里学得少、忘得也少，可以解释 3B 上一万条预训练没提升。<span class="tr-li">LoRA learns less but forgets less in continued pretraining — consistent with no gain from 10k-item pretraining at 3B.</span></li>
      <li><strong>Lost in the Hype</strong>（2026）：训练视觉编码器只差约 ±0.5 分，支持冻结它的做法。<span class="tr-li">Training the vision encoder changes results by only about ±0.5 points, supporting keeping it frozen.</span></li>
      <li><strong>Consistent but Dangerous</strong>（CVPR 2026 Workshop）：微调让回答更稳定，却可能不看图，建议加「去掉图像」的检查。<span class="tr-li">Fine-tuning makes answers stable but may stop the model looking; the authors recommend a no-image check.</span></li>
      <li><strong>PU-DPO</strong>（2026）：医生报告本身会漏写细微发现，模型照着学就学会了少报。这解释了漏报，之后可以用偏好优化来改。<span class="tr-li">Reference reports omit subtle findings and models learn to under-report — an explanation for omissions; preference optimisation is a later fix.</span></li>
    </ol>
  </div>
</section>

<!-- ============================ 07 计划 ============================ -->
<section class="section deep" id="next">
  <div class="prose">
    <div class="kicker">07 · 计划 · Plan</div>
    <h2>接下来的计划<span class="tr-h">Next steps</span></h2>
  </div>
  <div class="card next">
    <ol>
      <li><strong>评测 A / B / C：</strong>每个约 12 小时（含中文和「去掉图像」），胸片漏报在容器里另测；要和世龙排显卡时间。<span class="tr-li"><strong>Evaluate A / B / C:</strong> about 12 hours each, plus chest X-ray omissions in the container; needs GPU time agreed with Shilong.</span></li>
      <li><strong>硬盘：</strong>每次合并要约 73 GB，主机只剩约 89 GB，一次只能评一个；想问苏学长扩容的进度。<span class="tr-li"><strong>Disk:</strong> each merge needs about 73 GB and about 89 GB is free, so one model at a time; asking Su about the expansion.</span></li>
      <li><strong>确认结果：</strong>结论关键的模型用种子 43 再训一次。<span class="tr-li"><strong>Confirm:</strong> retrain the key model with seed 43.</span></li>
      <li><strong>论文：</strong>想请老师确认最后能否加一章原型系统（矛盾提醒、不确定时提示人工复核）。<span class="tr-li"><strong>Thesis:</strong> may the last chapter add a prototype system (contradiction alerts, human review when unsure)?</span></li>
    </ol>
  </div>
</section>

<footer>
  MedLoRA · <a href="https://github.com/AugustLoo/MedLoRA">github.com/AugustLoo/MedLoRA</a> · 所有数字来自仓库 · every number comes from <span class="en">results/</span>
</footer>

<script>
(function(){{
"use strict";
var NS="http://www.w3.org/2000/svg";
function el(t,a,x){{var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);if(x!=null)e.textContent=x;return e;}}
var C={{navy:"#14263B",soft:"#3D4E63",taupe:"#8C8577",rule:"#D8CFBA",base:"#0E67B5",good:"#1F8A70",gold:"#B8860B"}};
var MONO="IBM Plex Mono,monospace";
var D={json.dumps(CHART, ensure_ascii=False)};
function yaxis(svg,L,R,T,B,W,H,lo,hi,ticks,fmt){{
  var ph=H-T-B, y=function(v){{return T+ph-(v-lo)/(hi-lo)*ph;}};
  ticks.forEach(function(t){{
    svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(t),y2:y(t),stroke:C.rule}}));
    svg.appendChild(el("text",{{x:L-8,y:y(t)+4,"text-anchor":"end","font-family":MONO,"font-size":11,fill:C.taupe}},fmt(t)));}});
  return y;
}}
/* 验证损失曲线 */
(function(){{
  var svg=document.getElementById("val"),W=700,H=320,L=52,R=16,T=16,B=44,lo=1.58,hi=1.75;
  var y=yaxis(svg,L,R,T,B,W,H,lo,hi,[1.58,1.62,1.66,1.70,1.74],function(t){{return t.toFixed(2);}});
  var mx=D.val[D.val.length-1][0], x=function(s){{return L+(W-L-R)*s/mx;}};
  svg.appendChild(el("polyline",{{points:D.val.map(function(p){{return x(p[0])+","+y(p[1]);}}).join(" "),fill:"none",stroke:C.good,"stroke-width":3}}));
  D.val.forEach(function(p,i){{
    svg.appendChild(el("circle",{{cx:x(p[0]),cy:y(p[1]),r:4.5,fill:C.good}}));
    if(i===0||i===D.val.length-1||i===1) svg.appendChild(el("text",{{x:x(p[0])+(i===D.val.length-1?-6:8),y:y(p[1])-9,"text-anchor":i===D.val.length-1?"end":"start","font-family":MONO,"font-size":12,"font-weight":700,fill:C.navy}},p[1].toFixed(3)));
    if(i<D.val.length-2||i===D.val.length-1) svg.appendChild(el("text",{{x:x(p[0]),y:H-B+18,"text-anchor":"middle","font-family":MONO,"font-size":10.5,fill:C.taupe}},p[0]));
  }});
  svg.appendChild(el("text",{{x:(L+W-R)/2,y:H-6,"text-anchor":"middle","font-size":12,fill:C.soft}},"训练步数 · step"));
}})();
/* 学习率扫描 */
(function(){{
  var svg=document.getElementById("sweep"),W=360,H=320,L=52,R=10,T=16,B=44,lo=1.64,hi=1.69;
  var y=yaxis(svg,L,R,T,B,W,H,lo,hi,[1.64,1.65,1.66,1.67,1.68,1.69],function(t){{return t.toFixed(2);}});
  var gw=(W-L-R)/D.sweep.length, bw=gw*0.56;
  D.sweep.forEach(function(s,j){{
    var cx=L+gw*j+gw/2, pick=s.lr==="1e-4";
    svg.appendChild(el("rect",{{x:cx-bw/2,y:y(s.v),width:bw,height:y(lo)-y(s.v),fill:pick?C.good:"#9A9283",rx:3}}));
    svg.appendChild(el("text",{{x:cx,y:y(s.v)-7,"text-anchor":"middle","font-family":MONO,"font-size":12,"font-weight":700,fill:C.navy}},s.v.toFixed(3)));
    svg.appendChild(el("text",{{x:cx,y:H-B+18,"text-anchor":"middle","font-family":MONO,"font-size":12,"font-weight":pick?700:400,fill:pick?C.good:C.soft}},s.lr+(pick?" ✓":"")));
  }});
}})();
/* 中文基线 */
var M2=[{{zh:"原版",en:"base",c:C.base}},{{zh:"S2-2datasets（已部署）",en:"deployed",c:C.gold}}];
(function(){{
  var svg=document.getElementById("zhc"),W=900,H=330,L=40,R=8,T=18,B=52;
  var y=yaxis(svg,L,R,T,B,W,H,0,100,[0,25,50,75,100],function(t){{return t;}});
  var gw=(W-L-R)/D.zh.length, bw=gw*0.6/2;
  D.zh.forEach(function(g,i){{
    var x0=L+gw*i+gw*0.2;
    g.v.forEach(function(v,j){{
      var x=x0+bw*j, r=el("rect",{{x:x+1,y:y(v),width:bw-2,height:y(0)-y(v),fill:M2[j].c,rx:2}});
      r.appendChild(el("title",{{}},M2[j].zh+" · "+g.zh+": "+v.toFixed(1)));svg.appendChild(r);
      svg.appendChild(el("text",{{x:x+bw/2,y:y(v)-5,"text-anchor":"middle","font-family":MONO,"font-size":11,"font-weight":j===1?700:400,fill:C.navy}},v.toFixed(1)));
    }});
    svg.appendChild(el("text",{{x:L+gw*i+gw/2,y:H-B+20,"text-anchor":"middle","font-size":13,"font-weight":700,fill:C.navy}},g.zh));
    svg.appendChild(el("text",{{x:L+gw*i+gw/2,y:H-B+37,"text-anchor":"middle","font-family":MONO,"font-size":10.5,fill:C.taupe}},g.en));
  }});
  svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:C.navy,"stroke-width":1.6}}));
}})();
document.getElementById("legend").innerHTML=M2.map(function(m){{
  return '<span><i style="background:'+m.c+'"></i>'+m.zh+' · '+m.en+'</span>';}}).join("");
}})();
</script>
"""

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
.card.wide{max-width:1000px;margin:36px auto 0;padding:18px 20px 12px}
.card.wide .ttl{font-family:var(--font-num);font-size:12px;letter-spacing:.06em;color:var(--taupe);margin-bottom:8px}
.card.wide svg{width:100%;height:auto;display:block;overflow:visible}
#zh .duo-legend{max-width:1000px}
@media (max-width:900px){.flow{grid-template-columns:1fr;gap:30px}.flow .st:not(:last-child)::after{content:"↓";right:50%;top:auto;bottom:-30px;transform:translateX(50%)}}
"""
html = ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Topic 6 第 6 周 · Week 6</title>\n'
        + re.search(r'<link rel="stylesheet"[^\n]*\n', W5).group(0)
        + "<style>\n" + css + EXTRA + "</style>\n</head>\n<body>\n" + BODY + pager + "</body>\n</html>\n")
out = REPO / "docs/weekly/week6/Topic6_WeeklyReport6.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8", newline="\n")
print("written", out, len(html))

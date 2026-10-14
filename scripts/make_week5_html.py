"""第 5 周周报网页 (中英对照, 精简版): docs/weekly/week5/Topic6_WeeklyReport5.html
版式 (CSS + 点击翻页) 原样取自 week4 网页; 每节一两句话, 对比用图表; 数字从 results/ 与 outputs/eval/ 的快照读, 不手抄。
用法: python scripts/make_week5_html.py
"""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
W4 = (REPO / "docs/weekly/week4/Topic6_WeeklyReport4.html").read_text(encoding="utf-8")
css = re.search(r"<style>\n(.*?)/\* ---- week 4 ---- \*/", W4, re.S).group(1)
pager = W4[W4.index('<nav id="pager"'):W4.index("</body>")]

M = json.loads((REPO / "results/interns2_cxp_2026-10-13.json").read_text(encoding="utf-8"))["models"]
J = json.loads((REPO / "outputs/eval/chexbert_interns2_6ds.json").read_text(encoding="utf-8"))
HELD = set(json.loads((REPO / "data/pubmedqa_split.json").read_text(encoding="utf-8"))["test"])
KEYS = ["base", "S2-2datasets", "S2-6datasets", "S2-CheXpertPlus"]
SLAKE_ZH_TEST = sum(r["q_lang"] == "zh" for r in json.loads((REPO / "data/raw/SLAKE/test.json").read_text(encoding="utf-8")))


def r1(x):
    """一位小数, 四舍五入 (95.35 → 95.4, 不用银行家舍入)。"""
    return str(Decimal(str(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))


def maybe(tag):
    rows = [json.loads(x) for x in open(REPO / f"outputs/eval/pubmedqa_{tag}_preds.jsonl", encoding="utf-8")]
    held = [r for r in rows if r["pubid"] in HELD]
    return sum(r["gold"] == r["pred"] == "maybe" for r in held), sum(r["gold"] == "maybe" for r in held)


V = {}
for k in KEYS:
    ok, n = maybe(M[k]["tag"])
    V[k] = {"slake": M[k]["slake_closed"], "maybe": 100 * ok / n, "maybe_n": f"{ok}/{n}",
            "path": M[k]["pathvqa_closed"], "medqa": M[k]["medqa"], "pneu": M[k]["pneumonia_closed"]["accuracy"],
            "descmiss": M[k]["chexpert"]["freetext"]["abnormal_no_abnormality_reported_pct"]}
b, d2, d6, cx = (V[k] for k in KEYS)

# 图表数据: 准确率 (越高越好) 与描述漏报 (越低越好)
ACC = [("slake", "SLAKE 是非题", "SLAKE closed"), ("maybe", "敢说「不确定」", "PubMedQA maybe"),
       ("path", "病理图 *", "PathVQA *"), ("medqa", "医师考试 *", "MedQA *"), ("pneu", "儿童肺炎 *", "PneumoniaMNIST *")]
CHART = {"acc": [{"zh": zh, "en": en, "v": [float(r1(V[k][key])) for k in KEYS]} for key, zh, en in ACC],
         "miss": [float(r1(V[k]["descmiss"])) for k in KEYS]}

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
    <p class="sub">老师要的两个 35B 新模型都训练完、评测完了。</p>
    <p class="sub tr">Both new 35B models the instructor asked for, S2-6datasets and S2-CheXpertPlus, are trained and evaluated.</p>
    <div class="asof-line">Intern-S2-Preview 35B · LoRA SFT · one seed per model · results as of 2026-10-14</div>
    <div class="stats">
      <div class="stat"><div class="v" style="color:var(--c-good)">{r1(d6['descmiss'])}%<small>← {r1(d2['descmiss'])}%</small></div><div class="bar" style="background:var(--c-good)"></div><div class="l">成人胸片描述漏报<span class="tr-l">abnormal X-rays described as normal</span></div></div>
      <div class="stat"><div class="v" style="color:var(--c-good)">{r1(d6['path'])}<small>← {r1(d2['path'])}</small></div><div class="bar" style="background:var(--c-good)"></div><div class="l">病理图问答 PathVQA<span class="tr-l">pathology VQA accuracy</span></div></div>
      <div class="stat"><div class="v" style="color:var(--c-base)">{r1(J['agreement_with_keyword_judge_pct'])}%</div><div class="bar" style="background:var(--c-base)"></div><div class="l">CheXbert 复核一致<span class="tr-l">agreement with CheXbert</span></div></div>
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
      <li><strong>S2-6datasets：</strong>用六个医学数据集的训练部分（15,953 条）训练，和测试题重叠的样本自动去掉。<span class="tr-li"><strong>S2-6datasets:</strong> trained on the train splits of six medical datasets (15,953 examples), test overlap removed.</span></li>
      <li><strong>S2-CheXpertPlus：</strong>只用 1 万对胸片 → 医生报告训练。<span class="tr-li"><strong>S2-CheXpertPlus:</strong> trained on 10,000 chest X-ray → report pairs only.</span></li>
      <li><strong>评测：</strong>四个模型全部测完，描述漏报用 CheXbert 复核，与原判断 {r1(J['agreement_with_keyword_judge_pct'])}% 一致。<span class="tr-li"><strong>Evaluation:</strong> all four models tested; CheXbert re-check agrees {r1(J['agreement_with_keyword_judge_pct'])}% with the original judge.</span></li>
      <li><strong>交给苏学长：</strong>模型对比接口、示例题、训练说明，用于演示界面。<span class="tr-li"><strong>For the demo UI:</strong> comparison API, example questions and training notes.</span></li>
    </ol>
  </div>
</section>

<!-- ============================ 02 对比图 ============================ -->
<section class="section" id="cmp">
  <div class="prose">
    <div class="kicker">02 · 结果 · Results</div>
    <h2>四个模型对比<span class="tr-h">Four models compared</span></h2>
    <p class="lede">S2-6datasets 整体最好；S2-CheXpertPlus 只在胸片描述上好。</p>
    <p class="lede tr">S2-6datasets is best overall; S2-CheXpertPlus helps only on X-ray descriptions.</p>
  </div>
  <div class="duo">
    <div class="card"><div class="ttl">答题准确率 %（越高越好）· accuracy % (higher is better)</div><svg id="acc" viewBox="0 0 700 340" role="img" aria-label="Accuracy of the four models on five tests"></svg></div>
    <div class="card"><div class="ttl">胸片描述漏报 %（越低越好）· described as normal % (lower is better)</div><svg id="miss" viewBox="0 0 360 340" role="img" aria-label="Missed findings in descriptions for the four models"></svg></div>
  </div>
  <div class="legend-x duo-legend" id="legend"></div>
  <div class="srcline">* 对 S2-6datasets 是练过的同类题（没练过测试题）。「敢说不确定」= PubMedQA 中答案为 maybe 的 55 题答对的比例。每个模型只训练一次。<span class="tr-s">* In-domain for S2-6datasets (never the test items). "maybe" = share of the 55 PubMedQA maybe questions answered correctly. One training run per model.</span></div>
</section>

<!-- ============================ 03 中文数据 · 讨论 ============================ -->
<section class="section deep" id="zh">
  <div class="prose">
    <div class="kicker">03 · 讨论 · For discussion</div>
    <h2>加入中文数据<span class="tr-h">Adding Chinese data</span></h2>
    <p class="lede">苏学长发现：把英文问题换成中文，模型就容易答错。原因是训练数据全是英文，而系统落地后以中文问答为主。</p>
    <p class="lede tr">Su found that the models often answer wrongly once an English question is asked in Chinese: all training data is English, yet the deployed system will mainly be used in Chinese.</p>
  </div>
  <div class="split">
    <div class="card"><span class="tag" style="background:var(--c-b1)">真实中文题 · real</span><div class="n" style="color:var(--c-b1)">SLAKE 中文</div><p>SLAKE 本身中英双语，之前只用了英文一半：中文测试题 {SLAKE_ZH_TEST:,} 道，训练题约 4,900 道。</p><p class="tr">SLAKE is bilingual and only its English half was used: {SLAKE_ZH_TEST:,} Chinese test and about 4,900 training questions.</p></div>
    <div class="card"><span class="tag" style="background:var(--c-b2)">翻译伪数据 · pseudo</span><div class="n" style="color:var(--c-b2)">翻译扩充</div><p>把 PathVQA、MedQA、VQA-RAD 等英文训练题翻译成中文，作为伪数据补充；用服务器上的开源模型翻译，再人工抽查。</p><p class="tr">Translate English training items (PathVQA, MedQA, VQA-RAD, …) into Chinese as pseudo-data, using an open model on the server, with a hand check.</p></div>
  </div>
  <div class="card next">
    <h3>计划与想讨论的问题<span class="tr-h3">Plan and open questions</span></h3>
    <ol>
      <li><strong>先测：</strong>用 SLAKE 的 {SLAKE_ZH_TEST:,} 道中文测试题考现有模型，量出中文差多少。<span class="tr-li"><strong>Measure first:</strong> run the current models on the {SLAKE_ZH_TEST:,} SLAKE Chinese test questions.</span></li>
      <li><strong>再训：</strong>在 S2-6datasets 基础上加入中文题，训一个中英双语版。<span class="tr-li"><strong>Then train</strong> a bilingual version on top of S2-6datasets.</span></li>
      <li><strong>讨论：</strong>翻译伪数据加多少合适？翻译的题能不能拿来考试，还是只用真实中文题考？<span class="tr-li"><strong>To discuss:</strong> how much pseudo-data to add, and whether translated items may be used for testing or only real Chinese questions.</span></li>
    </ol>
  </div>
</section>

<!-- ============================ 04 计划 ============================ -->
<section class="section" id="next">
  <div class="prose">
    <div class="kicker">04 · 计划 · Plan</div>
    <h2>接下来的计划<span class="tr-h">Next steps</span></h2>
  </div>
  <div class="card next">
    <ol>
      <li><strong>上线 S2-6datasets</strong>，替换现在部署的模型（等世龙同意）。<span class="tr-li"><strong>Deploy S2-6datasets</strong> in place of the current model, once Shilong agrees.</span></li>
      <li><strong>中文数据</strong>：先测，再训中英双语版。<span class="tr-li"><strong>Chinese data:</strong> measure, then train a bilingual version.</span></li>
      <li><strong>论文系统功能</strong>：问答与描述矛盾时提醒；回答「不确定」时提示人工复核。<span class="tr-li"><strong>Thesis system features:</strong> flag contradictions, and ask for human review when unsure.</span></li>
      <li><strong>确认结果</strong>：换一个随机种子重训；论文题目 10/16 定稿。<span class="tr-li"><strong>Confirm results</strong> with a second seed; fix the thesis title by 16 October.</span></li>
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
var C={{navy:"#14263B",soft:"#3D4E63",taupe:"#8C8577",rule:"#D8CFBA"}};
var MODELS=[{{zh:"原版",en:"base",c:"#0E67B5"}},{{zh:"S2-2datasets",en:"deployed",c:"#B8860B"}},
            {{zh:"S2-6datasets",en:"6 datasets",c:"#1F8A70"}},{{zh:"S2-CheXpertPlus",en:"X-ray reports",c:"#8E44AD"}}];
var MONO="IBM Plex Mono,monospace";
var D={json.dumps(CHART, ensure_ascii=False)};

function axes(svg,L,R,T,B,W,H,hi){{
  var ph=H-T-B, y=function(v){{return T+ph-(v/hi)*ph;}};
  [0,.25,.5,.75,1].forEach(function(f){{var t=f*hi;
    svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(t),y2:y(t),stroke:C.rule}}));
    svg.appendChild(el("text",{{x:L-8,y:y(t)+4,"text-anchor":"end","font-family":MONO,"font-size":11,fill:C.taupe}},t));}});
  return y;
}}
/* 左: 五项测试 × 四个模型 */
(function(){{
  var svg=document.getElementById("acc"),W=700,H=340,L=40,R=8,T=18,B=50;
  var y=axes(svg,L,R,T,B,W,H,100), gw=(W-L-R)/D.acc.length, bw=gw*0.8/4;
  D.acc.forEach(function(g,i){{
    var x0=L+gw*i+gw*0.1, best=Math.max.apply(null,g.v);
    g.v.forEach(function(v,j){{
      var x=x0+bw*j, r=el("rect",{{x:x+1,y:y(v),width:bw-2,height:y(0)-y(v),fill:MODELS[j].c,rx:2}});
      r.appendChild(el("title",{{}},MODELS[j].zh+" · "+g.zh+": "+v.toFixed(1)));svg.appendChild(r);
      svg.appendChild(el("text",{{x:x+bw/2,y:y(v)-5,"text-anchor":"middle","font-family":MONO,"font-size":9.5,
        "font-weight":v===best?700:400,fill:v===best?"#1F8A70":C.soft}},v.toFixed(1)));
    }});
    svg.appendChild(el("text",{{x:L+gw*i+gw/2,y:H-B+20,"text-anchor":"middle","font-size":13,"font-weight":700,fill:C.navy}},g.zh));
    svg.appendChild(el("text",{{x:L+gw*i+gw/2,y:H-B+37,"text-anchor":"middle","font-family":MONO,"font-size":10.5,fill:C.taupe}},g.en));
  }});
  svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:C.navy,"stroke-width":1.6}}));
}})();
/* 右: 描述漏报, 虚线 = 原版 */
(function(){{
  var svg=document.getElementById("miss"),W=360,H=340,L=40,R=8,T=18,B=50;
  var y=axes(svg,L,R,T,B,W,H,80), gw=(W-L-R)/4, bw=gw*0.62;
  svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(D.miss[0]),y2:y(D.miss[0]),stroke:MODELS[0].c,"stroke-width":1.3,"stroke-dasharray":"5 4",opacity:.8}}));
  D.miss.forEach(function(v,j){{
    var cx=L+gw*j+gw/2, r=el("rect",{{x:cx-bw/2,y:y(v),width:bw,height:y(0)-y(v),fill:MODELS[j].c,rx:3}});
    r.appendChild(el("title",{{}},MODELS[j].zh+": "+v.toFixed(1)+"%"));svg.appendChild(r);
    svg.appendChild(el("text",{{x:cx,y:y(v)-6,"text-anchor":"middle","font-family":MONO,"font-size":12,"font-weight":700,fill:C.navy}},v.toFixed(1)));
    svg.appendChild(el("text",{{x:cx,y:H-B+20,"text-anchor":"middle","font-size":12,"font-weight":700,fill:MODELS[j].c}},MODELS[j].zh.replace("S2-","")));
  }});
  svg.appendChild(el("line",{{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:C.navy,"stroke-width":1.6}}));
}})();
document.getElementById("legend").innerHTML=MODELS.map(function(m){{
  return '<span><i style="background:'+m.c+'"></i>'+m.zh+' · '+m.en+'</span>';}}).join("")+
  '<span style="color:#8C8577">绿色数字 = 该项最好 · green = best · 虚线 = 原版 · dashed = base</span>';
}})();
</script>
"""

EXTRA = """/* ---- week 5 ---- */
#cover h1 .hi{color:var(--c-good)}
.legend-x{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--font-num);font-size:12px;color:var(--ink-soft);margin-top:10px}
.legend-x i{display:inline-block;width:14px;height:12px;margin-right:6px;vertical-align:-1px;border:1px solid rgba(20,38,59,.35)}
.duo{max-width:1160px;margin:40px auto 0;display:grid;grid-template-columns:minmax(0,1.9fr) minmax(0,1fr);gap:22px}
.duo .card{padding:18px 20px 12px}
.duo .ttl{font-family:var(--font-num);font-size:12px;letter-spacing:.06em;color:var(--taupe);margin-bottom:8px}
.duo svg{width:100%;height:auto;display:block;overflow:visible}
.duo-legend{max-width:1160px;margin:14px auto 0}
.next ol{margin-top:4px}
#zh .next{margin-top:28px}
.split p.tr{margin-top:10px;font-size:13px}
@media (max-width:900px){.duo{grid-template-columns:1fr}}
"""
html = ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Topic 6 第 5 周 · Week 5</title>\n'
        + re.search(r'<link rel="stylesheet"[^\n]*\n', W4).group(0)
        + "<style>\n" + css + EXTRA + "</style>\n</head>\n<body>\n" + BODY + pager + "</body>\n</html>\n")
out = REPO / "docs/weekly/week5/Topic6_WeeklyReport5.html"
out.write_text(html, encoding="utf-8", newline="\n")
print("written", out, len(html))

"""docs/report.md → docs/report.pdf (mistune 转 HTML, Edge 无头打印, A4, 本机字体, 不连网)。

正文里的 \\citep{a,b} 换成参考文献列表里的编号 [3, 7]; 列表按 report.md 末尾的顺序编号。
用法: python scripts/make_report_pdf.py [输出 PDF 的绝对路径]   (Edge 只认绝对路径, 相对路径会静默不写文件)
"""
import pathlib
import re
import subprocess
import sys

import mistune

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "docs" / "report.md"
PDF = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO / "docs" / "report.pdf"
HTML = REPO / "outputs" / "report_print.html"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink:#16202C; --soft:#45505E; --rule:#D5D9DE; --tint:#F4F6F8; --accent:#0E67B5; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: Cambria, Georgia, "Times New Roman", "Microsoft YaHei", serif; color: var(--ink);
       font-size: 10.5pt; line-height: 1.5; margin: 0; background: #fff; }
h1 { font-size: 18pt; line-height: 1.25; margin: 0 0 6pt; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-bottom: 2pt; border-bottom: 0.75pt solid var(--rule); break-after: avoid; }
h3 { font-size: 11.5pt; margin: 13pt 0 4pt; break-after: avoid; }
p, li { margin: 4pt 0; text-align: justify; }
ul, ol { padding-left: 18pt; margin: 4pt 0; }
hr { border: 0; border-top: 0.75pt solid var(--rule); margin: 12pt 0; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.8pt; background: var(--tint);
       padding: 0.3pt 2.5pt; border-radius: 2pt; overflow-wrap: break-word; }
pre { background: var(--tint); border: 0.75pt solid var(--rule); border-radius: 3pt; padding: 6pt 8pt;
      white-space: pre-wrap; word-break: break-all; margin: 6pt 0; }
pre code { background: none; padding: 0; font-size: 7.8pt; line-height: 1.4; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 8.2pt; line-height: 1.3; }
tr { break-inside: avoid; }
th, td { border: 0.6pt solid var(--rule); padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #E9EDF1; font-weight: 700; }
td code, th code { font-size: 7.6pt; }
blockquote { margin: 8pt 0; padding: 5pt 9pt; border-left: 3pt solid var(--accent); background: var(--tint); }
blockquote p { margin: 0; }
img { display: block; max-width: 82%; max-height: 120mm; margin: 6pt auto 2pt; break-inside: avoid; }
"""

text = SRC.read_text(encoding="utf-8")
refs = re.findall(r"^(\d+)\. .*`([A-Za-z0-9]+)`\s*$", text, flags=re.M)
num = {key: int(n) for n, key in refs}


def cite(m):
    keys = [k.strip() for k in m.group(1).split(",")]
    if keys == ["key"]:  # 参考文献说明里举例的 `\citep{key}`, 原样保留
        return m.group(0)
    missing = [k for k in keys if k not in num]
    if missing:
        sys.exit(f"参考文献列表里没有: {missing}")
    return "[" + ", ".join(str(n) for n in sorted(num[k] for k in keys)) + "]"


body_md = re.sub(r"\\citep\{([^}]*)\}", cite, text)
body = mistune.create_markdown(plugins=["table", "strikethrough"], escape=False)(body_md)
# 行内代码 (路径) 只在 / 和 _ 后面允许换行; 代码块原样保留
parts = re.split(r"(<pre>.*?</pre>)", body, flags=re.S)
body = "".join(p if p.startswith("<pre>") else
               re.sub(r"<code>(.*?)</code>",
                      lambda m: "<code>" + m.group(1).replace("/", "/<wbr>").replace("_", "_<wbr>") + "</code>", p, flags=re.S)
               for p in parts)
base_href = "file:///" + str(SRC.parent).replace("\\", "/") + "/"  # md 里的相对图片路径
HTML.parent.mkdir(exist_ok=True)
HTML.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><base href="{base_href}">'
                f'<title>{SRC.stem}</title><style>{CSS}</style></head><body>{body}</body></html>',
                encoding="utf-8")

old = PDF.stat().st_mtime if PDF.is_file() else None
url = "file:///" + str(HTML).replace("\\", "/")
r = subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--user-data-dir=" + str(REPO / "outputs" / "edge_pdf_profile"),
                    f"--print-to-pdf={PDF}", url], capture_output=True, text=True, timeout=180)
if not PDF.is_file() or PDF.stat().st_mtime == old:
    sys.exit(f"PDF 没生成: {r.stderr[-500:]}")
print("written", PDF, PDF.stat().st_size, "bytes")

"""限时试用只走合并和服务, 失败时退出, 清理时保留补丁。所有命令只在临时目录里运行。"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/server/gpu_window.sh"
BASH = shutil.which("bash")
if Path("C:/Program Files/Git/bin/bash.exe").is_file():
    BASH = "C:/Program Files/Git/bin/bash.exe"
pytestmark = pytest.mark.skipif(not BASH, reason="需要 bash")


def run_functions(tmp_path, body):
    source = SOURCE.read_text(encoding="utf-8")
    functions = source[source.index("log() {"):source.index("\ncase $STEPS in")]
    root = tmp_path.as_posix()
    script = tmp_path / "case.sh"
    script.write_text(
        f"ROOT='{root}'\nREPO=$ROOT\nLOG=$ROOT/log\nJOB=''\nDEMO_DIR=''\n"
        "DEADLINE_AT='test deadline'\n"
        + functions
        + "\nlog() { echo \"$*\"; }\n"
        + body,
        encoding="utf-8", newline="\n",
    )
    return subprocess.run([BASH, str(script)], capture_output=True, text=True, encoding="utf-8")


@pytest.mark.parametrize("merge_rc,serve_rc,expected,calls", [(1, 0, 1, 1), (0, 1, 1, 2), (2, 0, 2, 1), (0, 0, 2, 2)])
def test_demo_failure_and_deadline(tmp_path, merge_rc, serve_rc, expected, calls):
    result = run_functions(tmp_path, f"""
run_until_deadline() {{
  printf '%s\\n' "$*" >> "$ROOT/calls"
  if [[ $* == *run_sft35.sh* ]]; then return {merge_rc}; fi
  return {serve_rc}
}}
left() {{ echo 0; }}
demo a interns2_s2a
rc=$?
echo "RESULT=$rc"
echo "OWNED=$DEMO_DIR"
""")
    assert result.returncode == 0, result.stderr
    assert f"RESULT={expected}" in result.stdout
    commands = (tmp_path / "calls").read_text().splitlines()
    assert len(commands) == calls
    assert "run_sft35.sh merge a" in commands[0]
    if calls == 2:
        assert "serve.sh start" in commands[1]
    assert "OWNED=" + tmp_path.as_posix() + "/merged/interns2_s2a" in result.stdout


def test_demo_refuses_existing_directory(tmp_path):
    existing = tmp_path / "merged/interns2_s2a"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("keep")
    result = run_functions(tmp_path, """
run_until_deadline() { echo 'UNEXPECTED'; return 99; }
demo a interns2_s2a
echo "RESULT=$? OWNED=$DEMO_DIR"
""")
    assert "RESULT=1 OWNED=\n" in result.stdout
    assert "UNEXPECTED" not in result.stdout
    assert marker.read_text() == "keep"


def test_restore_removes_only_owned_merge_and_stops_demo(tmp_path):
    owned = tmp_path / "merged/interns2_s2a"
    owned.mkdir(parents=True)
    (owned / "temporary.txt").write_text("temporary")
    adapter = tmp_path / "outputs/interns2_s2a/adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("permanent")
    result = run_functions(tmp_path, """
DEMO_DIR=$ROOT/merged/interns2_s2a
stop_job() { :; }
bash() { printf '%s\\n' "$*" >> "$ROOT/stops"; }
ss() { echo ':23333 '; }
restore
""")
    assert result.returncode == 0, result.stderr
    assert not owned.exists()
    assert adapter.read_text() == "permanent"
    assert "serve.sh stop" in (tmp_path / "stops").read_text()


@pytest.mark.parametrize("case,message", [
    ("multiple", "demo 一次只能试用一个模型"),
    ("missing", "缺少补丁或 adapter_config.json"),
    ("existing", "已存在, 不覆盖也不清理"),
    ("port", "23334 已有服务"),
    ("space", "至少要有 80 GB"),
    ("late", "至少要有 15 分钟"),
])
def test_preflight_refuses_before_touching_service(tmp_path, case, message):
    repo = tmp_path / "MedLoRA"
    adapter = repo / "outputs/interns2_s2a"
    adapter.mkdir(parents=True)
    (tmp_path / "merged").mkdir()
    if case != "missing":
        (adapter / "adapter_model.safetensors").write_text("keep")
        (adapter / "adapter_config.json").write_text("{}")
    if case == "existing":
        (tmp_path / "merged/interns2_s2a").mkdir()
    commands = tmp_path / "bin"
    commands.mkdir()
    stubs = {
        "ss": "echo ':23334 '" if case == "port" else ":",
        "df": "printf 'Avail\\n70G\\n'" if case == "space" else "printf 'Avail\\n90G\\n'",
        "date": "echo 1000" if case == "late" else "echo 100000000000",
        "sleep": "echo UNEXPECTED_WAIT; exit 99",
        "tmux": "echo UNEXPECTED_SERVICE; exit 99",
        "nvidia-smi": "echo UNEXPECTED_GPU; exit 99",
    }
    for name, body in stubs.items():
        target = commands / name
        target.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8", newline="\n")
        target.chmod(0o755)
    script = tmp_path / "window.sh"
    script.write_text(SOURCE.read_text(encoding="utf-8").replace(
        "ROOT=/home/ubuntu/chunqian", f"ROOT='{tmp_path.as_posix()}'"), encoding="utf-8", newline="\n")
    # 在 Bash 内设置路径, 保留 Windows 下盘符路径的正确含义。
    env = dict(os.environ, STEPS="demo", MODELS="a b" if case == "multiple" else "a", DEADLINE_AT="test")
    stub_path = commands.as_posix()
    if os.name == "nt":
        stub_path = "/" + stub_path[0].lower() + stub_path[2:]
    result = subprocess.run([BASH, "-c", 'export PATH="$1:$PATH"; exec bash "$2"',
                             "test", stub_path, script.as_posix()],
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=10)
    assert result.returncode == 1, result.stdout + result.stderr
    assert message in result.stdout
    assert "UNEXPECTED" not in result.stdout + result.stderr

"""이 프로젝트의 학습을 이 세션의 콘솔과 완전히 분리된(어떤 콘솔에도
연결되지 않은) 프로세스로 띄운다. Task Scheduler로 띄운 프로세스가 이
세션의 콘솔 신호(Ctrl+C 등)에 함께 반응해서 죽는 문제를 겪은 뒤 만든
스크립트 - subprocess.DETACHED_PROCESS 플래그로 콘솔 자체를 아예 할당받지
않게 한다.

사용법:
    python launch_detached.py -- train.py --sizes l x --resume
"""
import subprocess
import sys
from pathlib import Path

# DETACHED_PROCESS(콘솔 자체를 안 가짐)는 PyTorch DataLoader의 멀티프로세싱
# 워커 생성이 멈춰버리는 문제가 있어서, 대신 "새 콘솔을 따로 할당"받는
# CREATE_NEW_CONSOLE을 쓴다 — 이 세션의 콘솔과는 분리되면서도(Ctrl+C 등 신호를
# 같이 받지 않음) 워커 프로세스 스폰에 필요한 정상적인 콘솔 컨텍스트는 유지된다.
CREATE_NEW_CONSOLE = 0x00000010
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
PYTHON = r"C:\Users\user\miniconda3\python.exe"

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        raise SystemExit("사용법: python launch_detached.py -- <script.py> [args...]")

    log_path = ROOT / "train_l_x.log"
    log_f = open(log_path, "a", encoding="utf-8", errors="replace")

    # 새 콘솔은 할당하되(워커 스폰용) 화면에는 안 보이도록 숨긴다
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0  # SW_HIDE

    proc = subprocess.Popen(
        [PYTHON] + args,
        cwd=str(SCRIPTS_DIR),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB,
        startupinfo=startupinfo,
        close_fds=True,
    )
    print(f"launched detached pid={proc.pid}, logging to {log_path}")

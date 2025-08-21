# run_pipeline.py  (inside data_gen/)
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # .../data_gen
ROOT = HERE.parent                          # 项目根目录（与 data_gen 同级）
TRAIN = ROOT / "train"                  # 如果你的脚本放在 scripts 文件夹里

def run(cmd, cwd=HERE):
    print(f"[RUN] (cwd={cwd}) {cmd}")
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    py = sys.executable  # 当前解释器

    # 原有步骤（在 data_gen 下运行）
    #run([py, "update_qqq_options_dataset.py"])
    run([py, "QQQ_TQQQ_update.py"])
    run([py, "build_all_market_data.py"])
    run([py, "fetch_VIX.py"])

    print("[STEP] fetch_fear_and_greed_index.py x5")
    for i in range(1, 6):
        print(f"  -> Run {i}/5")
        run([py, "fetch_fear_and_greed_index.py"])

    # 新增：在“项目根目录或 scripts 目录”下跑上一级的脚本
    # 1）如果 make_put_signals_single.py 就在上一层根目录：
    run([py, str(TRAIN/ "make_put_signals_single.py")], cwd=ROOT)

    # 2）如果它在上一层的 scripts/ 目录里（更常见）：
    # run([py, str(SCRIPTS / "make_put_signals_single.py")], cwd=ROOT)

    print("[DONE] Pipeline completed.")

if __name__ == "__main__":
    main()
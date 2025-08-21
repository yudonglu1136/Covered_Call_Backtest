# run_pipeline.py  (inside data_gen/)
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent      # .../data_gen
ROOT = HERE.parent                          
TRAIN = ROOT / "train"                  

def run(cmd, cwd=HERE):
    print(f"[RUN] (cwd={cwd}) {cmd}")
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    py = sys.executable  # 当前解释器


    run([py, "update_qqq_options_dataset.py"])
    run([py, "QQQ_TQQQ_update.py"])
    run([py, "build_all_market_data.py"])
    run([py, "fetch_VIX.py"])

    print("[STEP] fetch_fear_and_greed_index.py x5")
    for i in range(1, 6):
        print(f"  -> Run {i}/5")
        run([py, "fetch_fear_and_greed_index.py"])


    run([py, str(TRAIN/ "make_put_signals_single.py")], cwd=ROOT)

    print("[DONE] Pipeline completed.")

if __name__ == "__main__":
    main()
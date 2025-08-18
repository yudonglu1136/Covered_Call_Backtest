# run_pipeline.py  (place this file inside data_gen/)
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # this is data_gen/

def run(cmd):
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, cwd=HERE, check=True)

def main():
    py = sys.executable  # current interpreter/env
    run([py, "update_qqq_options_dataset.py"])
    run([py, "QQQ_TQQQ_update.py"])
    run([py, "generate_put_signals.py"])

    print("[STEP] fetch_fear_and_greed_index.py x5")
    for i in range(1, 6):
        print(f"  -> Run {i}/5")
        run([py, "fetch_fear_and_greed_index.py"])

    print("[DONE] Pipeline completed.")

if __name__ == "__main__":
    main()

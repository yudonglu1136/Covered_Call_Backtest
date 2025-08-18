#!/usr/bin/env python3
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
env = ROOT / ".env"
example = ROOT / ".env.example"

if env.exists():
    print("[INFO] .env already exists. Review values if something fails.")
else:
    if not example.exists():
        raise SystemExit("[ERROR] .env.example is missing. Create it first.")
    shutil.copyfile(example, env)
    print("[OK] Created .env from .env.example. Fill in your API keys.")

print("Next:")
print("  1) Open .env and paste your keys")
print("  2) Either export them into the shell or use python-dotenv (see below)")

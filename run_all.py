"""Run the full pipeline end to end: clean -> features -> train."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"

for step in ["1_clean.py", "2_features.py", "3_train.py"]:
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, str(SCRIPTS / step)], check=True)

print("\nPipeline complete. Start the app with: python app/app.py")

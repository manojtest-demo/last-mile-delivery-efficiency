import os
from datetime import datetime

os.makedirs("analysis_artifacts", exist_ok=True)

# Example output file
with open("analysis_artifacts/data.txt", "w") as f:
    f.write("Hello from automation\n")

# ✅ Force Git change every run
with open("analysis_artifacts/_meta.txt", "w") as f:
    f.write(f"updated_at={datetime.utcnow().isoformat()}\n")
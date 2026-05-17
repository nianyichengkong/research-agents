import os
import re
from datetime import datetime


def save_report(content: str, topic: str, output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)

    safe_topic = re.sub(r'[\\/:*?"<>|]', "", topic)
    safe_topic = safe_topic[:50].strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.md"

    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path

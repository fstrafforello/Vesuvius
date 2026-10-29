"""Convert a '# %%' percent-format script into a .ipynb next to it (optionally executed)."""

import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

src = Path(sys.argv[1])
cells = []
for block in re.split(r"^# %%", src.read_text(encoding="utf-8"), flags=re.M)[1:]:
    header, _, body = block.partition("\n")
    if "[markdown]" in header:
        cells.append(new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines())))
    else:
        cells.append(new_code_cell(body.strip()))

nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = src.with_suffix(".ipynb")
if "--execute" in sys.argv:
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=900, resources={"metadata": {"path": str(src.parent)}}).execute()
nbformat.write(nb, out)
print("wrote", out)

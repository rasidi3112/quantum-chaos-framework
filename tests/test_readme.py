"""The README quick start must actually run (PLAN: criteria for completion)."""

import contextlib
import io
import pathlib
import re

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_readme_python_blocks_execute():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.S)
    assert blocks, "README has no python code block"
    for code in blocks:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            exec(compile(code, "README.md", "exec"), {})
        assert out.getvalue().strip()

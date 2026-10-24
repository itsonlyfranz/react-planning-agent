"""The README flow diagram must match the compiled LangGraph graph (offline, no API key)."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)


def test_readme_flow_diagram_matches_compiled_graph():
    readme = (ROOT / "README.md").read_text()
    match = render_graph.BLOCK.search(readme)
    assert match, "README.md is missing the generated flow-diagram block"
    assert match.group(0) == render_graph.render_block(), (
        "README flow diagram drifted from the compiled graph; run scripts/render_graph.py"
    )


def test_flow_diagram_contains_every_graph_node():
    mermaid = render_graph.render_mermaid()
    for node in ("gate", "think", "act", "reflect", "critique", "finish", "degrade"):
        assert re.search(rf"^\t{node}\(", mermaid, re.MULTILINE), node


def test_readme_has_hand_written_component_diagram():
    readme = (ROOT / "README.md").read_text()
    assert len(re.findall(r"```mermaid", readme)) == 2

"""The README flow diagram must match the compiled LangGraph graph (offline, no API key)."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)

NODE = re.compile(r"^\s*(\w+)\(", re.MULTILINE)
EDGE = re.compile(r"^\s*(\w+) (-->|-\.->) (\w+);", re.MULTILINE)


def test_readme_flow_diagram_matches_compiled_graph():
    readme = (ROOT / "README.md").read_text()
    match = render_graph.BLOCK.search(readme)
    assert match, "README.md is missing the generated flow-diagram block"
    block = match.group(0)
    graph = render_graph.build_agent().graph.get_graph()
    assert set(NODE.findall(block)) == set(graph.nodes), (
        "README flow diagram nodes drifted from the compiled graph; run scripts/render_graph.py"
    )
    assert {(src, arrow == "-.->", dst) for src, arrow, dst in EDGE.findall(block)} == {
        (e.source, e.conditional, e.target) for e in graph.edges
    }, "README flow diagram edges drifted from the compiled graph; run scripts/render_graph.py"

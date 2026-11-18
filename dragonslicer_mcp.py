"""MCP server (stdio) exposing the DragonSlicer pipeline to an agent: import -> arrange -> supports -> .goo -> printer.

Register it with:  claude mcp add dragonslicer -- python C:\\path\\to\\dragonslicer_mcp.py
Printer-moving tools (start / pause / resume / stop / delete) refuse unless confirm=true.
"""
import os
import sys

HERE = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)     # default project folder lives next to the program

from mcp.server.mcpserver import MCPServer  # noqa: E402

import functools  # noqa: E402

from core.agent_api import TOOLS, ProjectError  # noqa: E402

INSTRUCTIONS = """Slicer for ELEGOO resin printers. Typical flow: new_project -> add_model -> set_object (size/rotation) ->
arrange -> set_supports -> check_project -> send_project (export .goo + upload). Starting, pausing, stopping a print and
deleting files on the printer need confirm=true: only pass it after the user has explicitly agreed for that action.
Paths are absolute; `project` is a folder (default dragonslicer_project)."""

server = MCPServer("dragonslicer", instructions=INSTRUCTIONS)


def _safe(fn):
    """A problem the agent can fix comes back as {"error": ...} instead of an opaque tool failure."""
    @functools.wraps(fn)
    def run(*a, **k):
        try:
            return fn(*a, **k)
        except ProjectError as e:
            return {"error": str(e)}
    return run


for fn in TOOLS:
    server.add_tool(_safe(fn))

if __name__ == "__main__":
    server.run("stdio")

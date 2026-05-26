from mcp.server.fastmcp import FastMCP
import subprocess
import sys
import tempfile
import os

mcp = FastMCP("python-repl")

@mcp.tool()
async def execute_python_code(code: str) -> str:
    """Executes Python code and returns the result.
    Use this to run python code snippets and see their results"""

    print(f"🐍 MCP tool called with code: {code[:100]}...")
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".py", delete=False) as code_file:
        code_file.write(code)
        code_file.flush()
        temp_path = code_file.name

    result = subprocess.run(
        [sys.executable, temp_path],
        capture_output=True,
        text=True
    )
    os.unlink(temp_path)

    if result.returncode == 0:
        print(f"📤 MCP output: {result.stdout}")
        return result.stdout
    else:
        return f"Error: {result.stderr}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
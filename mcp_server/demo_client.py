"""Start the MCP server over stdio, list its tools and call two of them (end-to-end check)."""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(email):
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])
            for name, args in (("match_stats", {}), ("lookup_by_email", {"email": email})):
                res = await s.call_tool(name, args)
                text = res.content[0].text if res.content else str(res)
                print(f"\n{name}({args}) ->\n{text[:600]}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "test@example.com"))

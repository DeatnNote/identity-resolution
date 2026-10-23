"""
MCP server exposing the identity graph to an AI assistant or agent, read-only.

  python -m mcp_server.server          # stdio transport (e.g. for Claude Desktop / an MCP client)

Tools: lookup_by_email, lookup_by_phone, lookup_by_name_zip, get_identity, match_stats.
Every response contains masked fields only (see identity_tools.py).
"""
try:                                             # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:                              # mcp 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as _Server

from mcp_server.identity_tools import IdentityStore

server = _Server("identity-graph")
_store = None


def store():
    global _store
    if _store is None:
        _store = IdentityStore()
    return _store


@server.tool()
def lookup_by_email(email: str) -> dict:
    """Find identities that have a record with this email (normalised: case, gmail dots, +tags)."""
    return store().lookup_by_email(email)


@server.tool()
def lookup_by_phone(phone: str) -> dict:
    """Find identities that have a record with this phone number (any formatting)."""
    return store().lookup_by_phone(phone)


@server.tool()
def lookup_by_name_zip(first_name: str, last_name: str, zip5: str) -> dict:
    """Find identities by first name (nicknames resolved), last name and 5-digit zip."""
    return store().lookup_by_name_zip(first_name, last_name, zip5)


@server.tool()
def get_identity(identity_id: str) -> dict:
    """All source records linked to one identity, with masked contact fields."""
    return store().get_identity(identity_id)


@server.tool()
def match_stats() -> dict:
    """Summary of the latest published identity-resolution run."""
    return store().match_stats()


if __name__ == "__main__":
    server.run()

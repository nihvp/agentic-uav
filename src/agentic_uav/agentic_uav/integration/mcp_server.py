"""Integration layer: MCP server for external (non-flight) tools.

Phase 4 connects the reasoning layer to this server as an MCP client. Every
back-end here is a MOCK and says so in its description and output, so the LLM
and the mission trace never mistake it for real data.

Run standalone:  python3 -m agentic_uav.integration.mcp_server
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agentic-uav-integration")


@mcp.tool()
def weather_advisory(x: float, y: float) -> dict:
    """[MOCK] Local wind and visibility at map position (x, y). Not real weather data."""
    return {"mock": True, "wind_mps": 4.0, "gust_mps": 7.0, "visibility_km": 10.0}


@mcp.tool()
def emergency_alert_dispatch(severity: str, description: str, x: float, y: float) -> dict:
    """[MOCK] Send an incident alert to dispatch. Nothing is actually sent."""
    return {"mock": True, "accepted": True, "severity": severity, "location": [x, y]}


def main():
    mcp.run()


if __name__ == "__main__":
    main()

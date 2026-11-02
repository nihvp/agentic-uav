#!/usr/bin/env python3
"""
Integration Layer Node.
Infrastructure manager providing secure external service discovery and tool orchestration
following the Model Context Protocol (MCP) standard.
Exposes MCP services to the ROS 2 cognitive architecture.
"""

import json
import rclpy
from rclpy.node import Node

from agentic_uav_interfaces.srv import ExecuteMCPTool, ListMCPTools
from agentic_uav.integration.mcp_server_registry import MCPToolRegistry


class MCPIntegrationNode(Node):
    def __init__(self):
        super().__init__("mcp_integration_node")

        # Parameters
        self.declare_parameter("allowed_tokens", ["uav_reasoning_node_auth_token", "admin_secret_token"])
        tokens = self.get_parameter("allowed_tokens").get_parameter_value().string_array_value

        self.registry = MCPToolRegistry(allowed_tokens=list(tokens))

        # Services
        self.srv_list = self.create_service(
            ListMCPTools, "/integration/list_mcp_tools", self.handle_list_tools
        )
        self.srv_execute = self.create_service(
            ExecuteMCPTool, "/integration/execute_mcp_tool", self.handle_execute_tool
        )

        self.get_logger().info("MCP Integration Layer initialized. Registered tools: "
                               + ", ".join([t["name"] for t in self.registry.list_tools()]))

    def handle_list_tools(self, request, response):
        tools = self.registry.list_tools()
        response.tools_schema_json = json.dumps({"tools": tools}, indent=2)
        response.success = True
        return response

    def handle_execute_tool(self, request, response):
        tool_name = request.tool_name
        token = request.client_token

        try:
            arguments = json.loads(request.arguments_json) if request.arguments_json else {}
        except json.JSONDecodeError as err:
            response.success = False
            response.error_message = f"Invalid JSON arguments: {str(err)}"
            return response

        self.get_logger().info(f"Received MCP execution request: '{tool_name}' from client.")
        success, res_json, err_msg = self.registry.execute_tool(tool_name, arguments, client_token=token)

        response.success = success
        response.result_json = res_json
        response.error_message = err_msg
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MCPIntegrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

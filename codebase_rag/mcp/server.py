"""MCP server implementation for code-graph-rag.

This module provides the main MCP server that exposes code-graph-rag's
capabilities via the Model Context Protocol.
"""

import json
import os
import sys
from pathlib import Path

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from codebase_rag.config import settings
from codebase_rag.mcp.tools import create_mcp_tools_registry
from codebase_rag.services.graph_service import MemgraphIngestor
from codebase_rag.services.llm import CypherGenerator


def setup_logging() -> None:
    """Configure logging to stderr for MCP stdio transport."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def get_project_root() -> Path:
    """Get the project root from environment, settings, or parent process working directory.

    Priority order:
    1. TARGET_REPO_PATH environment variable (explicit configuration)
    2. TARGET_REPO_PATH from settings
    3. CLAUDE_PROJECT_ROOT environment variable (set by Claude Code)
    4. PWD environment variable (inherited from parent process/shell)
    5. Current working directory (fallback)

    Returns:
        Path to the target repository

    Raises:
        ValueError: If the resolved path is invalid
    """
    # Try explicit TARGET_REPO_PATH first
    repo_path: str | None = (
        os.environ.get("TARGET_REPO_PATH") or settings.TARGET_REPO_PATH
    )

    if not repo_path:
        # Try Claude Code project root env var
        repo_path = os.environ.get("CLAUDE_PROJECT_ROOT")

        if not repo_path:
            # Try PWD from parent process (often set by shells and reflects where command was run)
            repo_path = os.environ.get("PWD")

        if repo_path:
            logger.info(f"[GraphCode MCP] Using inferred project root: {repo_path}")
        else:
            # Last resort: current working directory
            repo_path = str(Path.cwd())
            logger.info(
                f"[GraphCode MCP] No project root configured, using current directory: {repo_path}"
            )

    project_root = Path(repo_path).resolve()

    if not project_root.exists():
        raise ValueError(f"Target repository path does not exist: {project_root}")

    if not project_root.is_dir():
        raise ValueError(f"Target repository path is not a directory: {project_root}")

    logger.info(f"[GraphCode MCP] Project root resolved to: {project_root}")
    return project_root


def create_server() -> tuple[Server, MemgraphIngestor]:
    """Create and configure the MCP server.

    Returns:
        Tuple of (configured MCP server instance, MemgraphIngestor instance)
    """
    setup_logging()

    # Get project root
    try:
        project_root = get_project_root()
        logger.info(f"[GraphCode MCP] Using project root: {project_root}")
    except ValueError as e:
        logger.error(f"[GraphCode MCP] Configuration error: {e}")
        raise

    # Initialize services
    logger.info("[GraphCode MCP] Initializing services...")

    ingestor = MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=settings.MEMGRAPH_BATCH_SIZE,
        username=settings.MEMGRAPH_USERNAME,
        password=settings.MEMGRAPH_PASSWORD,
    )

    # CypherGenerator gets config from settings automatically
    cypher_generator = CypherGenerator()

    # Create tools registry
    tools = create_mcp_tools_registry(
        project_root=str(project_root),
        ingestor=ingestor,
        cypher_gen=cypher_generator,
    )

    logger.info("[GraphCode MCP] Services initialized successfully")

    # Create MCP server
    server = Server("graph-code")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available MCP tools.

        Tool schemas are dynamically generated from the MCPToolsRegistry,
        ensuring consistency between tool definitions and handlers.
        """
        schemas = tools.get_tool_schemas()
        return [
            Tool(
                name=schema["name"],
                description=schema["description"],
                inputSchema=schema["inputSchema"],
            )
            for schema in schemas
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool execution requests.

        Tool handlers are dynamically resolved from the MCPToolsRegistry,
        ensuring consistency with tool definitions.

        Returns:
            list[TextContent]: MCP-compliant response containing the tool result as JSON text
        """
        logger.info(
            f"[GraphCode MCP] Calling tool: {name} with arguments: {list(arguments.keys())}"
        )

        try:
            # Resolve handler from registry
            handler_info = tools.get_tool_handler(name)
            if not handler_info:
                error_msg = f"Unknown tool: {name}"
                logger.error(f"[GraphCode MCP] {error_msg}")
                return [TextContent(type="text", text=f"Error: {error_msg}")]

            handler, returns_json = handler_info
            logger.debug(
                f"[GraphCode MCP] Handler resolved for {name}, returns_json={returns_json}"
            )

            # Call handler with unpacked arguments
            result = await handler(**arguments)
            logger.debug(f"[GraphCode MCP] Handler executed successfully for {name}")

            # Format result based on output type
            if returns_json:
                try:
                    result_text = json.dumps(result, indent=2)
                    logger.info(
                        f"[GraphCode MCP] JSON serialized result for {name}: {len(result_text)} bytes"
                    )
                except (TypeError, ValueError) as json_err:
                    logger.error(
                        f"[GraphCode MCP] Failed to JSON serialize result for tool '{name}': {json_err}",
                        exc_info=True,
                    )
                    # Fallback: convert to string representation with error info
                    fallback_result = {
                        "output": str(result),
                        "serialization_error": str(json_err),
                        "error_type": "json_serialization_failed",
                    }
                    result_text = json.dumps(fallback_result, indent=2)
                    logger.info(
                        f"[GraphCode MCP] Using fallback serialization for {name}"
                    )
            else:
                result_text = str(result)
                logger.debug(
                    f"[GraphCode MCP] String formatted result for {name}: {len(result_text)} bytes"
                )

            # Return MCP-compliant response
            response = [TextContent(type="text", text=result_text)]
            logger.debug(f"[GraphCode MCP] Returning response for {name}")
            return response

        except Exception as e:
            error_msg = f"Error executing tool '{name}': {str(e)}"
            logger.error(f"[GraphCode MCP] {error_msg}", exc_info=True)
            # Return error in consistent format
            error_response = {
                "error": error_msg,
                "error_type": type(e).__name__,
                "tool": name,
            }
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

    return server, ingestor


async def main() -> None:
    """Main entry point for the MCP server."""
    logger.info("[GraphCode MCP] Starting MCP server...")

    server, ingestor = create_server()
    logger.info("[GraphCode MCP] Server created, starting stdio transport...")

    # Use context manager to ensure proper cleanup of Memgraph connection
    with ingestor:
        logger.info(
            f"[GraphCode MCP] Connected to Memgraph at {settings.MEMGRAPH_HOST}:{settings.MEMGRAPH_PORT}"
        )
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream, server.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"[GraphCode MCP] Fatal error: {e}")
            raise
        finally:
            logger.info("[GraphCode MCP] Shutting down server...")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

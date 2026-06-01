from app.main import app


def test_mcp_manager_is_attached_to_app_state():
    assert hasattr(app.state, "mcp_client_manager")
    assert hasattr(app.state, "mcp_tool_registry")

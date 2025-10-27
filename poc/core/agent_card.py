"""A2A Agent Card builder for service metadata.

Agent Cards are the standard way A2A agents describe their capabilities.
"""

from typing import Any, Dict, List, Optional


class A2AAgentCardBuilder:
    """Builder for creating A2A Agent Cards."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        version: str,
    ):
        """Initialize agent card builder.

        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            version: Agent version
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.version = version
        self.description = ""
        self.tools: List[Dict[str, Any]] = []
        self.contact_email = ""
        self.documentation_url = ""
        self.metadata: Dict[str, Any] = {}

    def set_description(self, description: str) -> "A2AAgentCardBuilder":
        """Set agent description.

        Args:
            description: Agent description

        Returns:
            Self for chaining
        """
        self.description = description
        return self

    def add_tool(
        self,
        tool_name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
    ) -> "A2AAgentCardBuilder":
        """Add a tool to the agent card.

        Args:
            tool_name: Name of the tool
            description: Tool description
            parameters: Tool parameters schema
            examples: Usage examples

        Returns:
            Self for chaining
        """
        tool = {
            "name": tool_name,
            "description": description,
            "parameters": parameters or {},
            "examples": examples or [],
        }
        self.tools.append(tool)
        return self

    def set_contact(self, email: str) -> "A2AAgentCardBuilder":
        """Set contact email.

        Args:
            email: Contact email address

        Returns:
            Self for chaining
        """
        self.contact_email = email
        return self

    def set_documentation(self, url: str) -> "A2AAgentCardBuilder":
        """Set documentation URL.

        Args:
            url: URL to documentation

        Returns:
            Self for chaining
        """
        self.documentation_url = url
        return self

    def add_metadata(self, key: str, value: Any) -> "A2AAgentCardBuilder":
        """Add custom metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self.metadata[key] = value
        return self

    def build(self) -> Dict[str, Any]:
        """Build the A2A Agent Card.

        Returns:
            Agent Card as dictionary
        """
        return {
            "id": self.agent_id,
            "name": self.agent_name,
            "version": self.version,
            "description": self.description,
            "tools": self.tools,
            "contact": self.contact_email,
            "documentation": self.documentation_url,
            "metadata": self.metadata,
        }


def create_agent_card(
    agent_id: str,
    agent_name: str,
    version: str,
    description: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    contact_email: str = "",
    documentation_url: str = "",
) -> Dict[str, Any]:
    """Convenience function to create A2A Agent Card.

    Args:
        agent_id: Unique agent identifier
        agent_name: Human-readable agent name
        version: Agent version
        description: Agent description
        tools: List of tools
        contact_email: Contact email
        documentation_url: Documentation URL

    Returns:
        Agent Card as dictionary
    """
    builder = A2AAgentCardBuilder(agent_id, agent_name, version)
    builder.set_description(description)
    builder.set_contact(contact_email)
    builder.set_documentation(documentation_url)

    for tool in tools or []:
        builder.add_tool(
            tool["name"],
            tool.get("description", ""),
            tool.get("parameters"),
            tool.get("examples"),
        )

    return builder.build()

"""Fetch and convert Notion page content to plain text."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_page_id(url_or_id: str) -> str:
    """Parse Notion URL or validate raw page ID."""
    url_or_id = url_or_id.strip()
    # Match UUID format (with or without dashes)
    if re.match(r'^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$', url_or_id, re.I):
        return url_or_id.replace("-", "")
    # Extract from Notion URL
    match = re.search(r'([0-9a-f]{32})(?:\?|$)', url_or_id, re.I)
    if match:
        return match.group(1)
    # Try the last segment after the last dash
    match = re.search(r'-([0-9a-f]{32})(?:\?|$)', url_or_id, re.I)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract Notion page ID from: {url_or_id}")


def fetch_notion_page(page_id: str, notion_api_key: str) -> str:
    """Fetch all blocks from a Notion page, convert to plain text."""
    from notion_client import Client

    client = Client(auth=notion_api_key)
    blocks = _get_all_blocks(client, page_id)
    text = _blocks_to_text(blocks, client)
    return text.strip()


def _get_all_blocks(client: Any, block_id: str) -> list[dict]:
    """Fetch all child blocks, handling pagination."""
    blocks = []
    cursor = None
    while True:
        kwargs: dict[str, Any] = {"block_id": block_id}
        if cursor:
            kwargs["start_cursor"] = cursor
        response = client.blocks.children.list(**kwargs)
        blocks.extend(response.get("results", []))
        if not response.get("has_more"):
            break
        cursor = response.get("next_cursor")
    return blocks


def _blocks_to_text(blocks: list[dict], client: Any, depth: int = 0) -> str:
    """Convert Notion blocks to plain text."""
    lines: list[str] = []

    for block in blocks:
        block_type = block.get("type", "")
        block_data = block.get(block_type, {})

        text = ""
        if block_type in ("paragraph", "quote", "callout"):
            text = _rich_text_to_str(block_data.get("rich_text", []))
        elif block_type in ("heading_1", "heading_2", "heading_3"):
            level = block_type[-1]
            text = "#" * int(level) + " " + _rich_text_to_str(block_data.get("rich_text", []))
        elif block_type in ("bulleted_list_item", "numbered_list_item"):
            text = "- " + _rich_text_to_str(block_data.get("rich_text", []))
        elif block_type == "to_do":
            checked = "x" if block_data.get("checked") else " "
            text = f"[{checked}] " + _rich_text_to_str(block_data.get("rich_text", []))
        elif block_type == "code":
            lang = block_data.get("language", "")
            code = _rich_text_to_str(block_data.get("rich_text", []))
            text = f"```{lang}\n{code}\n```"
        elif block_type == "divider":
            text = "---"
        elif block_type == "toggle":
            text = _rich_text_to_str(block_data.get("rich_text", []))

        if text:
            lines.append(text)

        # Recurse into children if present
        if block.get("has_children") and depth < 3:
            child_blocks = _get_all_blocks(client, block["id"])
            child_text = _blocks_to_text(child_blocks, client, depth + 1)
            if child_text:
                lines.append(child_text)

    return "\n".join(lines)


def _rich_text_to_str(rich_text: list[dict]) -> str:
    """Convert Notion rich text array to plain string."""
    return "".join(item.get("plain_text", "") for item in rich_text)

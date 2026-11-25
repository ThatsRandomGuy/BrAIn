from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import re

from brain_roundtrip import BrainPipeline
from brain_search import BrainSearch


@dataclass
class CommandResult:
    ok: bool
    action: str
    message: str
    data: Dict[str, object]


class BrainCommands:
    """
    Tiny command layer for BrAIn maintenance actions.

    Supported commands:
    - show pending merges
    - show approved merges
    - show rejected merges
    - approve merge <left> <right>
    - reject merge <left> <right>
    - apply approved merges
    - show recent history
    - clear history
    - search <query>
    - show recent artifacts
    """

    def __init__(self, pipeline: Optional[BrainPipeline] = None):
        self.pipeline = pipeline or BrainPipeline()
        self.searcher = BrainSearch(
            history_path=self.pipeline.history.history_path,
            output_dir="brain_output",
        )

    def run(self, command: str) -> CommandResult:
        raw = command.strip()
        normalized = raw.lower().strip()

        if normalized == "show pending merges":
            pending = self.pipeline.queue.get_pending()
            self.pipeline.history.log(
                event_type="command",
                summary="Viewed pending merges",
                payload={"count": len(pending)},
            )
            return CommandResult(
                ok=True,
                action="show_pending_merges",
                message=f"Found {len(pending)} pending merge(s).",
                data={"pending": pending},
            )

        if normalized == "show approved merges":
            approved = self.pipeline.queue.get_approved()
            self.pipeline.history.log(
                event_type="command",
                summary="Viewed approved merges",
                payload={"count": len(approved)},
            )
            return CommandResult(
                ok=True,
                action="show_approved_merges",
                message=f"Found {len(approved)} approved merge(s).",
                data={"approved": approved},
            )

        if normalized == "show rejected merges":
            rejected = self.pipeline.queue.get_rejected()
            self.pipeline.history.log(
                event_type="command",
                summary="Viewed rejected merges",
                payload={"count": len(rejected)},
            )
            return CommandResult(
                ok=True,
                action="show_rejected_merges",
                message=f"Found {len(rejected)} rejected merge(s).",
                data={"rejected": rejected},
            )

        if normalized == "show recent history":
            recent = self.pipeline.history.recent(limit=20)
            return CommandResult(
                ok=True,
                action="show_recent_history",
                message=f"Found {len(recent)} recent history event(s).",
                data={"recent_history": recent},
            )

        if normalized == "clear history":
            cleared = self.pipeline.history.clear()
            return CommandResult(
                ok=True,
                action="clear_history",
                message=f"Cleared {cleared['cleared_count']} history event(s).",
                data=cleared,
            )

        if normalized == "show recent artifacts":
            artifacts = self.searcher.recent_artifacts(limit=20)
            self.pipeline.history.log(
                event_type="command",
                summary="Viewed recent artifacts",
                payload={"count": len(artifacts)},
            )
            return CommandResult(
                ok=True,
                action="show_recent_artifacts",
                message=f"Found {len(artifacts)} recent artifact(s).",
                data={"recent_artifacts": artifacts},
            )

        if normalized == "apply approved merges":
            applied = self.pipeline.apply_approved_merges(prefer="right")
            return CommandResult(
                ok=True,
                action="apply_approved_merges",
                message=f"Applied {len(applied['applied_aliases'])} approved merge(s).",
                data=applied,
            )

        approve_match = re.fullmatch(r"approve merge\s+(\S+)\s+(\S+)", raw, flags=re.IGNORECASE)
        if approve_match:
            left = approve_match.group(1).strip()
            right = approve_match.group(2).strip()
            item = self.pipeline.queue.approve(left, right)
            if item is None:
                return CommandResult(
                    ok=False,
                    action="approve_merge",
                    message=f"No pending merge found for {left} and {right}.",
                    data={},
                )

            history_event = self.pipeline.history.log(
                event_type="approve_merge",
                summary=f"Approved merge {left} ↔ {right}",
                payload={"left": left, "right": right},
            )

            return CommandResult(
                ok=True,
                action="approve_merge",
                message=f"Approved merge {left} ↔ {right}.",
                data={"approved_item": item, "history_event": history_event},
            )

        reject_match = re.fullmatch(r"reject merge\s+(\S+)\s+(\S+)", raw, flags=re.IGNORECASE)
        if reject_match:
            left = reject_match.group(1).strip()
            right = reject_match.group(2).strip()
            item = self.pipeline.queue.reject(left, right)
            if item is None:
                return CommandResult(
                    ok=False,
                    action="reject_merge",
                    message=f"No pending merge found for {left} and {right}.",
                    data={},
                )

            history_event = self.pipeline.history.log(
                event_type="reject_merge",
                summary=f"Rejected merge {left} ↔ {right}",
                payload={"left": left, "right": right},
            )

            return CommandResult(
                ok=True,
                action="reject_merge",
                message=f"Rejected merge {left} ↔ {right}.",
                data={"rejected_item": item, "history_event": history_event},
            )

        search_match = re.fullmatch(r"search\s+(.+)", raw, flags=re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip()
            hits = self.searcher.search(query, limit=15)

            self.pipeline.history.log(
                event_type="search",
                summary=f"Searched BrAIn for: {query}",
                payload={"query": query, "hits": len(hits)},
            )

            return CommandResult(
                ok=True,
                action="search",
                message=f"Found {len(hits)} search hit(s) for '{query}'.",
                data={"query": query, "hits": hits},
            )

        return CommandResult(
            ok=False,
            action="unknown_command",
            message="Unknown command.",
            data={
                "supported_commands": [
                    "show pending merges",
                    "show approved merges",
                    "show rejected merges",
                    "approve merge <left> <right>",
                    "reject merge <left> <right>",
                    "apply approved merges",
                    "show recent history",
                    "clear history",
                    "search <query>",
                    "show recent artifacts",
                ]
            },
        )


if __name__ == "__main__":
    commands = BrainCommands()

    demo_commands = [
        "show pending merges",
        "search strawberries",
        "show recent history",
    ]

    for cmd in demo_commands:
        result = commands.run(cmd)
        print("=" * 72)
        print("COMMAND")
        print(cmd)
        print("\nRESULT")
        print(result.message)
        print(result.data)
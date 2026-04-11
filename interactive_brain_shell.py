from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from brain_hybrid_prototype import BrainHybridPrototype, HybridTurnResult, StoragePolicy
from terminal_brain_formatting import (
    heading,
    pretty_json,
    render_turn_sections,
    summarize_memory_entries,
    summarize_paths,
)


DEFAULT_STORAGE_MODE = "store_all_prompts"


@dataclass
class ShellState:
    debug: bool = False
    raw: bool = False
    store_enabled: bool = True
    storage_mode: str = DEFAULT_STORAGE_MODE
    last_result: Optional[HybridTurnResult] = None
    last_goal_weights: dict[str, float] | None = None


def _help_text() -> str:
    return "\n".join(
        [
            heading("COMMANDS"),
            "/help                Show this help",
            "/quit                Exit the shell",
            "/debug on|off        Toggle detailed output",
            "/raw on|off          Toggle raw JSON output",
            "/memory              Show recent stored memory entries",
            "/memory full         Show detailed recent stored memory entries",
            "/last                Show the full trace from the last prompt",
            "/paths               Show candidate paths from the last prompt",
            "/pending             Show the current pending clarification, if any",
            "/goals               Show active goals and last runtime weights",
            "/store on|off        Toggle automatic storage",
            "/store mode <mode>   Set storage mode",
            "                     Modes: store_all_prompts, store_only_successful_reasoning,",
            "                            store_final_results_only, store_disabled",
            "",
            "Plain text input is routed before BrAIn/LLM execution.",
        ]
    )


def _set_storage_policy(prototype: BrainHybridPrototype, mode: str) -> None:
    prototype.storage_policy = StoragePolicy.for_mode(mode)


def _print_memory(prototype: BrainHybridPrototype, full: bool) -> None:
    print(heading("MEMORY"))
    print(summarize_memory_entries(prototype.list_memory_entries(), full=full))


def _print_goals(prototype: BrainHybridPrototype, shell_state: ShellState) -> None:
    print(heading("GOALS"))
    payload = {
        "storage_mode": shell_state.storage_mode,
        "store_enabled": shell_state.store_enabled,
        "last_goal_weights": shell_state.last_goal_weights or {},
        "engine_goals": prototype.active_goals_summary(),
        "tracked_goals_v6": prototype.tracked_goals_summary(),
        "tracked_strategies_v7": prototype.tracked_strategies_summary(),
    }
    print(pretty_json(payload))


def _print_pending(prototype: BrainHybridPrototype) -> None:
    pending = prototype.get_pending_clarification()
    print(heading("PENDING CLARIFICATION"))
    if pending is None:
        print("No clarification is pending.")
        return
    print(pretty_json(pending))


def _print_last(shell_state: ShellState) -> None:
    if shell_state.last_result is None:
        print("No prompt has been processed yet.")
        return
    print(heading("LAST TRACE"))
    print(pretty_json(shell_state.last_result.trace.to_dict()))


def _print_paths(shell_state: ShellState, raw: bool) -> None:
    if shell_state.last_result is None:
        print("No prompt has been processed yet.")
        return
    print(heading("CANDIDATE PATHS"))
    print(summarize_paths(shell_state.last_result.trace.goal_scored_paths, full=raw))


def _process_command(command: str, prototype: BrainHybridPrototype, shell_state: ShellState) -> bool:
    if command in {"/quit", "/exit", ":quit", ":exit"}:
        return False
    if command in {"/help", ":help"}:
        print(_help_text())
        return True
    if command.startswith("/debug ") or command.startswith(":debug "):
        shell_state.debug = command.split()[-1].lower() == "on"
        print(f"Debug mode {'enabled' if shell_state.debug else 'disabled'}.")
        return True
    if command.startswith("/raw ") or command.startswith(":raw "):
        shell_state.raw = command.split()[-1].lower() == "on"
        print(f"Raw output {'enabled' if shell_state.raw else 'disabled'}.")
        return True
    if command in {"/memory", ":memory"}:
        _print_memory(prototype, full=False)
        return True
    if command in {"/memory full", ":memory full"}:
        _print_memory(prototype, full=True)
        return True
    if command in {"/last", ":last"}:
        _print_last(shell_state)
        return True
    if command in {"/paths", ":paths"}:
        _print_paths(shell_state, raw=shell_state.raw or shell_state.debug)
        return True
    if command in {"/pending", ":pending"}:
        _print_pending(prototype)
        return True
    if command in {"/goals", ":goals"}:
        _print_goals(prototype, shell_state)
        return True
    if command.startswith("/store ") or command.startswith(":store "):
        parts = command.split()
        if len(parts) >= 2 and parts[1] in {"on", "off"}:
            shell_state.store_enabled = parts[1] == "on"
            mode = shell_state.storage_mode if shell_state.store_enabled else "store_disabled"
            _set_storage_policy(prototype, mode)
            print(f"Automatic storage {'enabled' if shell_state.store_enabled else 'disabled'}.")
            return True
        if len(parts) >= 3 and parts[1] == "mode":
            requested_mode = parts[2]
            shell_state.storage_mode = requested_mode
            shell_state.store_enabled = requested_mode != "store_disabled"
            _set_storage_policy(prototype, requested_mode if shell_state.store_enabled else "store_disabled")
            print(f"Storage mode set to {requested_mode}.")
            return True
        print("Usage: /store on|off OR /store mode <mode>")
        return True
    print("Unknown command. Use /help.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="interactive_brain_shell",
        description="Terminal BrAIn + LLM interactive shell.",
    )
    parser.add_argument(
        "--state-dir",
        default=str(Path("brain_output") / "interactive_prototype.state"),
        help="State directory for prototype memory and model state.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Start with detailed terminal output enabled.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Start with raw JSON output enabled.",
    )
    parser.add_argument(
        "--storage-mode",
        default=DEFAULT_STORAGE_MODE,
        choices=(
            "store_all_prompts",
            "store_only_successful_reasoning",
            "store_final_results_only",
            "store_disabled",
        ),
        help="Storage policy mode for prompt/result/trace writes.",
    )
    args = parser.parse_args()

    prototype = BrainHybridPrototype(
        state_dir=args.state_dir,
        storage_policy=StoragePolicy.for_mode(args.storage_mode),
    )
    shell_state = ShellState(
        debug=args.debug,
        raw=args.raw,
        store_enabled=args.storage_mode != "store_disabled",
        storage_mode=args.storage_mode,
    )

    print(heading("BrAIn Terminal Shell"))
    print("Type a prompt to run the routed BrAIn assistant.")
    print("Use /help for commands.")

    while True:
        try:
            prompt = input("\nbrain> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.startswith("/") or prompt.startswith(":"):
            if not _process_command(prompt, prototype, shell_state):
                break
            continue
        if not shell_state.store_enabled:
            prototype.storage_policy = StoragePolicy.for_mode("store_disabled")
        else:
            prototype.storage_policy = StoragePolicy.for_mode(shell_state.storage_mode)
        result = prototype.process_user_turn(prompt, inspect=shell_state.debug)
        shell_state.last_result = result
        shell_state.last_goal_weights = result.trace.goal_weights
        print(render_turn_sections(result.to_dict(), debug=shell_state.debug, raw=shell_state.raw))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

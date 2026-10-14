"""Explicit command-line entry point for the independent skill pool."""

import argparse
import json
import sys
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(
        description="Independent self-evolving skill pool (no server)."
    )
    result.add_argument(
        "--root", help="Independent storage directory; defaults to SKILL_POOL_ROOT."
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in ["init", "status", "history", "recover", "demo", "maintain-due"]:
        commands.add_parser(name)
    for name in ["import", "generate", "observe"]:
        commands.add_parser(name).add_argument("file", help="UTF-8 SKILL.md or JSON draft/trace")
    commands.add_parser("integrate").add_argument("candidate_id")
    usage = commands.add_parser("record-usage")
    usage.add_argument("reference_id")
    usage.add_argument("task_id")
    usage.add_argument("--independent", action="store_true")
    maintenance = commands.add_parser("maintain")
    maintenance.add_argument("cluster_id")
    maintenance.add_argument(
        "--action", choices=["PROMOTE", "SPLIT", "MERGE", "REFACTOR", "ARCHIVE", "KEEP"]
    )
    discovery = commands.add_parser("discover")
    discovery.add_argument("query")
    discovery.add_argument("--limit", type=int)
    loading = commands.add_parser("load")
    loading.add_argument("id")
    loading.add_argument("--reference", action="store_true")
    commands.add_parser("rollback").add_argument("operation_id")
    commands.add_parser("smoke").add_argument(
        "--llm", action="store_true", help="Also call the configured real Agent model."
    )
    return result


def _read_input(args):
    if args.command not in {"import", "generate", "observe"}:
        return None
    content = Path(args.file).read_text(encoding="utf-8-sig")
    if args.command == "import" and Path(args.file).suffix.lower() != ".json":
        return content
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at line {exc.lineno}, column {exc.colno}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _execute(manager, args, value):
    command = args.command
    if command in {"init", "status"}:
        return manager.status()
    if command == "import":
        return manager.import_candidate(
            value, source={"type": "manual", "path": str(Path(args.file).resolve())}
        )
    if command == "generate":
        return manager.generate_candidate(value)
    if command == "observe":
        return manager.observe(value)
    if command == "integrate":
        return manager.integrate(args.candidate_id)
    if command == "record-usage":
        return manager.record_usage(args.reference_id, args.task_id, independent=args.independent)
    if command == "maintain":
        return manager.maintain(args.cluster_id, action=args.action)
    if command == "maintain-due":
        return manager.maintain_due()
    if command == "discover":
        return manager.discover(args.query, limit=args.limit)
    if command == "load":
        return manager.load_reference(args.id) if args.reference else manager.load_master(args.id)
    if command == "history":
        return manager.history()
    if command == "rollback":
        return manager.rollback(args.operation_id)
    if command == "recover":
        return manager.recover()
    raise ValueError("Unsupported command")


def main(argv=None, manager_factory=None):
    args = parser().parse_args(argv)
    manager = None
    try:
        value = _read_input(args)
        if args.command == "demo":
            from .demo import run_demo

            result = run_demo(args.root)
        elif args.command == "smoke":
            from .demo import run_smoke

            result = run_smoke(with_llm=args.llm)
        else:
            if manager_factory is None:
                from . import SkillManager

                manager_factory = SkillManager
            manager = manager_factory(root=args.root)
            result = _execute(manager, args, value)
        if args.command == "load":
            print(result, end="" if result.endswith("\n") else "\n")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        from .demo import SmokeFailure

        # Provider exceptions may include credentials or request bodies; expose only type.
        message = (
            str(exc)
            if isinstance(exc, (ValueError, KeyError, FileNotFoundError, SmokeFailure))
            else type(exc).__name__
        )
        print(f"Skill pool error: {message}", file=sys.stderr)
        return 1
    finally:
        if manager is not None:
            manager.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())

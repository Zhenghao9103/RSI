import argparse
import json
from dataclasses import asdict

from . import Memory


def main() -> int:
    parser = argparse.ArgumentParser(description="Local procedural memory")
    parser.add_argument("--db", default="data/memory.db")
    commands = parser.add_subparsers(dest="command", required=True)
    learn = commands.add_parser("learn", help="Learn a strategy from feedback")
    learn.add_argument("feedback")
    learn.add_argument("--previous-answer", default="")
    ask = commands.add_parser("ask", help="Answer with retrieved strategies")
    ask.add_argument("question")
    commands.add_parser("skills", help="List current skill versions")
    commands.add_parser("retry", help="Retry pending reflections and index writes")
    commands.add_parser("rebuild", help="Rebuild the index from current skills")
    args = parser.parse_args()

    code = 0
    with Memory(args.db) as memory:
        if args.command == "learn":
            status = memory.learn(args.feedback, args.previous_answer)
            result = {"status": status}
            code = int(status in {"disabled", "pending", "rejected"})
        elif args.command == "ask":
            response = memory.ask(args.question)
            result = asdict(response)
            code = int(response.failure == "generation_failed")
        elif args.command == "skills":
            result = memory.skills()
        elif args.command == "retry":
            memory.retry()
            result = {"status": "retry_attempted"}
        else:
            memory.rebuild()
            result = {"status": "rebuilt"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

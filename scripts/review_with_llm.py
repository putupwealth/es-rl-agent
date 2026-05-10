"""
Milestone 4 — LLM review generator.

Reads one evaluation report directory (or latest-run pointer), loads
llm_input_packet.json, sends it to an LLM with a structured review prompt,
and writes llm_review.md into the same report directory.

Usage:
    python scripts/review_with_llm.py reports/latest_run.txt
    python scripts/review_with_llm.py reports/run_123
    python scripts/review_with_llm.py reports/latest_run.txt --stdout-only

Environment:
    OPENAI_API_KEY must be set, typically via .env
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_PROMPT_FILE = "prompts/eval_review_prompt.txt"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PACKET_FILE = "llm_input_packet.json"
DEFAULT_OUTPUT_FILE = "llm_review.md"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate llm_review.md from llm_input_packet.json."
    )
    parser.add_argument(
        "run_target",
        help="Report directory path or pointer file such as reports/latest_run.txt",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help="Path to the review prompt template.",
    )
    parser.add_argument(
        "--packet-file",
        default=DEFAULT_PACKET_FILE,
        help="Packet filename inside the report directory.",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output markdown filename inside the report directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI model name to use.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print review to stdout instead of writing a file.",
    )
    return parser.parse_args()


def resolve_report_dir(run_target: str) -> Path:
    """
    Resolve a report directory argument.

    Accepts either:
    - a direct report directory path, or
    - a .txt pointer file whose contents are the real report directory path
      (for example reports/latest_run.txt).
    """
    path = Path(run_target)

    if path.is_file() and path.suffix.lower() == ".txt":
        resolved = path.read_text(encoding="utf-8").strip()
        if not resolved:
            raise ValueError(f"Run pointer is empty: {path}")
        return Path(resolved)

    return path


def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def load_packet(packet_path: Path) -> dict:
    if not packet_path.exists():
        raise FileNotFoundError(f"Packet file not found: {packet_path}")

    with open(packet_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_messages(prompt_template: str, packet: dict):
    packet_json = json.dumps(packet, indent=2, ensure_ascii=False)
    user_content = f"{prompt_template}\n{packet_json}"
    return [
        {
            "role": "system",
            "content": (
                "You are a careful RL experiment reviewer. "
                "You must use only supplied evidence and produce concise markdown."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def generate_review(client: OpenAI, model: str, messages) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("LLM returned empty review output.")

    return content.strip()


def main():
    args = parse_args()

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    try:
        report_dir = resolve_report_dir(args.run_target)
        if not report_dir.exists() or not report_dir.is_dir():
            raise FileNotFoundError(f"Report directory not found: {report_dir}")

        prompt_path = Path(args.prompt_file)
        packet_path = report_dir / args.packet_file
        output_path = report_dir / args.output_file

        prompt_template = load_text_file(prompt_path)
        packet = load_packet(packet_path)

        messages = build_messages(prompt_template, packet)
        review_markdown = generate_review(client, args.model, messages)

        if args.stdout_only:
            print(review_markdown)
            return

        output_path.write_text(review_markdown + "\n", encoding="utf-8")

        print(f"Saved review: {output_path}")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
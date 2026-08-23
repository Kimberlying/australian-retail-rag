from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import docs_dir, index_path
from .ingest import build_index
from .pipeline import RAGPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Australian Retail RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the local retrieval index")
    ingest_parser.add_argument("--docs", default=str(docs_dir()))
    ingest_parser.add_argument("--index", default=str(index_path()))

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question")
    query_parser.add_argument("--top-k", type=int, default=4)
    query_parser.add_argument("--index", default=str(index_path()))

    subparsers.add_parser("serve", help="Start the optional FastAPI server")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "ingest":
        retriever = build_index(Path(args.docs), Path(args.index))
        print(f"Indexed {len(retriever.chunks)} chunks -> {args.index}")
        return 0

    if args.command == "query":
        index = Path(args.index)
        if not index.exists():
            print(f"Index not found: {index}. Run `retail-rag ingest` first.", file=sys.stderr)
            return 2
        result = RAGPipeline.from_index(index).ask(args.question, top_k=args.top_k)
        print(result.answer)
        print("\nCitations:")
        print(json.dumps(result.citations, ensure_ascii=False, indent=2))
        return 0

    if args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print("Install API dependencies first: python -m pip install -e '.[api]'", file=sys.stderr)
            return 2
        uvicorn.run("retail_rag.api:app", host="127.0.0.1", port=8000, reload=False)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import REPO_ROOT, get_settings
from .ingest import build_index, load_chunks
from .logging_config import configure_logging
from .pipeline import RAGPipeline
from .retrieval import TfidfRetriever

DEFAULT_GOLDEN_SET = REPO_ROOT / "evals" / "golden_set.jsonl"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"


def _threshold(value: str) -> tuple[str, float]:
    name, sep, minimum = value.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("expected METRIC=VALUE, e.g. recall=0.8")
    try:
        return name.strip(), float(minimum)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {minimum!r}") from exc


def _parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Australian Retail RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the local retrieval index")
    ingest_parser.add_argument("--docs", type=Path, default=settings.docs_dir)
    ingest_parser.add_argument("--index", type=Path, default=settings.index_path)

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question")
    query_parser.add_argument("--top-k", type=int, default=settings.top_k)
    query_parser.add_argument("--index", type=Path, default=settings.index_path)
    query_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    eval_parser = subparsers.add_parser("eval", help="Run the golden-set evaluation")
    eval_parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_SET)
    eval_parser.add_argument("--docs", type=Path, default=settings.docs_dir)
    eval_parser.add_argument("--k", type=int, default=settings.top_k)
    eval_parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    eval_parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    eval_parser.add_argument("--min-score", type=float, default=settings.min_score)
    eval_parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    eval_parser.add_argument("--stem", default="eval", help="Report file name stem")
    eval_parser.add_argument(
        "--judge", action="store_true", help="Grade faithfulness with an LLM judge (costs tokens)"
    )
    eval_parser.add_argument(
        "--fail-under",
        type=_threshold,
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Exit non-zero if a summary metric is below VALUE (repeatable)",
    )
    return parser


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import (  # noqa: PLC0415 - keep CLI start-up light
        check_thresholds,
        load_golden_set,
        render_markdown,
        run_evaluation,
        write_reports,
    )

    # Per-query INFO logs would drown the report; failures are still logged.
    logging.getLogger("retail_rag.pipeline").setLevel(logging.WARNING)
    settings = get_settings().model_copy(update={"min_score": args.min_score})
    examples = load_golden_set(args.golden, docs_dir=args.docs)
    # Build a fresh in-memory index so every run is reproducible from source documents.
    chunks = load_chunks(args.docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    pipeline = RAGPipeline(TfidfRetriever(chunks), settings)

    judge = None
    if args.judge:
        from .evaluation.judge import FaithfulnessJudge  # noqa: PLC0415

        judge = FaithfulnessJudge(settings)

    report = run_evaluation(
        pipeline,
        examples,
        k=args.k,
        judge=judge,
        config={
            "retriever": "tfidf",
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "min_score": args.min_score,
            "n_chunks": len(chunks),
            "model": settings.anthropic_model if settings.llm_enabled else None,
        },
    )
    json_path, md_path = write_reports(report, args.output_dir, stem=args.stem)
    print(render_markdown(report))
    print(f"Reports written to {json_path} and {md_path}")

    failures = check_thresholds(report.summary, dict(args.fail_under))
    if failures:
        print("\nQuality gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    if args.fail_under:
        print("Quality gate passed.")
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 - one exit code per command
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.command == "ingest":
        retriever = build_index(
            args.docs, args.index, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )
        print(f"Indexed {len(retriever.chunks)} chunks -> {args.index}")
        return 0

    if args.command == "query":
        if not args.index.exists():
            print(f"Index not found: {args.index}. Run `retail-rag ingest` first.", file=sys.stderr)
            return 2
        result = RAGPipeline.from_index(args.index, settings).ask(args.question, top_k=args.top_k)
        if args.json:
            payload = {
                "question": result.question,
                "answer": result.answer,
                "citations": result.citations,
                "generated_by": result.generated_by,
                "refused": result.refused,
                "latency_ms": result.latency_ms,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(result.answer)
            print("\nCitations:")
            print(json.dumps(result.citations, ensure_ascii=False, indent=2))
        return 0

    if args.command == "serve":
        try:
            import uvicorn  # noqa: PLC0415 - optional dependency
        except ImportError:
            print(
                "Install API dependencies first: python -m pip install -e '.[api]'", file=sys.stderr
            )
            return 2
        uvicorn.run("retail_rag.api:app", host=args.host, port=args.port, reload=False)
        return 0

    if args.command == "eval":
        return _cmd_eval(args)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import cast

from .config import REPO_ROOT, Settings, get_settings
from .ingest import build_index, load_chunks
from .logging_config import configure_logging
from .pipeline import RAGPipeline, answer_payload
from .retrieval import create_retriever
from .sql import SQLTool, build_database

DEFAULT_GOLDEN_SET = REPO_ROOT / "evals" / "golden_set.jsonl"
ROUTER_HOLDOUT = REPO_ROOT / "evals" / "router_holdout.jsonl"
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
    query_parser.add_argument(
        "--stream", action="store_true", help="Print the answer as it is generated"
    )

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    eval_parser = subparsers.add_parser("eval", help="Run the golden-set evaluation")
    eval_parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_SET)
    eval_parser.add_argument("--docs", type=Path, default=settings.docs_dir)
    eval_parser.add_argument("--k", type=int, default=settings.top_k)
    eval_parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    eval_parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    eval_parser.add_argument(
        "--retriever", choices=["tfidf", "bm25", "dense", "hybrid"], default=settings.retriever
    )
    eval_parser.add_argument(
        "--reranker", choices=["none", "cross-encoder"], default=settings.reranker
    )
    eval_parser.add_argument(
        "--metadata-filters",
        action=argparse.BooleanOptionalAction,
        default=settings.metadata_filters,
        help="Restrict retrieval to the company / fiscal year a question names",
    )
    eval_parser.add_argument(
        "--router",
        action=argparse.BooleanOptionalAction,
        default=settings.router_enabled,
        help="Route questions to documents, the SQL database, or both",
    )
    eval_parser.add_argument(
        "--min-score",
        type=float,
        default=settings.min_score,
        help="Evidence-gate threshold (default: the retriever's calibrated default)",
    )
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
    eval_parser.add_argument(
        "--fail-over",
        type=_threshold,
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Exit non-zero if a summary metric is above VALUE (repeatable)",
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
    settings = get_settings().model_copy(
        update={
            "min_score": args.min_score,
            "reranker": args.reranker,
            "metadata_filters": args.metadata_filters,
            "router_enabled": args.router,
        }
    )
    examples = load_golden_set(args.golden, docs_dir=args.docs)
    # Build a fresh in-memory index (and SQL database) so every run is reproducible
    # from the source documents and the seeded data generator.
    chunks = load_chunks(args.docs, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    retriever = create_retriever(args.retriever, chunks, settings)

    judge = None
    if args.judge:
        from .evaluation.judge import FaithfulnessJudge  # noqa: PLC0415

        judge = FaithfulnessJudge(settings)

    with tempfile.TemporaryDirectory() as workdir:
        sql_tool = SQLTool(build_database(Path(workdir) / "retail.db"))
        pipeline = RAGPipeline(retriever, settings, sql_tool=sql_tool)
        report = run_evaluation(
            pipeline,
            examples,
            k=args.k,
            judge=judge,
            config={
                "retriever": retriever.name,
                "embedding_model": (
                    settings.embedding_model if args.retriever in ("dense", "hybrid") else None
                ),
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "reranker": settings.reranker_model if settings.reranker != "none" else None,
                "metadata_filters": settings.metadata_filters,
                "router": settings.router_enabled,
                "min_score": pipeline.min_score,
                "n_chunks": len(chunks),
                "model": settings.anthropic_model if settings.llm_enabled else None,
            },
        )
    if args.router and ROUTER_HOLDOUT.is_file():
        from .evaluation.routing import router_holdout  # noqa: PLC0415

        accuracy, misses = router_holdout(ROUTER_HOLDOUT)
        report.summary["route_accuracy_holdout"] = accuracy
        report.router_holdout_misses = misses
    json_path, md_path = write_reports(report, args.output_dir, stem=args.stem)
    print(render_markdown(report))
    print(f"Reports written to {json_path} and {md_path}")

    failures = check_thresholds(report.summary, dict(args.fail_under), dict(args.fail_over))
    if failures:
        print("\nQuality gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    if args.fail_under or args.fail_over:
        print("Quality gate passed.")
    return 0


def _cmd_query(args: argparse.Namespace, settings: Settings) -> int:
    pipeline = RAGPipeline.from_index(args.index, settings)
    if args.stream:
        payload: dict[str, object] = {}
        for event in pipeline.stream(args.question, top_k=args.top_k):
            if event["event"] == "token":
                print(event["text"], end="", flush=True)
            elif event["event"] == "done":
                payload = event
        print()
        _print_details(payload)
        return 0
    result = answer_payload(pipeline.ask(args.question, top_k=args.top_k))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
        _print_details(result)
    return 0


def _print_details(payload: dict[str, object]) -> None:
    print(f"\nRoute: {payload.get('route')} · generated by: {payload.get('generated_by')}")
    if payload.get("refused"):
        print(f"Refused ({payload.get('refusal_reason')}).")
    if payload.get("sql_queries"):
        print("SQL:")
        for statement in cast("list[str]", payload["sql_queries"]):
            print(f"  {statement}")
    if payload.get("citations"):
        print("Citations:")
        print(json.dumps(payload["citations"], ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 - one exit code per command
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.command == "ingest":
        retriever = build_index(args.docs, args.index, settings)
        print(f"Indexed {len(retriever.chunks)} chunks with {retriever.name} -> {args.index}")
        database = build_database(settings.sql_db_path)
        print(f"Built the synthetic operational database -> {database}")
        return 0

    if args.command == "query":
        if not args.index.exists():
            print(f"Index not found: {args.index}. Run `retail-rag ingest` first.", file=sys.stderr)
            return 2
        return _cmd_query(args, settings)

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

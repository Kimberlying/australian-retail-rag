from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from retail_rag.chunking import chunk_text
from retail_rag.ingest import build_index
from retail_rag.models import DocumentChunk
from retail_rag.pipeline import RAGPipeline
from retail_rag.retrieval import TfidfRetriever


class RAGTests(unittest.TestCase):
    def test_chunking_respects_overlap_contract(self) -> None:
        chunks = chunk_text("one two three four five six seven eight nine ten", chunk_size=20, overlap=5)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_retriever_returns_relevant_source(self) -> None:
        retriever = TfidfRetriever(
            [
                DocumentChunk(
                    chunk_id="a", source="sales.md", text="normalised ecommerce sales growth was 23.3 percent"
                ),
                DocumentChunk(
                    chunk_id="b", source="policy.md", text="unknown returns require human review"
                ),
            ]
        )
        result = retriever.search("ecommerce sales growth")
        self.assertEqual(result[0].chunk.source, "sales.md")

    def test_ingest_and_pipeline(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "index.json"
            retriever = build_index(project_root / "data" / "documents", index_path)
            result = RAGPipeline(retriever).ask("What is the normalised eCommerce sales growth?")
            self.assertTrue(index_path.exists())
            self.assertTrue(result.citations)
            self.assertIn("23.3", result.answer)


if __name__ == "__main__":
    unittest.main()

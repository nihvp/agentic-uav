"""Learning layer: RAG query / insert over a persistent vector store."""

import json

import rclpy
from rclpy.node import Node

from agentic_uav_interfaces.srv import RAGInsert, RAGQuery
from agentic_uav.learning.sop_documents import DEFAULT_KNOWLEDGE_DOCUMENTS
from agentic_uav.learning.vector_store import HashingEmbedder, SentenceTransformerEmbedder, VectorStore


class LearningNode(Node):
    def __init__(self):
        super().__init__("learning_node")
        path = self.declare_parameter("store_path", "/data/memory.jsonl").value
        embedder_name = self.declare_parameter("embedder", "sentence-transformers").value

        embedder = HashingEmbedder()
        if embedder_name == "sentence-transformers":
            try:
                embedder = SentenceTransformerEmbedder()
            except Exception as ex:
                self.get_logger().warn(f"sentence-transformers unavailable ({ex}); using hashing embedder.")

        self.store = VectorStore(path, embedder)
        if len(self.store) == 0:
            for d in DEFAULT_KNOWLEDGE_DOCUMENTS:
                self.store.insert(d["id"], d["content"], d["category"], title=d.get("title", ""),
                                  tags=d.get("tags", []))
            self.get_logger().info(f"Seeded {len(self.store)} SOP documents.")

        self.create_service(RAGQuery, "/learning/rag_query", self._on_query)
        self.create_service(RAGInsert, "/learning/rag_insert", self._on_insert)
        self.get_logger().info(f"Learning layer up ({len(self.store)} documents at {path}).")

    def _on_query(self, request, response):
        hits = self.store.query(request.query_text, request.top_k or 3, request.filter_category)
        response.success = True
        response.document_ids = [d["id"] for d, _ in hits]
        response.contents = [d["content"] for d, _ in hits]
        response.similarity_scores = [s for _, s in hits]
        return response

    def _on_insert(self, request, response):
        try:
            meta = json.loads(request.metadata_json) if request.metadata_json else {}
            self.store.insert(request.document_id, request.content, request.category or "GENERAL", metadata=meta)
            response.success, response.message = True, f"Stored {request.document_id}."
        except Exception as ex:
            response.success, response.message = False, str(ex)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LearningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Learning Layer Node.
Implements Retrieval-Augmented Generation (RAG) for UAV mission knowledge integration,
SOP lookup, and cross-mission episodic memory retention.
"""

import json
import rclpy
from rclpy.node import Node

from agentic_uav.srv import RAGQuery, RAGInsert
from agentic_uav.learning.vector_store import SemanticVectorStore
from agentic_uav.learning.default_sop_knowledge import DEFAULT_KNOWLEDGE_DOCUMENTS


class LearningNode(Node):
    def __init__(self):
        super().__init__("learning_node")

        self.vector_store = SemanticVectorStore()

        # Ingest baseline standard operating procedures and previous mission logs
        self._load_baseline_knowledge()

        # Services
        self.srv_query = self.create_service(
            RAGQuery, "/learning/rag_query", self.handle_query
        )
        self.srv_insert = self.create_service(
            RAGInsert, "/learning/rag_insert", self.handle_insert
        )

        self.get_logger().info(
            f"Learning Layer active. Indexed {len(self.vector_store.documents)} knowledge documents."
        )

    def _load_baseline_knowledge(self):
        for doc in DEFAULT_KNOWLEDGE_DOCUMENTS:
            self.vector_store.insert(
                doc_id=doc["id"],
                content=f"{doc['title']}: {doc['content']}",
                category=doc["category"],
                metadata={"title": doc["title"], "tags": doc.get("tags", [])}
            )

    def handle_query(self, request, response):
        query_text = request.query_text
        top_k = request.top_k if request.top_k > 0 else 2
        category_filter = request.filter_category if request.filter_category else None

        self.get_logger().info(f"Received RAG query: '{query_text}' (top_k={top_k}, category={category_filter})")

        try:
            matches = self.vector_store.query(
                query_text=query_text,
                top_k=top_k,
                filter_category=category_filter
            )

            response.success = True
            response.document_ids = [m[0]["id"] for m in matches]
            response.contents = [m[0]["content"] for m in matches]
            response.similarity_scores = [float(m[1]) for m in matches]
            response.error_message = ""
        except Exception as e:
            self.get_logger().error(f"RAG query execution error: {e}")
            response.success = False
            response.document_ids = []
            response.contents = []
            response.similarity_scores = []
            response.error_message = str(e)

        return response

    def handle_insert(self, request, response):
        doc_id = request.document_id
        content = request.content
        category = request.category if request.category else "MISSION_LOG"

        try:
            meta = json.loads(request.metadata_json) if request.metadata_json else {}
        except Exception:
            meta = {}

        try:
            self.vector_store.insert(doc_id=doc_id, content=content, category=category, metadata=meta)
            self.get_logger().info(f"Indexed new document [{doc_id}] in category '{category}'")
            response.success = True
            response.message = f"Successfully stored and indexed document '{doc_id}'."
        except Exception as e:
            response.success = False
            response.message = f"Failed to insert document: {str(e)}"

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()

from agentic_uav.learning.vector_store import HashingEmbedder, VectorStore


def test_persists_and_retrieves(tmp_path):
    path = str(tmp_path / "mem.jsonl")
    store = VectorStore(path)
    store.insert("a", "landing zone blocked by debris, use secondary site", "SOP")
    store.insert("b", "battery low return to launch", "SOP")

    reloaded = VectorStore(path)
    assert len(reloaded) == 2
    assert reloaded.query("debris on landing zone", top_k=1)[0][0]["id"] == "a"


def test_hashing_embedder_is_deterministic():
    assert (HashingEmbedder().encode("fire hazard") == HashingEmbedder().encode("fire hazard")).all()

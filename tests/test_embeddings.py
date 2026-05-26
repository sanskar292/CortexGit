import pytest
from cortexgit.retrieval.embeddings import embed_text

@pytest.mark.integration
def test_embed_text():
    """
    Calls embed_text with a real string and confirms it returns a list of 1536 floats.
    Requires a valid OPENAI_API_KEY environment variable.
    """
    text = "Hello, CortexGit memory index!"
    result = embed_text(text)
    
    assert isinstance(result, list)
    assert len(result) == 1536
    assert all(isinstance(val, float) for val in result)

from streamlit_app import simple_clean, tokenize_and_remove_stopwords

def test_simple_clean():
    assert simple_clean(" Hello!!! ") == "Hello"

def test_tokenize_and_stopwords():
    toks = tokenize_and_remove_stopwords("This is a simple test sentence.")
    assert isinstance(toks, list)
    assert "simple" in [t.lower() for t in toks]
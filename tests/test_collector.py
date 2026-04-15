"""
tests/test_collector.py
────────────────────────
Testes unitários para a Camada 1 (collector).
"""

import pytest
from collector.deduplicator import compute_hash, canonicalize_url, Deduplicator
from collector.preprocessor import clean_text, tokenize_sentences, preprocess_article


# ── Deduplicator ──────────────────────────────────────────────────────────────

class TestCanonicalizeUrl:
    def test_removes_utm_params(self):
        url = "https://folha.com.br/noticia?utm_source=fb&utm_medium=social"
        clean = canonicalize_url(url)
        assert "utm_source" not in clean
        assert "utm_medium" not in clean

    def test_preserves_path(self):
        url = "https://folha.com.br/politica/2025/01/noticia.html"
        assert "politica/2025/01/noticia" in canonicalize_url(url)

    def test_lowercases_scheme_and_host(self):
        url = "HTTPS://FOLHA.COM.BR/noticia"
        canon = canonicalize_url(url)
        assert canon.startswith("https://folha.com.br")

    def test_removes_fragment(self):
        url = "https://folha.com.br/noticia#section1"
        assert "#" not in canonicalize_url(url)


class TestDeduplicator:
    def test_new_url_not_duplicate(self):
        dedup = Deduplicator()
        assert not dedup.is_duplicate("https://folha.com.br/noticia-1")

    def test_registered_url_is_duplicate(self):
        dedup = Deduplicator()
        url = "https://folha.com.br/noticia-2"
        dedup.register(url)
        assert dedup.is_duplicate(url)

    def test_different_urls_not_duplicate(self):
        dedup = Deduplicator()
        dedup.register("https://folha.com.br/noticia-a")
        assert not dedup.is_duplicate("https://folha.com.br/noticia-b")

    def test_same_url_different_tracking_params(self):
        dedup = Deduplicator()
        url_base = "https://folha.com.br/noticia-3"
        url_tracked = url_base + "?utm_source=twitter"
        dedup.register(url_base)
        # Com parâmetros de tracking removidos, devem ser considerados duplicata
        assert dedup.is_duplicate(url_tracked)


# ── Preprocessor ─────────────────────────────────────────────────────────────

class TestCleanText:
    def test_strips_html_tags(self):
        raw = "<p>Texto <strong>importante</strong>.</p>"
        assert "<" not in clean_text(raw)

    def test_removes_urls(self):
        raw = "Veja mais em https://exemplo.com sobre o assunto."
        assert "https://" not in clean_text(raw)

    def test_normalizes_whitespace(self):
        raw = "Texto   com    espaços\n\nextra."
        cleaned = clean_text(raw)
        assert "  " not in cleaned

    def test_respects_max_length(self, monkeypatch):
        import collector.preprocessor as prep
        monkeypatch.setattr(prep, "MAX_SNIPPET_CHARS", 10)
        raw = "A" * 50
        assert len(clean_text(raw)) <= 10


class TestTokenizeSentences:
    def test_splits_multiple_sentences(self):
        text = "O presidente discursou. A economia cresceu. O mercado reagiu."
        sentences = tokenize_sentences(text)
        assert len(sentences) >= 2

    def test_discards_too_short(self):
        text = "Ok. Sim. O ministro da Fazenda anunciou nova política fiscal."
        sentences = tokenize_sentences(text)
        # "Ok." e "Sim." devem ser descartados por serem muito curtos
        for s in sentences:
            assert len(s) >= 20

    def test_empty_string_returns_empty_list(self):
        assert tokenize_sentences("") == []


class TestPreprocessArticle:
    def test_returns_expected_keys(self):
        result = preprocess_article("O governo anunciou medidas econômicas hoje.")
        assert "snippet" in result
        assert "sentences" in result
        assert "sentence_count" in result

    def test_sentence_count_matches_list(self):
        result = preprocess_article("Primeira sentença longa o suficiente. Segunda sentença também longa.")
        assert result["sentence_count"] == len(result["sentences"])

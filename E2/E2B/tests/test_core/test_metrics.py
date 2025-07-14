import pytest
import numpy as np
from src.core.metrics import MetricsCalculator


class TestMetricsCalculator:

    @pytest.fixture
    def calculator(self):
        return MetricsCalculator()

    def test_semantic_similarity_identical(self, calculator):
        text = "This is a test sentence."
        similarity = calculator.calculate_semantic_similarity(text, text)
        assert similarity > 0.99

    def test_semantic_similarity_different(self, calculator):
        text1 = "The cat sat on the mat."
        text2 = "Dogs like to play fetch."
        similarity = calculator.calculate_semantic_similarity(text1, text2)
        assert 0 < similarity < 0.5

    def test_perfect_match(self, calculator):
        text1 = "Hello world"
        text2 = "  Hello world  "
        assert calculator.is_perfect_match(text1, text2)

    def test_perfect_match_different(self, calculator):
        text1 = "Hello world"
        text2 = "Hello World"
        assert not calculator.is_perfect_match(text1, text2)

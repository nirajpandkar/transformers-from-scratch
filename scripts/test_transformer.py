import torch
import pytest
from transformer import MultiHeadAttention
import logging

logger = logging.getLogger(__name__)

class TestScaledDotProductAttention:
    """Test suite for scaled_dot_product_attention function.

    This class provides a skeleton for testing the scaled_dot_product_attention
    implementation against PyTorch's built-in version. Tests will be added
    incrementally based on requirements.
    """

    @pytest.fixture
    def sample_inputs(self):
        """Fixture providing sample query, key, and value tensors."""
        batch_size, num_heads, seq_len, d_k = 1, 1, 3, 4
        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)
        return query, key, value

    def test_dimensions(self, sample_inputs):
        """Test that output dimensions match PyTorch's implementation."""
        query, key, value = sample_inputs

        # User's implementation
        output_user = MultiHeadAttention.scaled_dot_product_attention(query, key, value)

        # PyTorch's implementation
        output_torch = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        # Verify dimensions match
        assert output_user.shape == output_torch.shape
        assert output_user.shape == (1, 1, 3, 4)  # Expected shape based on inputs

    def test_output_values(self, sample_inputs):
        """Test that output values are close to PyTorch's implementation."""
        query, key, value = sample_inputs

        # User's implementation
        output_user = MultiHeadAttention.scaled_dot_product_attention(query, key, value)

        # PyTorch's implementation
        output_torch = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        logger.info("My Output")
        logger.info(output_user)

        logger.info("Library output")
        logger.info(output_torch)
        # Verify outputs are numerically close
        assert torch.allclose(output_user, output_torch, atol=1e-5)


# Placeholder for additional test classes and functions
# Add more test classes here as new functions are implemented

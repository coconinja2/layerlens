import pytest
import torch

from layer_profiler.repeatability import (
    FiniteDomainUnaryAtlas,
    profile_attention_heads,
    profile_tensor_repetition,
)


def sample_vectors():
    return torch.tensor(
        [[1, 2, 3, 4], [1, 2, 5, 6], [7, 8, 5, 6]], dtype=torch.bfloat16
    )


def test_tensor_profile_separates_scalar_coordinate_tile_and_vector_reuse():
    report = profile_tensor_repetition(sample_vectors(), signal="test", tile_size=2)

    assert report.distinct_scalar_bit_patterns == 8
    assert report.scalar_replay_hits == 4
    assert report.coordinate_replay_hits == 4
    assert report.tile_lookups == 6
    assert report.tile_replay_hits == 2
    assert report.vector_replay_hits == 0
    assert report.consecutive_unchanged_elements == 4
    assert report.consecutive_unchanged_ratio == pytest.approx(0.5)


def test_finite_domain_silu_atlas_matches_native_bfloat16_bits():
    atlas = FiniteDomainUnaryAtlas("silu", torch.bfloat16)
    values = torch.tensor([-10.0, -1.0, -0.0, 0.0, 1.0, 10.0], dtype=torch.bfloat16)

    actual = atlas(values)
    expected = torch.nn.functional.silu(values)
    report = atlas.profile(values, iterations=2)

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert report.domain_entries == 65536
    assert report.table_bytes == 131072
    assert report.approximation_added is False
    assert report.bitwise_output_agreement == 1.0
    assert report.max_absolute_error == 0.0


def test_attention_head_profile_finds_only_exact_head_vectors():
    report = profile_attention_heads(
        sample_vectors(), signal="query", heads=2, head_dim=2
    )

    assert report.head_vectors == 6
    assert report.exact_head_replay_hits == 2
    assert report.same_head_position_hits == 2
    assert report.within_sample_duplicate_heads == 0


def test_complete_unary_atlas_rejects_float32_domain():
    with pytest.raises(ValueError, match="float16 and bfloat16"):
        FiniteDomainUnaryAtlas("sigmoid", torch.float32)

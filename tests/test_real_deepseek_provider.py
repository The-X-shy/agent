"""Test real DeepSeek provider (opt-in only).

Requires:
  OPTIRESEARCH_ENABLE_REAL_LLM_TESTS=1
  DEEPSEEK_API_KEY=<valid key>
"""

import os

import pytest
from pydantic import BaseModel

from optiresearch.llm.deepseek_provider import DeepSeekProvider
from optiresearch.llm.registry import get_llm_provider


class TinySchema(BaseModel):
    answer: str


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_LLM_TESTS") != "1" or not os.getenv("DEEPSEEK_API_KEY"),
    reason="Real LLM test requires explicit opt-in and DEEPSEEK_API_KEY.",
)
def test_real_deepseek_provider_minimal_structured_output():
    provider = DeepSeekProvider()

    response = provider.complete([{"role": "user", "content": "Hello"}])
    structured = provider.structured_complete(
        [{"role": "user", "content": "Return JSON with an answer field saying hello."}],
        TinySchema,
    )

    assert provider.available() is True
    assert response.content
    assert structured.answer


@pytest.mark.skipif(
    os.getenv("OPTIRESEARCH_ENABLE_REAL_LLM_TESTS") != "1" or not os.getenv("DEEPSEEK_API_KEY"),
    reason="Real LLM test requires explicit opt-in and DEEPSEEK_API_KEY.",
)
def test_real_deepseek_provider_minimal_planning():
    """Test that DeepSeek can propose a minimal autonomous loop plan."""
    provider = get_llm_provider("deepseek")
    assert provider.available() is True

    from optiresearch.schemas.autonomous import ResearchIterationPlan
    plan = provider.structured_complete(
        [{"role": "user", "content": (
            "Propose one optical-HSI experiment. Use controlled_chromatic_edof encoder "
            "and optical_conditioned_linear reconstructor. "
            "Output valid JSON matching ResearchIterationPlan schema."
        )}],
        ResearchIterationPlan,
    )
    assert isinstance(plan, ResearchIterationPlan)
    assert plan.selected_encoder
    assert plan.selected_reconstructor
    assert plan.hypothesis

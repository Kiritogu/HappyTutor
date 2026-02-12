from src.agents.research.agents.reporting_agent import ReportingAgent


def test_extract_llm_text_field_prefers_json_field():
    response = '{"section_content": "Structured section body"}'

    content, used_fallback = ReportingAgent._extract_llm_text_field(
        response_text=response,
        field_name="section_content",
    )

    assert content == "Structured section body"
    assert used_fallback is False


def test_extract_llm_text_field_falls_back_to_raw_markdown():
    response = "## Section\n\nThis section is plain markdown, not JSON."

    content, used_fallback = ReportingAgent._extract_llm_text_field(
        response_text=response,
        field_name="section_content",
    )

    assert "plain markdown" in content
    assert used_fallback is True


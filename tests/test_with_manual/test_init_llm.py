from biominer_llm import init_llm, LLMConfig
import os


def test_init_llm(messages):
    config = LLMConfig(provider="openai", model="gpt-4o-mini")
    llm = init_llm(config=config)
    res = llm.invoke(messages)
    assert "2" in res.content


def test_init_llm_bigmodel(messages):
    config = LLMConfig(
        model="glm-4.5",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key=os.getenv("BIGMODEL_API_KEY"),
    )

    llm = init_llm(config=config)
    res = llm.invoke(messages)
    assert "2" in res.content

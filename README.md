# biominer-llm

## install

```bash
git clone https://github.com/open-prophetdb/biominer-llm.git
cd biominer-llm
pip install -e .
```

## usage    


### load config from .env file
```
from biominer_llm import LLMConfig, init_llm
messages = [
(
    "system",
    "You are a helpful translator. Translate the user sentence to Chinese.",
),
("human", "I love programming."),
]

llm = init_llm()
llm.invoke(messages)
```

### manually config 
```
messages = [
(
    "system",
    "You are a helpful translator. Translate the user sentence to Chinese.",
),
("human", "I love programming."),
]

config = LLMConfig(
    provider="openai",
    model="gpt-4o-mini"
)

llm = init_llm(config=config)
llm.invoke(messages)
```

### config with custom llm provider base_url
```
messages = [
(
    "system",
    "You are a helpful translator. Translate the user sentence to Chinese.",
),
("human", "I love programming."),
]

config = LLMConfig(
    model="glm-4.5",
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key=os.getenv("BIOMINER_AI_LLM_API_KEY")
)

llm = init_llm(config=config)
llm.invoke(messages)
```

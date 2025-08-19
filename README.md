# biominer-llm

## install

```bash
git clone https://github.com/open-prophetdb/biominer-llm.git
cd biominer-llm
pip install -e .
```

## supported providers:
- openai
- anthropic
- ollama
- xai
- gemini
- deepseek
- groq

Additionally, if you’re using a provider that’s not listed above, you can set a custom base_url instead.

## Priority

manual setting
```
config = LLMConfig( provider="openai", model="gpt-4o-mini" )

```
> 

biominer prefix env
```
import os
os.environ["BIOMINER_AI_LLM_PROVIDER"] = "openai"
os.environ["BIOMINER_AI_LLM_MODEL"] = "gpt-4o-mini"

from biominer_llm import LLMConfig, init_llm
config = LLMConfig()

```
>

biominer prefix env config file
```
import os
os.environ["BIOMINER_AI_ENV"] = "BIOMINER_AI_ENV.env.file"

from biominer_llm import LLMConfig, init_llm
config = LLMConfig()

## >

## default .env file
from biominer_llm import LLMConfig, init_llm
config = LLMConfig()

```
>

provider prefix env
```
import os
os.environ["OPENAI_API_KEY"] = "SK***"

from biominer_llm import LLMConfig, init_llm
config = LLMConfig()


```

## usage    


### load config from configure file

load from default .env file
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

load from default custom .env file

```
from biominer_llm import LLMConfig, init_llm
messages = [
(
    "system",
    "You are a helpful translator. Translate the user sentence to Chinese.",
),
("human", "I love programming."),
]
config = LLMConfig(_env_file="biominer.env")

llm = init_llm(config=config)
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

# vector-db

[semantic kernel](https://github.com/microsoft/semantic-kernel/tree/main) run using [llamafile](https://github.com/Mozilla-Ocho/llamafile) which exposes a server providing OpenAI compatible API for chat completion and embeddings.
  - ```./Llama-3.2-3B-Instruct.Q6_K.llamafile --server --v2 --verbose```
  - set ```export OPENAI_BASE_URL="http://127.0.0.1:8080/v1/"``` so that the requests to OpenAI are routed to the local server

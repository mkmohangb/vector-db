import asyncio
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_embedding import OpenAITextEmbedding
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.kernel import Kernel
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion("llama2", "blah"))
embedding_gen = OpenAITextEmbedding("llama2", "blah")
kernel.add_service(embedding_gen)

memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=embedding_gen)
kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")

collection_id = "generic"

async def populate_memory(memory: SemanticTextMemory) -> None:
    await memory.save_information(collection=collection_id, id="info3", 
                                  text="Your investments are $80,000")
    await memory.save_information(collection=collection_id, id="info1", 
                                  text="Your budget for 2024 is $100,000")
    await memory.save_information(collection=collection_id, id="info2", 
                                  text="Your savings from 2023 are $50,000")

async def search_memory_examples(memory: SemanticTextMemory) -> None:
    questions = [
        "What is my budget for 2024?",
        "What are my savings from 2023?",
        "What are my investments?",
    ]
    
    for question in questions:
        print(f"Question: {question}")
        result = await memory.search(collection_id, question)
        print(f"Answer: {result[0].text}\n")


async def main():
    await populate_memory(memory)

    await search_memory_examples(memory)



if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion("llama2", "blah"))


plugins_directory = "prompt_templates"

funFunctions = kernel.add_plugin(parent_directory=plugins_directory, plugin_name="FunPlugin")
jokeFunction = funFunctions["Joke"]

async def main():
    result = await kernel.invoke(jokeFunction, input="travel to dinosaur age", style="silly")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())


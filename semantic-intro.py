import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings


kernel = sk.Kernel()

kernel.add_service(OpenAIChatCompletion("llama2", "blah"))


print("kernel created")

# Define the request settings
req_settings = PromptExecutionSettings()

prompt = """
1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.

Give me the TLDR in exactly 5 words."""

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="tldr",
    template_format="semantic-kernel",
    execution_settings=req_settings,
)

function = kernel.add_function(
    function_name="tldr_function",
    plugin_name="tldr_plugin",
    prompt_template_config=prompt_template_config,
)

# Run your prompt
# Note: functions are run asynchronously
async def main():
    result = await kernel.invoke(function)
    print(result) # => Robots must not harm humans.

if __name__ == "__main__":
    asyncio.run(main())

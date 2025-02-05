import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.core_plugins.text_plugin import TextPlugin
from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
from semantic_kernel.planners import SequentialPlanner
from semantic_kernel.planners.sequential_planner.sequential_planner_config import SequentialPlannerConfig

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion("llama2", "blah"))

plugins_directory = "prompt_templates"

text_plugin = kernel.add_plugin(plugin=TextPlugin(), plugin_name="TextPlugin")
summarize_plugin = kernel.add_plugin(plugin_name="SummarizePlugin", parent_directory=plugins_directory)
shakespeare_func = KernelFunctionFromPrompt(
    function_name="Shakespeare",
    plugin_name="WriterPlugin",
    prompt="""
{{$input}}

Rewrite the above in the style of Shakespeare.
""",
    prompt_execution_settings=OpenAIChatPromptExecutionSettings(
        service_id=None,
        max_tokens=2000,
        temperature=0.8,
    ),
    description="Rewrite the input in the style of Shakespeare.",
)
kernel.add_function(plugin_name="WriterPlugin", function=shakespeare_func)

for plugin_name, plugin in kernel.plugins.items():
    for function_name, function in plugin.functions.items():
        print(f"Plugin: {plugin_name}, Function: {function_name}")

config = SequentialPlannerConfig()
config.allow_missing_functions = True
planner = SequentialPlanner(kernel, "default", config)

async def main():
    ask = """
    Tomorrow is Valentine's day. I need to come up with a few short poems.
    She likes Shakespeare so write using his style. She speaks French so write it in French.
    Convert the text to uppercase.
    """
    sequential_plan = await planner.create_plan(goal=ask)
    print("The plan's steps are:")
    for step in sequential_plan._steps:
        print(
        f"- {step.description.replace('.', '') if step.description else 'No description'} using {step.metadata.fully_qualified_name} with parameters: {step.parameters}"
        )
    result = await sequential_plan.invoke(kernel)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

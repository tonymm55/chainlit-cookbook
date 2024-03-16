import json
import ast
import os
from openai import AsyncOpenAI

from chainlit.playground.providers.openai import stringify_function_call
import chainlit as cl

api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

MAX_ITER = 5

def get_office_opening_hours(day, hour):
    """Check if the office is open or closed"""
    opening_hours = {
        "Monday": range(8, 20),
        "Tuesday": range(8, 20),
        "Wednesday": range(8, 20),
        "Thursday": range(8, 20),
        "Friday": range(8, 20),
        "Saturday": range(8, 20),
        "Sunday": range(8, 20)
    }

    if day in opening_hours and hour in opening_hours[day]:
        return "The office is open."
    else:
        return "The office is closed."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_office_opening_hours",
            "description": "Check if the office is open or closed",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "The day of the week (e.g., 'Monday', 'Tuesday', etc.)",
                    },
                    "hour": {
                        "type": "integer",
                        "description": "The hour of the day (24-hour format)",
                    },
                },
                "required": ["day", "hour"],
            },
        },
    },
]

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )

@cl.step(type="tool")
async def call_tool(tool_call, message_history):
    function_name = tool_call.function.name
    arguments = ast.literal_eval(tool_call.function.arguments)

    current_step = cl.context.current_step
    current_step.name = function_name

    current_step.input = arguments

    if function_name == "get_office_opening_hours":
      function_response = get_office_opening_hours(
        day=arguments.get("day"),
        hour=arguments.get("hour"),
    )
    else:
      function_response = "Function not found."

    current_step.output = function_response
    current_step.language = "text"

    message_history.append(
        {
            "role": "function",
            "name": function_name,
            "content": function_response,
            "tool_call_id": tool_call.id,
        }
    )


@cl.step(type="llm")
async def call_gpt4(message_history):
    settings = {
        "model": "gpt-4",
        "tools": tools,
        "tool_choice": "auto",
    }

    cl.context.current_step.generation = cl.ChatGeneration(
        provider="openai-chat",
        messages=[
            cl.GenerationMessage(
                formatted=m["content"], name=m.get("name"), role=m["role"]
            )
            for m in message_history
        ],
        settings=settings,
    )

    response = await client.chat.completions.create(
        messages=message_history, **settings
    )

    message = response.choices[0].message

    for tool_call in message.tool_calls or []:
        if tool_call.type == "function":
            await call_tool(tool_call, message_history)

    if message.content:
        cl.context.current_step.generation.completion = message.content
        cl.context.current_step.output = message.content

    elif message.tool_calls:
        completion = stringify_function_call(message.tool_calls[0].function)

        cl.context.current_step.generation.completion = completion
        cl.context.current_step.language = "json"
        cl.context.current_step.output = completion

    return message


@cl.on_message
async def run_conversation(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    cur_iter = 0

    while cur_iter < MAX_ITER:
        message = await call_gpt4(message_history)
        if not message.tool_calls:
            await cl.Message(content=message.content, author="Answer").send()
            break

        cur_iter += 1

import json
import ast
import os
from openai import AsyncOpenAI
from datetime import datetime, time, timedelta

from chainlit.playground.providers.openai import stringify_function_call
import chainlit as cl

api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# 1. Ask ChatGPT a Question, RuntimeWarning: coroutine 'AsyncCompletions.create' was never awaited

# 2. Use OpenAI’s Function Calling Feature

tools = [
    {
        "type": "function",
        "function":  {
            "name": "get_business_hours",
            "description": "Check business open hours before scheduling appointment",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "Day of the week, e.g. Monday",
                    },
                    "open": {
                        "type": "string",
                        "description": "Office opening time, e.g. 9am",
                    },
                    "close": {
                        "type": "string",
                        "description": "Office closing time, e.g. 8pm",
                    },
                },
                "required": ["day"],
            },
        }
    },
    {
        "type": "function",
        "function":  {
            "name": "is_business_open",
            "description": "Check if business is open now",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_day": {
                        "type": "string",
                        "description": "Today's day, e.g. Monday",
                    },
                    "current_time": {
                        "type": "string",
                        "description": "Current time, e.g. 09:30",
                    },
                },
                "required": ["day"],
            },
        }
    },
]

def get_business_hours(day=None):
    # Define office hours for each day of the week
    office_hours = {
        "Monday": {"open": "8:00", "close": "19:00"},
        "Tuesday": {"open": "8:00", "close": "19:00"},
        "Wednesday": {"open": "8:00", "close": "19:00"},
        "Thursday": {"open": "8:00", "close": "19:00"},
        "Friday": {"open": "8:00", "close": "19:00"},
        "Saturday": {"open": "9:00", "close": "16:00"},
        "Sunday": {"open": "10:00", "close": "15:00"},
    }

    if day is None:
        return json.dumps(office_hours)
    elif day in office_hours:
        return json.dumps(office_hours[day])
    else:
        return "Invalid day"
    
office_hours = get_business_hours()
print(office_hours)

def is_business_open(current_day=None, current_time=None):
    # If current_day and current_time are not provided, fetch them
    if current_day is None or current_time is None:
        now = datetime.now()
        current_day = now.strftime("%A")
        current_time = now.time()
        
        
    office_hours_json = get_business_hours()
    office_hours = json.loads(office_hours_json)

    # Get the office hours for the current day
    opening_time_str, closing_time_str = office_hours[current_day]["open"], office_hours[current_day]["close"]
    opening_time = datetime.strptime(opening_time_str, "%H:%M").time()
    closing_time = datetime.strptime(closing_time_str, "%H:%M").time()

    # Check if the current time is within the office hours
    if opening_time <= current_time <= closing_time:
        return "The office is open."
    else:
        return "The office is closed."

print(is_business_open())
print(datetime.now())

@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant named Nat, whose aim is to book a call with an advisor within office open hours."}],
    )

@cl.step(type="tool")
async def call_tool(tool_call, message_history):
    function_name = tool_call.function.name
    arguments = ast.literal_eval(tool_call.function.arguments)

    current_step = cl.context.current_step
    current_step.name = function_name

    current_step.input = arguments

    function_response = get_business_hours(
        day=arguments.get("day") 
    )
    
    function_response = is_business_open(
        current_day=arguments.get("current_day"), 
        current_time=arguments.get("current_time") 
    )

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
    
    # Append the function response as a new message to the history
    if cl.context.current_step.output:
        message_history.append({
            "role": "function",
            "content": cl.context.current_step.output,
            "tool_call_id": cl.context.current_step.id,
    })

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
    print(message)
    

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
    MAX_ITER = 5

    while cur_iter < MAX_ITER:
        message = await call_gpt4(message_history)
        if not message.tool_calls:
            await cl.Message(content=message.content, author="Answer").send()
            break

        cur_iter += 1
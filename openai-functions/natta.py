import json
import ast
import os
from openai import AsyncOpenAI
from datetime import datetime, time, timedelta
from dotenv import load_dotenv

from chainlit.playground.providers.openai import stringify_function_call
import chainlit as cl

api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

MAX_ITER = 5

tools = [
    {
        "name": "get_business_hours",
        "description": "Check office opening hours before scheduling appointment",
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
            "required": ["day", "open", "close"],
        },
    }
]

user_prompt = "Can you suggest an appointment today or tomorrow within opening and closing times for that day?"

# completion = client.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": user_prompt}],
#     functions=tools,
#     function_call="auto"
# )
# output = completion.choices[0].message
# print(output)

completion = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": user_prompt}],
    functions=tools,
function_call="auto")
  
output = completion.choices[0].message

output = completion.choices[0].message
print(output)

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

monday = json.loads(output.function_call.arguments).get("open")
params = json.loads(output.function_call.arguments)
type(params)

print(monday)
print(params)

chosen_function = eval(output.function_call.name)
office_hours=chosen_function()
print(office_hours)

def is_office_open():
    office_hours_json = get_business_hours()
    office_hours = json.loads(office_hours_json)

    current_day = datetime.now().strftime("%A")
    current_time = datetime.now().time()

    # Get the office hours for the current day
    opening_time_str, closing_time_str = office_hours[current_day]["open"], office_hours[current_day]["close"]
    opening_time = datetime.strptime(opening_time_str, "%H:%M").time()
    closing_time = datetime.strptime(closing_time_str, "%H:%M").time()

    # Check if the current time is within the office hours
    if opening_time <= current_time <= closing_time:
        return "The office is open."
    else:
        return "The office is closed."

print(is_office_open())

office_hours = get_business_hours()
# Confirm the content of the office hours variable
print(office_hours)

# Ensure that office_hours is not None and is properly formatted JSON
if office_hours:
    print("office_hours is not None")
    try:
        office_hours_json = json.loads(office_hours)
        print("office_hours is valid JSON")
    except json.JSONDecodeError as e:
        print("Error decoding office_hours JSON:", e)
else:
    print("office_hours is None")

# Check the completion request to ensure office_hours is passed correctly
print("Completion request:")
print({
    "role": "function",
    "name": output.function_call.name,
    "content": office_hours
})

second_completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": user_prompt},
        {"role": "function", "name": output.function_call.name, "content": office_hours},
    ],
    functions=tools,
)

# Check the response from the completion request
print("Completion response:")
print(second_completion)

# Manually call the function with different inputs
print("Testing get_business_hours() function:")

# Test case 1: Monday
print("Testing for Monday:")
monday_hours = get_business_hours("Monday")
print("Monday office hours:", monday_hours)

# Test case 2: Saturday
print("Testing for Saturday:")
saturday_hours = get_business_hours("Saturday")

# Test case 3: Sunday
print("Testing for Sunday:")
sunday_hours = get_business_hours("Sunday")
print("Sunday office hours:", sunday_hours)


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

    if function_name == "get_business_hours":
      function_response = get_business_hours(
        day=arguments.get("day"),
        open=arguments.get("open"),
        close=arguments.get("close"),
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
async def on_message(message: cl.Message):
    response = await client.chat.completions.create(
        messages=[
            {
                "content": "You are a helpful bot, you always reply in Spanish",
                "role": "system"
            },
            {
                "content": input,
                "role": "user"
            }
        ],
    )
    await cl.Message(content=response.choices[0].message.content).send()
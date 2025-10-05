import os
from pydantic import BaseModel, Field
from typing import Optional, List
from langgraph.graph import StateGraph, END
from tools import TOOLS
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

# --- Initialize LLM ---

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=api_key,
    temperature=0.7
)

memory_saver = InMemorySaver()
config1 = {"configurable": {"thread_id": 1}}
# --- Custom Schemas ---
class ToolCallRequest(BaseModel):
    user_query: str
    parameters: dict = {}

class AgentState(BaseModel):
    request: ToolCallRequest = None
    result: dict = None
    tool_name: str = None
    memory: dict = {} 

# --- Utility: Detect Missing Schema Fields ---
def get_missing_schema(tool_name: str, params: dict):
    if tool_name not in TOOLS:
        return []

    input_model, _, _ = TOOLS[tool_name]
    missing = []
    for name, field in input_model.model_fields.items():
        if field.is_required() and (name not in params or params[name] is None):
            missing.append({
                "name": name,
                "description": getattr(field, "description", "No description available"),
                "type": str(field.annotation)
            })
    return missing

# --- Node: Classify Tool ---
def classify_tool(state: AgentState):
    print("[Step] Classifying the tool based on user query...")
    user_query = state.request.user_query
    tool_descs = "\n".join([f"{name}: {desc}" for name, (_, _, desc) in TOOLS.items()])

    prompt = f"""
        You are an AI orchestrator. User said: "{user_query}". Available tools: {tool_descs}
        Return ONLY the exact tool key that best matches the query.
        If no tool matches, return "chat_tool".
        """
    tool_name = client.invoke(prompt).content.strip().lower()
    print(f"[Info] Selected tool: {tool_name}")
    return {"tool_name": tool_name}

# --- Node: Extract Parameters ---
def extract_parameters(state: AgentState):
    print("[Step] Extracting parameters from user query...")
    user_query = state.request.user_query
    tool_name = state.tool_name

    if tool_name not in TOOLS:
        print("[Warning] Tool not found in TOOLS. Returning empty parameters.")
        return {"request": ToolCallRequest(user_query=user_query, parameters={})}

    input_model, _, _ = TOOLS[tool_name]
    schema = input_model.model_json_schema()

    prompt = f"""
Extract parameters for tool '{tool_name}' from user query: "{user_query}".
Schema: {schema}
- Include only explicitly mentioned or obvious fields.
- Do NOT assume values for missing fields.
- Do not set the value of descriptive yourself.
- Return JSON only.
"""
    response = client.invoke(prompt)
    raw_text = response.content.strip().replace("```json", "").replace("```", "")

    try:
        params = json.loads(raw_text)
    except:
        print("[Warning] Failed to parse parameters JSON. Using empty dict.")
        params = {}

    print(f"[Info] Extracted parameters: {params}")
    return {"request": ToolCallRequest(user_query=user_query, parameters=params)}

# --- Node: Ask Missing Parameters ---
def ask_missing_params(state: AgentState):
    missing = get_missing_schema(state.tool_name, state.request.parameters)
    if not missing:
        print("[Info] No missing parameters. Proceeding to tool execution.")
        return {"next_node": "call_tool"}

    # Create a single combined question for all missing fields
    print("[Info] Missing parameters are", missing)
    questions = ", ".join([f"{field['name']} ({field['type']})" for field in missing])
    print(f"[Info] Asking user for: {questions}")
    prompt_text = (
        f"[Step] Missing parameters detected for tool '{state.tool_name}': {questions}\n"
        f"AI: Please provide values for these fields in a single response. "
        f"If you leave any field empty, I will fill it with default or inferred values.\n"
        f"User:"
    )
    asking_prompt = f"You are an AI assistant helping to fill missing parameters for a tool. Please provide values for: {missing}. Eg: Missing parameters are 'name': 'descriptive', 'description': 'Whether explanation should be descriptive', 'type': <class bool>, then you need to ask question like 'Do you want the explanation to be descriptive? (yes/no)'"
    asking_text = client.invoke(asking_prompt)
    asking_text_display = asking_text.content.strip()

    # Ask user input
    user_response = input(asking_text_display + "\n> ").strip()

    # Combine user response and schema + LLM to parse into structured JSON
    llm_prompt = f"""
        You are an intelligent assistant. You need to fill missing parameters for a tool.
        Tool: {state.tool_name}
        Missing fields: {missing}
        User response: "{user_response}"
        for example, Missing parameters are ['name': 'descriptive', 'description': 'Whether explanation should be descriptive', 'type': <class 'bool'>], then you need to extract the value of 'descriptive' from user response and set it to the type it belongs, if user response does not contain the value of 'descriptive', you need to set it to a sensible default value like False.
        
        Rules:
        - Extract values for missing fields from the user response.
        - If a value is missing in the user response, infer a sensible default.
        - Return only a valid JSON object containing all missing fields with their values.
        - Ensure types match the expected types (e.g., convert "yes"/"no" to boolean).
        """
    llm_result = client.invoke(llm_prompt)
    raw_text = llm_result.content.strip()
    clean_text = raw_text.replace("```json", "").replace("```", "").strip()
    print(f"[Info] LLM response for missing params: {clean_text}")


    try:
        user_values = json.loads(clean_text)
    except json.JSONDecodeError:
        print("[Warning] LLM did not return valid JSON. Using defaults for missing fields.")
        user_values = {}
        input_model, _, _ = TOOLS[state.tool_name]
        for name, field in input_model.model_fields.items():
            if field.is_required() and (name not in state.request.parameters or state.request.parameters[name] is None):
                if field.annotation == bool:
                    user_values[name] = False
                elif field.annotation == int:
                    user_values[name] = 1
                else:
                    user_values[name] = ""

    input_model, _, _ = TOOLS[state.tool_name]
    for key, value in user_values.items():
        field_type = input_model.model_fields[key].annotation
        if field_type == bool and isinstance(value, str):
            user_values[key] = value.lower() in ["true", "yes", "1"]
        elif field_type == int:
            user_values[key] = int(value)

    state.request.parameters.update(user_values)
    print(f"[Info] Updated parameters after AI parsing: {state.request.parameters}")

# --- Node: Call Tool ---
def call_tool(state: AgentState):
    print("[Step] Calling the selected tool with parameters...")
    tool_name = state.tool_name
    params = state.request.parameters
    if tool_name not in TOOLS:
        print(f"[Error] Unknown tool: {tool_name}")
        return {"result": {"error": f"Unknown tool {tool_name}"}}

    input_model, func, _ = TOOLS[tool_name]
    validated_input = input_model(**params)
    result = func(validated_input)
    state.memory["last_result"] = result
    return {"result": result}

graph = StateGraph(AgentState)
graph.add_node("classify_tool", classify_tool)
graph.add_node("extract_parameters", extract_parameters)
graph.add_node("ask_missing_params", ask_missing_params)
graph.add_node("call_tool", call_tool)

graph.set_entry_point("classify_tool")
graph.add_edge("classify_tool", "extract_parameters")
graph.add_edge("extract_parameters", "ask_missing_params")
graph.add_edge("ask_missing_params", "call_tool")
graph.add_edge("call_tool", END)

workflow = graph.compile(checkpointer=memory_saver)

if __name__ == "__main__":
    config1 = {"configurable": {"thread_id": 1}}

    quit_keywords = ["bye", "quit", "exit"]

    while True:
        user_input = input("\nEnter your query: ").strip()
        if user_input.lower() in quit_keywords:
            print("Exiting... Goodbye!")
            break

        request = ToolCallRequest(
            user_query=user_input,
            parameters={}
        )
        state = AgentState(request=request)
        result = workflow.invoke(state, config=config1)
        print("\n[Final Result]\n", result.get("result") if isinstance(result, dict) and "result" in result else result)

# def run_workflow(user_input: str):
#     """Run the LangGraph workflow for a given user query."""
#     config1 = {"configurable": {"thread_id": 1}}

#     request = ToolCallRequest(
#         user_query=user_input,
#         parameters={}
#     )
#     state = AgentState(request=request)
#     result = workflow.invoke(state, config=config1)
#     return result
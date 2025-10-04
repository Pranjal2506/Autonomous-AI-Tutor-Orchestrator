import os
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from tools import TOOLS
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyAL_soAvn-rgYHGfSzvosTpF7pbBnRapqk"
client = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.7
)

# --- Custom Schema for Orchestration ---
class ToolCallRequest(BaseModel):
    user_query: str = Field(..., description="Natural language request from user")
    parameters: dict = Field(..., description="Parameters for the selected tool")

# --- State ---
class AgentState(BaseModel):
    request: ToolCallRequest = None
    result: dict = None
    tool_name: str = None


# --- Node: Select Tool with Groq ---
def classify_tool(state: AgentState):
    user_query = state.request.user_query
    
    # Prepare tool descriptions for Groq
    tool_descriptions = "\n".join([f"{name}: {desc}" for name, (_, _, desc) in TOOLS.items()])
    
    prompt = f"""
    You are an AI orchestrator. A user said: "{user_query}".
    Available tools:
    {tool_descriptions}
    
    Instructions:
        1. Choose the most appropriate tool key from the description above.
        2. Return ONLY the exact tool key. Do NOT add any explanations, punctuation, or quotes.
        3. If the query does not clearly match any tool, return the key of the closest relevant tool.
        4. NEVER invent a tool key or return any value not in the list.

    Decide the best tool name to use (just return the tool key: quiz_generator, flashcard_maker, concept_explainer, summary_generator, example_creator, comparison_tool, answer_checker, topic_expansion).
    """
    
    response = client.invoke(prompt)
    tool_name = response.content.strip().lower()
    return {"tool_name": tool_name}


def extract_parameters(state: AgentState):
    user_query = state.request.user_query
    tool_name = state.tool_name
    
    if tool_name not in TOOLS:
        return {"parameters": {}}
    
    input_model, _, _ = TOOLS[tool_name]
    schema = input_model.model_json_schema()
    print(f"Schema for {tool_name}: {schema}")
    
    prompt = f"""
    You are a parameter extractor.
    User query: "{user_query}"
    Tool: {tool_name}
    Expected parameters schema: {schema}
    
    Return only a valid JSON object that matches the schema.
    """
    
    response = client.invoke(prompt)
    print(f"Raw response for parameters: {response.content}")
    raw_text = response.content
    clean_text = raw_text.strip().replace("```json", "").replace("```", "").strip()
    import json
    try:
        params = json.loads(clean_text)
        "geuy"
    except:
        params = {}
    
    print(f"Extracted parameters for {tool_name}: {params}")
    return {"request": ToolCallRequest(user_query=user_query, parameters=params)}


# --- Node: Call Selected Tool ---
def call_tool(state: AgentState):
    tool_name = state.tool_name
    params = state.request.parameters
    if tool_name not in TOOLS:
        return {"result": {"error": f"Unknown tool {tool_name}"}}
    
    input_model, func, _ = TOOLS[tool_name]
    validated_input = input_model(**params)  # schema validation
    result = func(validated_input)
    return {"result": result}


graph = StateGraph(AgentState)

graph.add_node("classify_tool", classify_tool)
graph.add_node("call_tool", call_tool)
graph.add_node("extract_parameters", extract_parameters)

graph.set_entry_point("classify_tool")
graph.add_edge("classify_tool", "extract_parameters")
graph.add_edge("extract_parameters", "call_tool")
graph.add_edge("call_tool", END)

workflow = graph.compile()


if __name__ == "__main__":
    request = ToolCallRequest(
        user_query="Can you evaluate my answer of the photosynthesis question and provide feedback? This is the ans: Photosynthesis is the process of reproduction of plants which they do asexually.",
        parameters = {"topic": "", "difficulty": "", "num_items": 0, "descriptive": True}
    )
    state = AgentState(request=request)
    result = workflow.invoke(state)
    print("Final Result:\n", result)

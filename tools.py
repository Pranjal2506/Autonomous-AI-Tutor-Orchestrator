# tools.py
from pydantic import BaseModel, Field
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Unified Input Model ---
class ToolInput(BaseModel):
    topic: str = Field(..., description="The topic, concept, or subject")
    difficulty: str = Field("easy", description="Difficulty level: easy, medium, hard")
    num_items: int = Field(3, description="Number of questions, flashcards, or explanations")
    descriptive: bool = Field(True, description="Whether explanation should be descriptive (only used in Concept Explainer)")
    
os.environ["GOOGLE_API_KEY"] = "AIzaSyAL_soAvn-rgYHGfSzvosTpF7pbBnRapqk"
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.7
)

# --- Tool 1: Quiz Generator ---
def quiz_generator_tool(input: ToolInput):
    prompt = f"Generate {input.num_items} {input.difficulty} quiz questions on the topic: {input.topic}."
    response = llm.invoke(prompt)
    return {
        response.content
    }

quiz_generator_desc = "Generates practice quiz questions based on topic and difficulty."

# --- Tool 2: Flashcard Maker ---
def flashcard_maker_tool(input: ToolInput):
    return {
        "tool": "FlashcardMaker",
        "topic": input.topic,
        "flashcards": [
            {"question": f"What is {input.topic}?", "answer": f"Explanation of {input.topic}"}
            for _ in range(input.num_items)
        ]
    }

flashcard_maker_desc = "Creates flashcards for a given topic. Supports examples."

# --- Tool 3: Concept Explainer ---
def concept_explainer_tool(input: ToolInput):
    return {
        "tool": "ConceptExplainer",
        "topic": input.topic,
        "explanations": [
            f"This is a {input.difficulty} explanation of {input.topic}."
            for _ in range(input.num_items)
        ],
        "style": "descriptive" if input.descriptive else "concise"
    }

concept_explainer_desc = "Explains a concept at a chosen difficulty level. Can be descriptive or concise."

# --- Tool 4: Summary Generator ---
def summary_generator_tool(input: ToolInput):
    prompt = f"Summarize the topic '{input.topic}' in {input.num_items} key points."
    response = llm.invoke(prompt)
    return {"tool": "SummaryGenerator", "summary": response.content}

summary_generator_desc = "Generates a concise summary of a topic in key points."

# --- Tool 5: Example Creator ---
def example_creator_tool(input: ToolInput):
    prompt = f"Create {input.num_items} examples to illustrate the concept '{input.topic}'."
    response = llm.invoke(prompt)
    return {"tool": "ExampleCreator", "examples": response.content}

example_creator_desc = "Creates practical examples for a given topic or concept."

# --- Tool 6: Comparison Tool ---
def comparison_tool(input: ToolInput):
    prompt = f"Compare '{input.topic}' with related concepts in {input.num_items} points."
    response = llm.invoke(prompt)
    return {"tool": "ComparisonTool", "comparison": response.content}

comparison_desc = "Compares a topic with related concepts in a structured format."

# --- Tool 7: Quiz Answer Checker ---
def answer_checker_tool(input: ToolInput):
    prompt = f"Check answers for {input.num_items} questions related to '{input.topic}'. Provide correct answers and explanations."
    response = llm.invoke(prompt)
    return {"tool": "AnswerChecker", "checked_answers": response.content}

answer_checker_desc = "Checks answers for given questions and provides corrections and explanations."

# --- Tool 8: Topic Expansion ---
def topic_expansion_tool(input: ToolInput):
    prompt = f"Expand the topic '{input.topic}' into {input.num_items} related subtopics or areas of study."
    response = llm.invoke(prompt)
    return {"tool": "TopicExpansion", "expanded_topics": response.content}

topic_expansion_desc = "Generates related subtopics or deeper areas for a given topic."


# Export all tools in a dict for orchestration
TOOLS = {
    "quiz_generator": (ToolInput, quiz_generator_tool, quiz_generator_desc),
    "flashcard_maker": (ToolInput, flashcard_maker_tool, flashcard_maker_desc),
    "concept_explainer": (ToolInput, concept_explainer_tool, concept_explainer_desc),
    "summary_generator": (ToolInput, summary_generator_tool, summary_generator_desc),
    "example_creator": (ToolInput, example_creator_tool, example_creator_desc),
    "comparison_tool": (ToolInput, comparison_tool, comparison_desc),
    "answer_checker": (ToolInput, answer_checker_tool, answer_checker_desc),
    "topic_expansion": (ToolInput, topic_expansion_tool, topic_expansion_desc),
}

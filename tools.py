from pydantic import BaseModel, Field
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

class QuizGeneratorInput(BaseModel):
    topic: str = Field(..., description="The topic for quiz questions")
    difficulty: str = Field("easy", description="Difficulty level: easy, medium, hard")
    num_items: int = Field(3, description="Number of quiz questions to generate")


class FlashcardMakerInput(BaseModel):
    topic: str = Field(..., description="The topic for flashcards")
    num_items: int = Field(3, description="Number of flashcards to create")


class ConceptExplainerInput(BaseModel):
    topic: str = Field(..., description="The topic or concept to explain")
    difficulty: str = Field("easy", description="Difficulty level: easy, medium, hard")
    num_items: int = Field(3, description="Number of explanations to generate")
    descriptive: bool = Field(..., description="Whether explanation should be descriptive")


class SummaryGeneratorInput(BaseModel):
    topic: str = Field(..., description="The topic to summarize")
    num_items: int = Field(3, description="Number of key points to summarize in")


class ExampleCreatorInput(BaseModel):
    topic: str = Field(..., description="The concept/topic to create examples for")
    num_items: int = Field(3, description="Number of examples to generate")


class ComparisonToolInput(BaseModel):
    topic: str = Field(..., description="The topic to compare")
    num_items: int = Field(3, description="Number of comparison points")


class AnswerCheckerInput(BaseModel):
    topic: str = Field(..., description="The topic of the questions to check answers for")
    num_items: int = Field(3, description="Number of answers to check")


class TopicExpansionInput(BaseModel):
    topic: str = Field(..., description="The topic to expand")
    num_items: int = Field(3, description="Number of subtopics to generate")

class ChatToolInput(BaseModel):
    user_query: str = Field(..., description="The query or message from the user to chat with the AI")
    context: str = Field("", description="Optional context from previous conversation to maintain continuity")


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=api_key,
    temperature=0.7
)


def quiz_generator_tool(input: QuizGeneratorInput):
    prompt = f"Generate {input.num_items} {input.difficulty} quiz questions on the topic: {input.topic}."
    response = llm.invoke(prompt)
    return {"tool": "QuizGenerator", "questions": response.content}


def flashcard_maker_tool(input: FlashcardMakerInput):
    prompt = f"""
    Create {input.num_items} flashcards for the topic "{input.topic}".
    Each flashcard should have a 'question' and an 'answer'.
    Format output as a numbered list.
    """
    response = llm.invoke(prompt)
    return {
        "tool": "FlashcardMaker",
        "topic": input.topic,
        "flashcards": response.content 
    }


def concept_explainer_tool(input: ConceptExplainerInput):
    style = "descriptive" if input.descriptive else "concise"
    prompt = f"""
    Explain the topic "{input.topic}" in {input.num_items} explanations.
    Difficulty level: {input.difficulty}.
    Style: {style}.
    Provide clear and structured explanations.
    """
    response = llm.invoke(prompt)
    return {
        "tool": "ConceptExplainer",
        "topic": input.topic,
        "explanations": response.content,
        "style": style
    }


def summary_generator_tool(input: SummaryGeneratorInput):
    prompt = f"Summarize the topic '{input.topic}' in {input.num_items} key points."
    response = llm.invoke(prompt)
    return {"tool": "SummaryGenerator", "summary": response.content}


def example_creator_tool(input: ExampleCreatorInput):
    prompt = f"Create {input.num_items} examples to illustrate the concept '{input.topic}'."
    response = llm.invoke(prompt)
    return {"tool": "ExampleCreator", "examples": response.content}


def comparison_tool(input: ComparisonToolInput):
    prompt = f"Compare '{input.topic}' with related concepts in {input.num_items} points."
    response = llm.invoke(prompt)
    return {"tool": "ComparisonTool", "comparison": response.content}


def answer_checker_tool(input: AnswerCheckerInput):
    prompt = f"Check answers for {input.num_items} questions related to '{input.topic}'. Provide correct answers and explanations."
    response = llm.invoke(prompt)
    return {"tool": "AnswerChecker", "checked_answers": response.content}


def topic_expansion_tool(input: TopicExpansionInput):
    prompt = f"Expand the topic '{input.topic}' into {input.num_items} related subtopics or areas of study."
    response = llm.invoke(prompt)
    return {"tool": "TopicExpansion", "expanded_topics": response.content}

def chat_tool(input_data: ChatToolInput):
    """
    Generic chat tool for queries that don't match other specific tools.
    """
    prompt = f"""
        You are a helpful AI assistant. User asked: "{input_data.user_query}".
        {f"Context: {input_data.context}" if input_data.context else ""}
        Respond concisely and clearly.
        """
    response = llm.invoke(prompt).content.strip()
    return {"response": response}


TOOLS = {
    "quiz_generator": (QuizGeneratorInput, quiz_generator_tool, "Generates practice quiz questions based on topic and difficulty."),
    "flashcard_maker": (FlashcardMakerInput, flashcard_maker_tool, "Creates flashcards for a given topic."),
    "concept_explainer": (ConceptExplainerInput, concept_explainer_tool, "Explains a concept at a chosen difficulty level."),
    "summary_generator": (SummaryGeneratorInput, summary_generator_tool, "Generates a concise summary of a topic in key points."),
    "example_creator": (ExampleCreatorInput, example_creator_tool, "Creates practical examples for a given topic or concept."),
    "comparison_tool": (ComparisonToolInput, comparison_tool, "Compares a topic with related concepts."),
    "answer_checker": (AnswerCheckerInput, answer_checker_tool, "Checks answers and provides corrections."),
    "topic_expansion": (TopicExpansionInput, topic_expansion_tool, "Generates related subtopics for a given topic."),
    "chat_tool": (ChatToolInput, chat_tool, "General chat tool for miscellaneous queries."),
    
}

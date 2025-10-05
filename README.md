# 🧠 AI Educational Orchestrator

## 📘 Overview
**AI Educational Orchestrator** is an intelligent multi-tool system that autonomously connects user queries to the correct educational tool using **LangGraph**, **LangChain**, and **Google Gemini (Generative AI)**.  
It performs **intent classification**, **parameter extraction**, **schema validation**, and **tool execution** — creating a self-directed, LLM-powered orchestration pipeline for educational tasks.

---

## ⚙️ System Architecture

The system is designed as a **LangGraph workflow**, where each node performs a cognitive step in the orchestration pipeline.

### 🧩 Graph Nodes

| Node | Function |
|------|-----------|
| `classify_tool` | Uses LLM reasoning to identify which tool best fits the user’s intent. |
| `extract_parameters` | Extracts structured parameters based on the tool’s Pydantic input schema. |
| `ask_missing_params` | Detects and requests missing parameters interactively using LLM guidance. |
| `call_tool` | Executes the appropriate tool function with validated parameters and returns the output. |

**Graph Engine:**  
- Built with `StateGraph(AgentState)`  
- Uses `InMemorySaver()` for checkpoint memory  
- Dynamically compiles workflow via `graph.compile()`

---

## 🧠 Functional Tools

| Tool | Purpose | Key Parameters |
|------|----------|----------------|
| **Quiz Generator** | Generates quiz questions based on topic and difficulty. | `topic`, `difficulty`, `num_items` |
| **Flashcard Maker** | Creates flashcards with Q/A pairs for revision. | `topic`, `num_items` |
| **Concept Explainer** | Explains concepts at various difficulty levels; can be descriptive or concise. | `topic`, `difficulty`, `num_items`, `descriptive` |
| **Summary Generator** | Summarizes any topic into concise key points. | `topic`, `num_items` |
| **Example Creator** | Generates examples that illustrate a concept. | `topic`, `num_items` |
| **Comparison Tool** | Compares a topic with related concepts or ideas. | `topic`, `num_items` |
| **Answer Checker** | Evaluates user answers and provides correct explanations. | `topic`, `num_items` |
| **Topic Expansion** | Expands a topic into related subtopics or areas of study. | `topic`, `num_items` |
| **Chat Tool** | Generic chat interface for open-ended queries. | `user_query`, `context` |

Each tool is registered in the **TOOLS registry**, defined as:
```python
TOOLS = {
    "quiz_generator": (QuizGeneratorInput, quiz_generator_tool, "Generates quizzes"),
    ...
}
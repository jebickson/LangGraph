from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

load_dotenv()

class State(TypedDict):
    question: str
    answer: str

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

def answer_question(state: State):
    response = llm.invoke(state["question"])
    return {"answer": response.content}

graph_builder = StateGraph(State)
graph_builder.add_node("answer", answer_question)
graph_builder.add_edge(START, "answer")
graph_builder.add_edge("answer", END)

app = graph_builder.compile()

result = app.invoke({
    "question": "Explain LangGraph in simple words.",
    "answer": ""
})

print(result["answer"])
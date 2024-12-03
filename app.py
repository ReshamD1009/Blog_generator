from flask import Flask, render_template, request
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

class BlogState(BaseModel):
    topic: str = Field(default="")
    outline: str = Field(default="")
    content: str = Field(default="")
    blog_post: str = Field(default="")

def get_llm():
    return ChatGroq(model="llama3-8b-8192")

def outline_blog(state: BlogState):
    llm = get_llm()
    outline_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that outlines blog posts."),
            ("user", "Outline a blog post about {topic} by taking the topic given."),
        ]
    )
    chain = outline_prompt | llm
    res = chain.invoke({"topic": state.topic})
    state.outline = res.content
    print(f"Outline generated: {state.outline}")
    return state

def content_writer(state: BlogState):
    search = TavilySearchResults()
    tools = [search]

    input_data = {
        "topic": state.topic,
        "outline": state.outline,
    }
    print(f"Input data for tools: {input_data}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant that writes blog posts based on the given topic and outline.Use the Tavily_Search tool to gather relevant information to improve accuracy and detail.
                         {agent_scratchpad}"""),
            ("user", "Topic: {topic}"),
            ("user", "Outline: {outline}"),
            ("user", "Using the provided tools, gather information to create a detailed blog post."),
        ]
    )

    llm = get_llm()
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    try:
        response = agent_executor.invoke(input_data)
        state.content = response["output"]
    except Exception as e:
        print(f"Error during agent execution: {e}")
        state.content = "Failed to generate content due to tool error."

    print(f"Content generated: {state.content}")
    return state



def formatter_agent(state: BlogState):
    llm = get_llm()
    formatter_agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert in formatting blog posts."),
            ("user", "Outline: {outline}"),
            ("user", "Content: {content}"),
        ]
    )
    chain = formatter_agent_prompt | llm
    res = chain.invoke({"outline": state.outline, "content": state.content})
    state.blog_post = res.content
    print(f"Formatted blog post generated: {state.blog_post}")
    return state

workflow = StateGraph(BlogState)

workflow.add_node("outline_blog", outline_blog)
workflow.add_node("content_writer", content_writer)
workflow.add_node("formatter_agent", formatter_agent)

workflow.add_edge(START, "outline_blog")
workflow.add_edge("outline_blog", "content_writer")
workflow.add_edge("content_writer", "formatter_agent")
workflow.add_edge("formatter_agent", END)

compiled_app = workflow.compile()

@app.route('/', methods=['GET', 'POST'])
def index():
    obj = BlogState()  

    if request.method == 'POST':
        topic = request.form['topic']
        obj.topic = topic
        obj.outline = ""
        obj.content = ""
        obj.blog_post = ""

        try:
            result = compiled_app.invoke(obj) 
            print(f"State after workflow: {result}")
            obj = result
        except Exception as e:
            print(f"Error during workflow execution: {e}")

        return render_template('index.html', obj=obj)

    return render_template('index.html', obj=obj)

if __name__ == '__main__':
    app.run(debug=True)

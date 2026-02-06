import os 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import gradio as gr

os.environ["OPENAI_API_KEY"] = "sk-test"

llm = ChatOpenAI(
    model="qwen3-8b-fp8",
    temperature=0.2,
    base_url="http://100.67.127.53:8000/v1",
)

def box_chat(message, history):
    messages = []

    for h in history:
        if isinstance(h, (list, tuple)) and len(h) == 2:
            user, bot = h
            if user:
                messages.append(HumanMessage(content=user))
            if bot:
                messages.append(AIMessage(content=bot))

    messages.append(HumanMessage(content=message))
    response = llm.invoke(messages)
    return response.content

demo = gr.ChatInterface(
    fn=box_chat,
    title="LangChain Chat Demo",
    description="Chat with me",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7960)
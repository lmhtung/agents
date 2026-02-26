from email.mime import message

from langchain_core.messages import HumanMessage, AIMessage

class ChatInterface:
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
    def chat(self, message, history):

        if not self.rag_system.agent_graph:
            return "⚠️ System not initialized!"
            
        try:
            result = self.rag_system.agent_graph.invoke(
                {"messages": [HumanMessage(content=message.strip())]},
                self.rag_system.get_config()
            )
            return result["messages"][-1].content
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    # def chat(self, message, history):
    #     if not self.rag_system.agent_graph:
    #         yield "⚠️ System not initialized!"
    #         return

    #     try:
    #         # Sử dụng stream để nhận phản hồi từng phần
    #         input_data = {"messages": [HumanMessage(content=message.strip())]}
    #         config = self.rag_system.get_config()
        
    #         full_response = ""
    #         # stream_mode="values" sẽ trả về state sau mỗi bước của node
    #         for chunk in self.rag_system.agent_graph.stream(input_data, config, stream_mode="values"):
    #             if "messages" in chunk:
    #                 last_msg = chunk["messages"][-1]
    #                 if isinstance(last_msg, AIMessage) and last_msg.content:
    #                     full_response = last_msg.content
    #                     yield full_response # Trả về nội dung mới nhất cho Gradio

    #     except Exception as e:
    #         yield f"❌ Error: {str(e)}"

    def clear_session(self):
        self.rag_system.reset_thread()
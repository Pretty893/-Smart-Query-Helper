from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from .query_router import QueryRouter
from .prompt_selector import PromptSelector
from .post_processor import PostProcessor
from .logger import get_logger, log_execution, RequestContext, generate_request_id
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser


class AgentState(TypedDict):
    query: str
    request_id: str
    routing_result: Optional[Dict[str, Any]]
    retrieved_docs: Optional[List[Any]]
    answer: Optional[str]
    post_processed_answer: Optional[str]
    history: Optional[List[Dict[str, str]]]
    error: Optional[str]


class AgentGraph:
    def __init__(self):
        self.logger = get_logger("agent_graph")
        self.router = QueryRouter()
        self.prompt_selector = PromptSelector()
        self.post_processor = PostProcessor()
        self.chat_model = ChatTongyi(model="qwen-turbo")
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("initialize", self._initialize)
        workflow.add_node("route", self._route)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("post_process", self._post_process)

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "route")
        workflow.add_edge("route", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "post_process")
        workflow.add_edge("post_process", END)

        return workflow.compile()

    @log_execution("agent_graph")
    def _initialize(self, state: AgentState) -> AgentState:
        request_id = generate_request_id()
        RequestContext.set_request_id(request_id)
        
        self.logger.info(
            "初始化对话",
            extra={"request_id": request_id, "query": state["query"][:50]},
        )
        
        return {
            **state,
            "request_id": request_id,
            "routing_result": None,
            "retrieved_docs": None,
            "answer": None,
            "post_processed_answer": None,
            "error": None,
        }

    @log_execution("agent_graph")
    def _route(self, state: AgentState) -> AgentState:
        try:
            routing_result = self.router.route(state["query"])
            
            self.logger.info(
                "路由决策完成",
                extra={
                    "request_id": state["request_id"],
                    "intent_type": routing_result.get("intent_type"),
                    "retriever_type": routing_result.get("retriever_type"),
                    "use_tool": routing_result.get("use_tool"),
                },
            )
            
            return {**state, "routing_result": routing_result}
        except Exception as e:
            self.logger.error(
                "路由失败",
                extra={"request_id": state["request_id"], "error": str(e)},
            )
            return {**state, "error": f"路由失败: {str(e)}"}

    @log_execution("agent_graph")
    def _retrieve(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state

        try:
            routing_result = state["routing_result"]
            
            if routing_result.get("use_tool") and routing_result.get("tool_name"):
                self.logger.info(
                    "调用工具",
                    extra={
                        "request_id": state["request_id"],
                        "tool_name": routing_result["tool_name"],
                    },
                )
                
                tool_result = self.router.call_tool(
                    routing_result["tool_name"],
                    **routing_result.get("tool_params", {}),
                )
                
                return {**state, "answer": str(tool_result), "retrieved_docs": []}

            retriever = self.router.get_retriever(routing_result["retriever_type"])
            docs = retriever.retrieve(state["query"])
            
            self.logger.info(
                "检索完成",
                extra={
                    "request_id": state["request_id"],
                    "doc_count": len(docs),
                    "retriever_type": routing_result["retriever_type"],
                },
            )
            
            return {**state, "retrieved_docs": docs}
        except Exception as e:
            self.logger.error(
                "检索失败",
                extra={"request_id": state["request_id"], "error": str(e)},
            )
            return {**state, "error": f"检索失败: {str(e)}"}

    @log_execution("agent_graph")
    def _generate(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state

        if state.get("answer"):
            return state

        try:
            docs = state["retrieved_docs"]
            
            if not docs:
                return {**state, "answer": "未找到相关文档，请尝试其他问题"}

            context = "\n\n".join([doc.page_content for doc in docs])
            routing_result = state["routing_result"]
            
            prompt_template = self.prompt_selector.select(routing_result["prompt_type"])
            
            self.logger.info(
                "生成回答",
                extra={
                    "request_id": state["request_id"],
                    "prompt_type": routing_result["prompt_type"],
                    "context_length": len(context),
                },
            )
            
            chain = prompt_template | self.chat_model | StrOutputParser()
            answer = chain.invoke({"question": state["query"], "context": context})
            
            return {**state, "answer": answer}
        except Exception as e:
            self.logger.error(
                "生成失败",
                extra={"request_id": state["request_id"], "error": str(e)},
            )
            return {**state, "error": f"生成失败: {str(e)}"}

    @log_execution("agent_graph")
    def _post_process(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state

        answer = state["answer"]
        
        if not answer:
            return {**state, "post_processed_answer": "未生成回答"}

        self.logger.info(
            "后处理完成",
            extra={"request_id": state["request_id"], "answer_length": len(answer)},
        )

        return {**state, "post_processed_answer": answer}

    def run(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        RequestContext.clear()
        
        inputs = {
            "query": query,
            "history": history or [],
        }
        
        result = self.graph.invoke(inputs)
        
        RequestContext.clear()
        
        return result

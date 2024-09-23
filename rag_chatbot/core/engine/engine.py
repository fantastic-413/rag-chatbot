from dataclasses import dataclass, field
from llama_index.core.chat_engine import CondensePlusContextChatEngine, SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import BaseNode
from typing import List, Optional
from .retriever import LocalRetriever
from ...setting import RAGSettings

from llama_index.core.callbacks import trace_method
from llama_index.core.types import Thread
from llama_index.core.schema import MetadataMode
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import BaseMemory
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatErrorEvent,
    StreamChatEndEvent,
    StreamChatStartEvent,
    StreamChatDeltaReceivedEvent,
)
import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

def is_function(message: ChatMessage) -> bool:
    """Utility for ChatMessage responses from OpenAI models."""
    return "tool_calls" in message.additional_kwargs

class LocalChatEngine:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal"
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._retriever = LocalRetriever(self._setting)
        self._host = host

    def set_engine(
        self,
        llm: LLM,
        nodes: List[BaseNode],
        language: str = "eng",
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:

        # Normal chat engine
        if len(nodes) == 0:
            return SimpleChatEngine.from_defaults(
                llm=llm,
                memory=ChatMemoryBuffer(
                    token_limit=self._setting.ollama.chat_token_limit
                )
            )

        # Chat engine with documents
        retriever = self._retriever.get_retrievers(
            llm=llm,
            language=language,
            nodes=nodes
        )
        return CustomCondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=ChatMemoryBuffer(
                token_limit=self._setting.ollama.chat_token_limit
            )
        )

class CustomCondensePlusContextChatEngine(CondensePlusContextChatEngine):
    
    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        chat_messages, context_source, context_nodes = self._run_c3(
            message, chat_history
        )
        
        # 从 context_nodes 中提取 chunk 及其 filename 信息
        retrieved_chunks = [
            {
                "text": node.text,  # 你的 chunk 文本
                "filename": node.metadata.get("file_name", "Unknown")  # 假设 filename 存在于 metadata 中
            }
            for node in context_nodes
        ]

        # pass the context, system prompt and user message as chat to LLM to generate a response
        chat_response = CustomStreamingAgentChatResponse(
            chat_stream=self._llm.stream_chat(chat_messages),
            sources=[context_source],
            source_nodes=context_nodes,
            retrieved_chunks=retrieved_chunks  # 添加检索到的 chunk 和文件名信息
        )
        thread = Thread(
            target=chat_response.write_response_to_history, args=(self._memory,)
        )
        thread.start()

        return chat_response

@dataclass
class CustomStreamingAgentChatResponse(StreamingAgentChatResponse):
    # 新增属性：检索到的 chunks 及其对应文件名
    retrieved_chunks: List[dict] = field(default_factory=list)
    
    @dispatcher.span
    def write_response_to_history(
        self,
        memory: BaseMemory,
        on_stream_end_fn: Optional[callable] = None,
    ) -> None:
        if self.chat_stream is None:
            raise ValueError(
                "chat_stream is None. Cannot write to history without chat_stream."
            )

        # try/except to prevent hanging on error
        dispatcher.event(StreamChatStartEvent())
        try:
            final_text = ""
            for chat in self.chat_stream:
                self.is_function = is_function(chat.message)
                if chat.delta:
                    dispatcher.event(
                        StreamChatDeltaReceivedEvent(
                            delta=chat.delta,
                        )
                    )
                    self.put_in_queue(chat.delta)
                final_text += chat.delta or ""
                
            # 在结束循环后将 chunks 信息加入队列
            if self.retrieved_chunks:
                self.put_in_queue("\n\n**Retrieved chunks and their source file names:**\n")
                for chunk in self.retrieved_chunks:
                    chunk_info = f"\nChunk: {chunk['text']}\nFilename: {chunk['filename']}\n"
                    self.put_in_queue(chunk_info)  # 加入检索到的 chunk 和文件名信息
                
            if self.is_function is not None:  # if loop has gone through iteration
                # NOTE: this is to handle the special case where we consume some of the
                # chat stream, but not all of it (e.g. in react agent)
                chat.message.content = final_text.strip()  # final message
                memory.put(chat.message)
        except Exception as e:
            dispatcher.event(StreamChatErrorEvent(exception=e))
            self.exception = e

            # This act as is_done events for any consumers waiting
            self.is_function_not_none_thread_event.set()

            # force the queue reader to see the exception
            self.put_in_queue("")
            raise
        dispatcher.event(StreamChatEndEvent())

        self.is_done = True

        # This act as is_done events for any consumers waiting
        self.is_function_not_none_thread_event.set()
        if on_stream_end_fn is not None and not self.is_function:
            on_stream_end_fn()
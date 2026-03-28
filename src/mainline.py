from __future__ import annotations

from pathlib import Path
from typing import Annotated
from typing import Any
from typing import TypedDict

from langchain.prompts import PromptTemplate
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from PIL import Image

from src.base.layout import Layout
from src.tools.models.layout_extractor import global_layout_extractor
from src.tools.models.llm_model import global_llm_model
from src.tools.pdf_reader import PDFReader


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]


class QuestionResolverNode:
    def __init__(self, layout: Layout, model: Any) -> None:
        """
        A question resolver node that resolves the question based on the layout.

        :param question: The question to resolve.
        :param layout: The layout of the document.
        """

        self.layout = layout
        self.model = model

    def resolve(self, state: AgentState) -> AgentState:
        """
        Resolve the question based on the layout.

        :return: The resolved question.
        """

        question = state["messages"][-1]["content"]
        resolved_question = self.model.ask_for(question, layout=self.layout)
        state["messages"].append(
            {"role": "assistant", "content": resolved_question},
        )
        return state

    def __call__(self, state: AgentState) -> AgentState:
        """
        Call the question resolver node.

        :param state: The agent state.
        :return: The agent state.
        """

        return self.resolve(state)


class ChatChainModel:
    def __init__(
        self,
        document_path: Path,
        crop: tuple[int, int, int, int, int] | None = None,
        image_mode: bool = False,
    ):
        """
        A chat chain model that uses a layout to generate a conversation.

        :param document_path: The path to the document.
        :param crop: The crop for the document (for specific regions).
        :param image_mode: Select the mode of the document.
        """

        self.layout = self._build_document(document_path, crop, image_mode)
        self.chain = self._build_chain()

    def _build_document(
        self,
        document_path: Path,
        crop: tuple[int, int, int, int, int] | None = None,
        image_mode: bool = False,
    ) -> Layout:
        """
        Make layout of the image.

        :param document_path: The input image.
        :param crop: The crop for the document (for specific regions).
        :param image_mode: Select the mode of the document.
        :return: None
        """

        self.images: list[Image.Image] | None = None
        if not image_mode:
            self.images = self._read_document(document_path)
        else:
            self.images = [self._read_image(document_path)]

        if crop is not None and self.images is not None:
            self.images = [self.images[crop[0]].crop(crop[1:])]

        layout = Layout.from_dict(
            {
                "blocks": [
                    box.update({"page_number": page_number}) or box
                    for page_number, image in enumerate(self.images)
                    for box in global_layout_extractor.make_layout(image)
                ],
                "page_count": len(self.images),  # type: ignore
            },
        )
        return layout

    def _read_document(
        self,
        document_path: Path,
    ) -> list[Image.Image]:
        """
        Read the document.

        :param document_path: The path to the document.
        :return: None
        """

        self.pdf_reader = PDFReader(document_path)
        return self.pdf_reader.images

    def _read_image(
        self,
        document_path: Path,
    ) -> Image.Image:
        """
        Read the image.

        :param document_path: The path to the document.
        :return: None
        """

        return Image.open(document_path)

    def _build_chain(self) -> Any:
        """
        Build the chain for the chat model.

        :return: The compiled chain for the chat model.
        """

        chain = StateGraph(AgentState)

        chain.add_node(
            "entry_node",
            QuestionResolverNode(self.layout, global_llm_model),
        )

        chain.add_edge(START, "entry_node")
        chain.add_edge("entry_node", END)

        return chain.compile()

    def get_layout(self) -> Layout:
        """
        Get the layout of the document.

        :return: The layout of the document.
        """

        return self.layout

    def handle_prompt(self, prompt: str) -> str:
        """
        Handle the prompt and return the response.

        :param prompt: The prompt to handle.
        :return: The response.
        """

        # Create initial state with the user prompt
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Invoke the compiled chain with the initial state
        result = self.chain.invoke(initial_state)
        
        # Extract the assistant's response from the final state
        if result and "messages" in result and len(result["messages"]) > 0:
            # Get the last message which should be the assistant's response
            last_message = result["messages"][-1]
            if last_message.get("role") == "assistant":
                return str(last_message.get("content", ""))
        
        # Fallback if no proper response found
        return "No response generated."


class ExtractionChainModel:
    def __init__(
        self,
        layout: Layout,
    ):
        """
        An extraction chain model that uses a layout to extract information from the document.

        :param layout: The layout of the document.
        """

        self.chain = self._build_chain()
        self.layout = layout

    def _build_chain(self) -> Any:
        """
        Build the chain for the extraction model.

        :return: The chain for the extraction model.
        """

        # The ExtractionChainModel appears to be incomplete/placeholder code
        # For now, return a simple compiled chain that does basic extraction
        chain = StateGraph(AgentState)
        
        chain.add_node(
            "extraction_node",
            QuestionResolverNode(self.layout, global_llm_model),
        )

        chain.add_edge(START, "extraction_node")
        chain.add_edge("extraction_node", END)

        return chain.compile()

    def extract_information(self) -> str:
        """
        Extract information based on the layout.

        :return: The extracted information.
        """

        # Create a default extraction prompt
        extraction_prompt = "Extract and summarize key information from this document."
        
        # Create initial state with the extraction prompt
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": extraction_prompt}]
        }
        
        # Invoke the compiled chain with the initial state
        result = self.chain.invoke(initial_state)
        
        # Extract the assistant's response from the final state
        if result and "messages" in result and len(result["messages"]) > 0:
            # Get the last message which should be the assistant's response
            last_message = result["messages"][-1]
            if last_message.get("role") == "assistant":
                return str(last_message.get("content", ""))
        
        # Fallback if no proper response found
        return "No information extracted."

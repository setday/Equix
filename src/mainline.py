from __future__ import annotations

from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph

from src.base.layout import Layout


class ChatChainModel:
    def __init__(
        self,
        layout: Layout,
    ):
        """
        A chat chain model that uses a layout to generate a conversation.

        :param layout: The layout of the document.
        """

        self.chain = self._build_chain()
        self.layout = layout

    def _build_chain(self) -> StateGraph:
        """
        Build the chain for the chat model.

        :return: The chain for the chat model.
        """

        chain = StateGraph()

        chain.add_state("start")
        chain.add_state("end")

        chain.add_edge(
            "start",
            "end",
            PromptTemplate("Ask for information about document", layout=self.layout),
        )

        return chain

    def generate_conversation(self) -> str:
        """
        Generate a conversation based on the layout.

        :return: The generated conversation.
        """

        result = self.chain.generate_conversation()

        assert isinstance(result, str)

        return result


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

    def _build_chain(self) -> StateGraph:
        """
        Build the chain for the extraction model.

        :return: The chain for the extraction model.
        """

        chain = StateGraph()

        chain.add_state("start")
        chain.add_state("end")

        chain.add_edge(
            "start",
            "end",
            PromptTemplate("Ask for information about document", layout=self.layout),
        )

        return chain

    def extract_information(self) -> str:
        """
        Extract information based on the layout.

        :return: The extracted information.
        """

        result = self.chain.generate_conversation()

        assert isinstance(result, str)

        return result

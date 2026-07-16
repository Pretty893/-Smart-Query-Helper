from typing import Dict, Any, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

from .config_manager import get_config
config = get_config()
from .skill_registry import skill_metadata


@skill_metadata(
    name="summary_processor",
    description="生成文本摘要，将长回答压缩为100-200字的简洁摘要，保留核心结论和关键步骤",
    skill_type="post_processor",
    parameters={
        "answer": {
            "type": "string",
            "description": "需要摘要的原始回答文本",
            "required": True
        },
        "max_length": {
            "type": "integer",
            "description": "摘要最大长度",
            "required": False,
            "default": 200
        }
    },
    returns={
        "type": "string",
        "description": "压缩后的摘要文本"
    },
    tags=["摘要", "压缩", "内容提炼"],
    examples=[
        "将回答压缩成200字以内",
        "提炼核心观点",
        "生成简短摘要"
    ],
    version="1.0.0"
)
class SummaryProcessor:
    def __init__(self):
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个文本摘要专家。请将以下回答压缩成简洁的摘要，保留关键信息。

要求：
1. 长度控制在 100-200 字
2. 保留核心结论和关键步骤
3. 语言流畅、自然
""",
                ),
                ("human", "原始回答：\n{answer}"),
            ]
        )

    def process(self, answer: str, max_length: int = 200) -> str:
        chain = self.prompt | self.chat_model | StrOutputParser()
        result = chain.invoke({"answer": answer})
        return result[:max_length]


@skill_metadata(
    name="translation_processor",
    description="翻译文本，支持多种目标语言，保持专业术语的准确性和格式结构",
    skill_type="post_processor",
    parameters={
        "answer": {
            "type": "string",
            "description": "需要翻译的原始文本",
            "required": True
        },
        "target_language": {
            "type": "string",
            "description": "目标语言，如English、Japanese",
            "required": False,
            "default": "English"
        }
    },
    returns={
        "type": "string",
        "description": "翻译后的文本"
    },
    tags=["翻译", "多语言", "本地化"],
    examples=[
        "翻译成英文",
        "翻译成日文",
        "英文转中文"
    ],
    version="1.0.0"
)
class TranslationProcessor:
    def __init__(self):
        self.chat_model = ChatTongyi(model=config.chat_model_name)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业翻译专家。请将以下中文回答翻译成目标语言。

要求：
1. 保持专业术语的准确性
2. 保持格式和结构
3. 翻译自然流畅
""",
                ),
                ("human", "目标语言：{target_language}\n\n原始文本：\n{answer}"),
            ]
        )

    def process(self, answer: str, target_language: str = "English") -> str:
        chain = self.prompt | self.chat_model | StrOutputParser()
        result = chain.invoke({"answer": answer, "target_language": target_language})
        return result


@skill_metadata(
    name="format_processor",
    description="格式转换，支持Markdown转HTML、Markdown转纯文本等格式转换",
    skill_type="post_processor",
    parameters={
        "answer": {
            "type": "string",
            "description": "需要转换格式的文本",
            "required": True
        },
        "format_type": {
            "type": "string",
            "description": "目标格式：markdown/html/plain_text",
            "required": False,
            "default": "markdown"
        }
    },
    returns={
        "type": "string",
        "description": "转换格式后的文本"
    },
    tags=["格式转换", "HTML", "纯文本"],
    examples=[
        "转换为HTML格式",
        "去掉所有格式",
        "转为纯文本"
    ],
    version="1.0.0"
)
class FormatProcessor:
    def process(self, answer: str, format_type: str = "markdown") -> str:
        if format_type == "html":
            return self._convert_to_html(answer)
        elif format_type == "plain_text":
            return self._convert_to_plain_text(answer)
        else:
            return answer

    def _convert_to_html(self, markdown_text: str) -> str:
        html = markdown_text
        html = html.replace("### ", "<h3>").replace("\n### ", "</h3>\n<h3>")
        html = html.replace("## ", "<h2>").replace("\n## ", "</h2>\n<h2>")
        html = html.replace("# ", "<h1>").replace("\n# ", "</h1>\n<h1>")
        html = html.replace("**", "<strong>").replace("</strong><strong>", "</strong><strong>")
        html = html.replace("\n", "<br>")
        return f"<div>{html}</div>"

    def _convert_to_plain_text(self, markdown_text: str) -> str:
        text = markdown_text
        text = text.replace("#", "")
        text = text.replace("**", "")
        text = text.replace("*", "")
        text = text.replace("`", "")
        return text.strip()


@skill_metadata(
    name="highlight_processor",
    description="关键词高亮，将指定关键词在文本中加粗显示，突出重点信息",
    skill_type="post_processor",
    parameters={
        "answer": {
            "type": "string",
            "description": "需要处理的文本",
            "required": True
        },
        "keywords": {
            "type": "list",
            "description": "需要高亮的关键词列表",
            "required": True
        }
    },
    returns={
        "type": "string",
        "description": "关键词高亮后的文本"
    },
    tags=["高亮", "关键词", "重点标记"],
    examples=[
        "高亮关键词'年假'",
        "标记重点信息",
        "突出关键条款"
    ],
    version="1.0.0"
)
class HighlightProcessor:
    def process(self, answer: str, keywords: list) -> str:
        highlighted = answer
        for keyword in keywords:
            highlighted = highlighted.replace(keyword, f"**{keyword}**")
        return highlighted


class PostProcessor:
    def __init__(self):
        self.processors = {
            "summary": SummaryProcessor(),
            "translate": TranslationProcessor(),
            "format": FormatProcessor(),
            "highlight": HighlightProcessor(),
        }

    def generate_summary(self, answer: str, max_length: int = 200) -> str:
        return self.processors["summary"].process(answer, max_length)

    def translate(self, answer: str, target_language: str = "English") -> str:
        return self.processors["translate"].process(answer, target_language)

    def format_answer(self, answer: str, format_type: str = "markdown") -> str:
        return self.processors["format"].process(answer, format_type)

    def add_highlights(self, answer: str, keywords: list) -> str:
        return self.processors["highlight"].process(answer, keywords)

    def list_processors(self) -> list:
        return [processor.METADATA for processor in self.processors.values()]

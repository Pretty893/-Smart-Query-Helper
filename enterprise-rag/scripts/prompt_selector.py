from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config_manager import get_config
config = get_config()
from .skill_registry import skill_metadata


@skill_metadata(
    name="policy_qa_prompt",
    description="制度问答模板，适合查询具体政策规定，如'病假可以请几天'，要求准确引用条款",
    skill_type="prompt",
    parameters={
        "context": {
            "type": "string",
            "description": "参考材料，包含制度条款内容",
            "required": True
        },
        "question": {
            "type": "string",
            "description": "用户问题，关于制度规定的查询",
            "required": True
        },
        "history": {
            "type": "list",
            "description": "对话历史",
            "required": False
        }
    },
    returns={
        "type": "string",
        "description": "包含最终回答、依据条款、风险提示的结构化文本"
    },
    tags=["制度问答", "政策规定", "条款查询"],
    examples=[
        "年假天数规定",
        "加班费标准",
        "试用期规定"
    ],
    version="1.0.0"
)
class PolicyQAPrompt:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是企业制度政策专家。请依据参考材料回答用户关于制度规定的问题。

要求：
1. 严格依据参考材料，不得编造内容
2. 回答要准确、简洁、专业
3. 如果材料中有具体条款编号，请引用条款编号
4. 如果材料不足，请明确说明"未找到明确依据"

输出格式：
### 最终回答
[你的回答]

### 依据条款
[引用的具体条款内容]

### 风险提示
[如有适用]
""",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "参考材料：\n{context}\n\n用户问题：{question}",
            ),
        ]
    )


@skill_metadata(
    name="process_guide_prompt",
    description="流程指引模板，适合查询办理流程，如'怎么报销差旅费'，要求分解为清晰步骤",
    skill_type="prompt",
    parameters={
        "context": {
            "type": "string",
            "description": "参考材料，包含流程说明",
            "required": True
        },
        "question": {
            "type": "string",
            "description": "用户问题，关于办理流程的查询",
            "required": True
        },
        "history": {
            "type": "list",
            "description": "对话历史",
            "required": False
        }
    },
    returns={
        "type": "string",
        "description": "包含操作步骤、关键节点、所需材料的结构化文本"
    },
    tags=["流程指引", "步骤说明", "办理流程"],
    examples=[
        "报销流程",
        "请假步骤",
        "入职手续"
    ],
    version="1.0.0"
)
class ProcessGuidePrompt:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是企业流程指引专家。请依据参考材料为用户提供清晰的步骤指引。

要求：
1. 将流程分解为清晰的步骤，编号列出
2. 每个步骤说明具体操作和注意事项
3. 标注关键节点和审批人
4. 如果材料不足，请明确说明

输出格式：
### 操作步骤
1. [步骤一]
2. [步骤二]
...

### 关键节点
[需要特别注意的环节]

### 所需材料
[如有适用]
""",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "参考材料：\n{context}\n\n用户问题：{question}",
            ),
        ]
    )


@skill_metadata(
    name="material_list_prompt",
    description="材料清单模板，适合查询所需材料，如'报销需要提交什么'，要求分类整理",
    skill_type="prompt",
    parameters={
        "context": {
            "type": "string",
            "description": "参考材料，包含材料要求",
            "required": True
        },
        "question": {
            "type": "string",
            "description": "用户问题，关于材料清单的查询",
            "required": True
        },
        "history": {
            "type": "list",
            "description": "对话历史",
            "required": False
        }
    },
    returns={
        "type": "string",
        "description": "包含必需材料表格和注意事项的结构化文本"
    },
    tags=["材料清单", "表格格式", "材料要求"],
    examples=[
        "报销需要什么材料",
        "入职要带什么",
        "申请需要提交什么"
    ],
    version="1.0.0"
)
class MaterialListPrompt:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是企业材料清单专家。请依据参考材料列出用户所需的所有材料。

要求：
1. 列出所有必需材料，分类整理
2. 说明每份材料的要求和格式
3. 标注材料的获取途径（如有）
4. 如果材料不足，请明确说明

输出格式：
### 必需材料
| 材料名称 | 要求 | 获取途径 |
|---------|------|---------|
| [材料1] | [要求] | [途径] |
| [材料2] | [要求] | [途径] |

### 注意事项
[填写时的注意事项]
""",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "参考材料：\n{context}\n\n用户问题：{question}",
            ),
        ]
    )


@skill_metadata(
    name="notice_summary_prompt",
    description="通知总结模板，适合要求总结通知，如'这份通知的重点是什么'，要求提取核心要点",
    skill_type="prompt",
    parameters={
        "context": {
            "type": "string",
            "description": "参考材料，包含通知内容",
            "required": True
        },
        "question": {
            "type": "string",
            "description": "用户问题，关于通知总结的请求",
            "required": True
        },
        "history": {
            "type": "list",
            "description": "对话历史",
            "required": False
        }
    },
    returns={
        "type": "string",
        "description": "包含核心内容、关键要素、重点提醒的结构化文本"
    },
    tags=["通知总结", "要点提取", "内容概括"],
    examples=[
        "通知重点是什么",
        "概括一下这份通知",
        "提炼核心内容"
    ],
    version="1.0.0"
)
class NoticeSummaryPrompt:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是企业通知总结专家。请依据参考材料为用户提炼通知的核心要点。

要求：
1. 提取通知的核心内容和关键信息
2. 标注生效日期、适用范围、责任人等关键要素
3. 总结重点注意事项
4. 如果材料不足，请明确说明

输出格式：
### 核心内容
[通知的主要事项]

### 关键要素
- 生效日期：[日期]
- 适用范围：[范围]
- 责任部门：[部门]

### 重点提醒
[需要特别关注的内容]
""",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "参考材料：\n{context}\n\n用户问题：{question}",
            ),
        ]
    )


@skill_metadata(
    name="complex_prompt",
    description="复杂问题分析模板，适合复杂问题，如'从入职到转正的完整流程'，要求多步推理",
    skill_type="prompt",
    parameters={
        "context": {
            "type": "string",
            "description": "参考材料，包含多个制度或流程",
            "required": True
        },
        "question": {
            "type": "string",
            "description": "用户问题，复杂问题或需要多步推理",
            "required": True
        },
        "history": {
            "type": "list",
            "description": "对话历史",
            "required": False
        }
    },
    returns={
        "type": "string",
        "description": "包含问题拆解、分步解答、综合结论的结构化文本"
    },
    tags=["复杂问题", "多步推理", "综合分析"],
    examples=[
        "从入职到转正的完整流程",
        "请5天年假需要什么手续",
        "报销差旅费有什么限制"
    ],
    version="1.0.0"
)
class ComplexPrompt:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是企业复杂问题分析专家。请依据参考材料分析用户的复杂问题，进行多步推理。

要求：
1. 将复杂问题分解为多个子问题
2. 逐一分析每个子问题的答案
3. 综合各子问题的答案得出最终结论
4. 如果材料不足，请明确说明

输出格式：
### 问题拆解
1. [子问题一]
2. [子问题二]
...

### 分步解答
1. [子问题一的答案]
2. [子问题二的答案]
...

### 综合结论
[最终结论]
""",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "参考材料：\n{context}\n\n用户问题：{question}",
            ),
        ]
    )


class PromptSelector:
    def __init__(self):
        self.templates = {
            "policy_qa": PolicyQAPrompt(),
            "process_guide": ProcessGuidePrompt(),
            "material_list": MaterialListPrompt(),
            "notice_summary": NoticeSummaryPrompt(),
            "complex": ComplexPrompt(),
        }

    def select(self, intent_type: str) -> ChatPromptTemplate:
        template_obj = self.templates.get(intent_type)
        return template_obj.template if template_obj else self.templates["policy_qa"].template

    def get_template_metadata(self, intent_type: str = None) -> Dict[str, Any]:
        if intent_type is None:
            return {key: template.METADATA for key, template in self.templates.items()}
        template = self.templates.get(intent_type)
        return template.METADATA if template else {}

    def get_template_names(self) -> Dict[str, str]:
        return {
            key: template.METADATA["name"] for key, template in self.templates.items()
        }

    def list_templates(self) -> list:
        return [
            {"type": key, "name": template.METADATA["name"], "description": template.METADATA["description"]}
            for key, template in self.templates.items()
        ]

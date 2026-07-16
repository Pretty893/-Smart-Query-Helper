from typing import Dict, Any, Type, Optional
import json
import os


SKILL_REGISTRY = {}


def skill_metadata(
    name: str,
    description: str,
    skill_type: str = "retriever",
    parameters: Optional[Dict[str, Dict]] = None,
    returns: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    examples: Optional[list] = None,
    version: str = "1.0.0"
):
    parameters = parameters or {}
    returns = returns or {}
    tags = tags or []
    examples = examples or []

    def decorator(cls: Type):
        cls.METADATA = {
            "name": name,
            "description": description,
            "skill_type": skill_type,
            "parameters": parameters,
            "returns": returns,
            "tags": tags,
            "examples": examples,
            "version": version,
            "class_name": cls.__name__
        }

        if skill_type not in SKILL_REGISTRY:
            SKILL_REGISTRY[skill_type] = {}
        SKILL_REGISTRY[skill_type][name] = cls

        return cls

    return decorator


def get_skill_metadata(skill_type: str = None, name: str = None) -> Dict[str, Any]:
    if skill_type is None:
        return SKILL_REGISTRY

    if skill_type not in SKILL_REGISTRY:
        return {}

    if name is None:
        return SKILL_REGISTRY[skill_type]

    skill_cls = SKILL_REGISTRY[skill_type].get(name)
    if skill_cls and hasattr(skill_cls, 'METADATA'):
        return skill_cls.METADATA
    return {}


def list_skills(skill_type: str = None) -> list:
    if skill_type is None:
        all_skills = []
        for st in SKILL_REGISTRY:
            for name, cls in SKILL_REGISTRY[st].items():
                if hasattr(cls, 'METADATA'):
                    all_skills.append(cls.METADATA)
        return all_skills

    skills = []
    if skill_type in SKILL_REGISTRY:
        for name, cls in SKILL_REGISTRY[skill_type].items():
            if hasattr(cls, 'METADATA'):
                skills.append(cls.METADATA)
    return skills


def search_skills(query: str) -> list:
    matched = []
    for skill_type in SKILL_REGISTRY:
        for name, cls in SKILL_REGISTRY[skill_type].items():
            if not hasattr(cls, 'METADATA'):
                continue

            metadata = cls.METADATA
            search_text = f"{metadata['name']} {metadata['description']} {' '.join(metadata['tags'])} {' '.join(metadata['examples'])}"

            if query.lower() in search_text.lower():
                matched.append(metadata)

    return matched


def generate_tool_calling_prompt() -> str:
    tools = []
    if "tool" in SKILL_REGISTRY:
        for name, cls in SKILL_REGISTRY["tool"].items():
            if hasattr(cls, 'METADATA'):
                tools.append(cls.METADATA)

    if not tools:
        return "没有可用的工具。"

    prompt = "可用工具列表：\n\n"
    for tool in tools:
        params_str = ""
        for param_name, param_config in tool.get("parameters", {}).items():
            required_mark = "*" if param_config.get("required") else ""
            params_str += f"  - {param_name}{required_mark}: {param_config.get('description', '')} (类型: {param_config.get('type', 'string')})\n"

        prompt += f"""工具名称：{tool['name']}
描述：{tool['description']}
参数：
{params_str}
返回：{tool.get('returns', {}).get('description', '结果')}
示例：{', '.join(tool.get('examples', []))}
标签：{', '.join(tool.get('tags', []))}

"""

    return prompt


def export_metadata_to_json(output_path: str = None):
    output_path = output_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "skills_metadata.json"
    )

    all_metadata = {}
    for skill_type in SKILL_REGISTRY:
        all_metadata[skill_type] = {}
        for name, cls in SKILL_REGISTRY[skill_type].items():
            if hasattr(cls, 'METADATA'):
                all_metadata[skill_type][name] = cls.METADATA

    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    return output_path

from typing import Dict, Any, Optional
import json
import requests

from .config_manager import get_config
config = get_config()
from .skill_registry import skill_metadata


@skill_metadata(
    name="oa_approval_query",
    description="查询OA审批进度，获取审批状态、当前步骤、下一步骤、进度百分比等信息",
    skill_type="tool",
    parameters={
        "approval_id": {
            "type": "string",
            "description": "审批单ID，如APR-2026-001",
            "required": True
        }
    },
    returns={
        "type": "object",
        "description": "审批进度信息",
        "properties": {
            "approval_id": {"type": "string", "description": "审批单ID"},
            "status": {"type": "string", "description": "审批状态：pending/approved/rejected"},
            "current_step": {"type": "string", "description": "当前审批步骤"},
            "next_step": {"type": "string", "description": "下一步骤"},
            "progress": {"type": "string", "description": "进度百分比"},
            "approved_by": {"type": "string", "description": "审批人"},
            "approved_time": {"type": "string", "description": "审批时间"}
        }
    },
    tags=["OA审批", "流程查询", "审批进度"],
    examples=[
        "查询审批单APR-2026-001的进度",
        "我的请假审批到哪一步了",
        "报销审批进展如何"
    ],
    version="1.0.0"
)
class OAApprovalTool:
    def execute(self, approval_id: str) -> Dict[str, Any]:
        return {
            "success": True,
            "data": {
                "approval_id": approval_id,
                "status": "approved",
                "current_step": "部门经理审批",
                "next_step": "HR确认",
                "progress": "75%",
                "approved_by": "张三",
                "approved_time": "2026-07-02 10:30",
            },
        }


@skill_metadata(
    name="email_notify",
    description="发送邮件通知，支持指定收件人、主题和内容",
    skill_type="tool",
    parameters={
        "to": {
            "type": "string",
            "description": "收件人邮箱地址",
            "required": True
        },
        "subject": {
            "type": "string",
            "description": "邮件主题",
            "required": True
        },
        "content": {
            "type": "string",
            "description": "邮件正文内容",
            "required": True
        }
    },
    returns={
        "type": "object",
        "description": "邮件发送结果",
        "properties": {
            "success": {"type": "boolean", "description": "是否发送成功"},
            "to": {"type": "string", "description": "收件人"},
            "subject": {"type": "string", "description": "邮件主题"},
            "status": {"type": "string", "description": "发送状态"},
            "message": {"type": "string", "description": "提示信息"}
        }
    },
    tags=["邮件通知", "消息推送", "通知发送"],
    examples=[
        "给张三发送会议通知",
        "发送报销提醒邮件",
        "通知员工培训安排"
    ],
    version="1.0.0"
)
class EmailNotifyTool:
    def execute(self, to: str, subject: str, content: str) -> Dict[str, Any]:
        return {
            "success": True,
            "data": {
                "to": to,
                "subject": subject,
                "status": "pending",
                "message": "邮件已进入发送队列",
            },
        }


@skill_metadata(
    name="calendar_check",
    description="查询会议室可用性，获取指定日期和时间段可用的会议室列表",
    skill_type="tool",
    parameters={
        "date": {
            "type": "string",
            "description": "日期，格式YYYY-MM-DD",
            "required": True
        },
        "time_range": {
            "type": "string",
            "description": "时间段，如09:00-10:00",
            "required": True
        }
    },
    returns={
        "type": "object",
        "description": "会议室可用性信息",
        "properties": {
            "date": {"type": "string", "description": "查询日期"},
            "time_range": {"type": "string", "description": "查询时间段"},
            "available_rooms": {"type": "list", "description": "可用会议室列表"},
            "occupied_rooms": {"type": "list", "description": "已占用会议室列表"}
        }
    },
    tags=["会议室", "日程查询", "预约管理"],
    examples=[
        "明天下午2点有哪些会议室可用",
        "本周三上午的会议室预订情况",
        "找一个明天上午9点的会议室"
    ],
    version="1.0.0"
)
class CalendarCheckTool:
    def execute(self, date: str, time_range: str) -> Dict[str, Any]:
        return {
            "success": True,
            "data": {
                "date": date,
                "time_range": time_range,
                "available_rooms": ["会议室A", "会议室B", "培训室"],
                "occupied_rooms": ["会议室C"],
            },
        }


@skill_metadata(
    name="hr_data_query",
    description="查询员工HR数据，包括年假剩余天数、在职状态、部门信息等",
    skill_type="tool",
    parameters={
        "employee_id": {
            "type": "string",
            "description": "员工ID",
            "required": True
        },
        "data_type": {
            "type": "string",
            "description": "数据类型：annual_leave/employment_status/department",
            "required": True
        }
    },
    returns={
        "type": "object",
        "description": "员工HR数据",
        "properties": {
            "annual_leave": {"type": "object", "description": "年假信息"},
            "employment_status": {"type": "object", "description": "在职状态"},
            "department": {"type": "object", "description": "部门信息"}
        }
    },
    tags=["HR数据", "员工信息", "年假查询"],
    examples=[
        "查询员工1001的年假剩余天数",
        "查看张三的在职状态",
        "李四在哪个部门"
    ],
    version="1.0.0"
)
class HRDataQueryTool:
    def execute(self, employee_id: str, data_type: str) -> Dict[str, Any]:
        mock_data = {
            "annual_leave": {"remaining_days": 5, "used_days": 10, "total_days": 15},
            "employment_status": {"status": "在职", "hire_date": "2024-01-15", "probation_end": "2024-04-15"},
            "department": {"name": "技术部", "manager": "李四"},
        }
        return {
            "success": True,
            "data": mock_data.get(data_type, {}),
        }


class ToolCaller:
    def __init__(self):
        self.tools = {
            "oa_approval_query": OAApprovalTool(),
            "email_notify": EmailNotifyTool(),
            "calendar_check": CalendarCheckTool(),
            "hr_data_query": HRDataQueryTool(),
        }

    def list_tools(self) -> list:
        return [tool.METADATA for tool in self.tools.values()]

    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"success": False, "error": f"工具 {tool_name} 不可用"}

        try:
            tool = self.tools[tool_name]
            metadata = tool.METADATA

            for param_name, param_config in metadata.get("parameters", {}).items():
                if param_config.get("required") and param_name not in kwargs:
                    return {"success": False, "error": f"缺少必需参数: {param_name}"}

            return tool.execute(**kwargs)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_tool_description(self, tool_name: str) -> Optional[str]:
        tool = self.tools.get(tool_name)
        return tool.METADATA["description"] if tool else None

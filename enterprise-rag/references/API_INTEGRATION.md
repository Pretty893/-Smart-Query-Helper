# 外部系统对接文档

## 概述

本文档描述了如何将外部系统 API 集成到 RAG 系统中，实现与企业现有系统的联动。

## 集成架构

```
RAG 系统 ←→ ToolCaller ←→ 外部系统 API
              │
              ▼
         统一调用接口
```

## 支持的外部系统

### 1. OA 审批系统

**功能**：查询审批进度、发起审批

**调用方式**：

```python
from enterprise_rag.scripts.tool_caller import ToolCaller

tool_caller = ToolCaller()
result = tool_caller.call_tool(
    "oa_approval_query",
    approval_id="APR20260702001"
)
```

**返回结果**：

```json
{
    "success": true,
    "data": {
        "approval_id": "APR20260702001",
        "status": "approved",
        "current_step": "部门经理审批",
        "next_step": "HR确认",
        "progress": "75%",
        "approved_by": "张三",
        "approved_time": "2026-07-02 10:30"
    }
}
```

### 2. 邮件通知系统

**功能**：发送邮件通知

**调用方式**：

```python
result = tool_caller.call_tool(
    "email_notify",
    to="employee@company.com",
    subject="审批进度提醒",
    content="您的请假申请已通过审批"
)
```

**返回结果**：

```json
{
    "success": true,
    "data": {
        "to": "employee@company.com",
        "subject": "审批进度提醒",
        "status": "pending",
        "message": "邮件已进入发送队列"
    }
}
```

### 3. 日历系统

**功能**：查询会议室可用性

**调用方式**：

```python
result = tool_caller.call_tool(
    "calendar_check",
    date="2026-07-05",
    time_range="10:00-12:00"
)
```

**返回结果**：

```json
{
    "success": true,
    "data": {
        "date": "2026-07-05",
        "time_range": "10:00-12:00",
        "available_rooms": ["会议室A", "会议室B", "培训室"],
        "occupied_rooms": ["会议室C"]
    }
}
```

### 4. HR 数据系统

**功能**：查询员工 HR 数据

**调用方式**：

```python
result = tool_caller.call_tool(
    "hr_data_query",
    employee_id="EMP001",
    data_type="annual_leave"
)
```

**返回结果**：

```json
{
    "success": true,
    "data": {
        "remaining_days": 5,
        "used_days": 10,
        "total_days": 15
    }
}
```

## 集成步骤

### 步骤一：注册工具

在 `tool_caller.py` 的 `available_tools` 字典中注册新工具：

```python
"custom_tool": {
    "name": "custom_tool",
    "description": "自定义工具描述",
    "parameters": ["param1", "param2"],
}
```

### 步骤二：实现工具

在 `_execute_tool` 方法中添加工具的实现：

```python
def _execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
    if tool_name == "custom_tool":
        return self._custom_tool(**kwargs)
```

### 步骤三：调用工具

在业务逻辑中调用工具：

```python
result = tool_caller.call_tool("custom_tool", param1="value1", param2="value2")
```

## 安全考虑

### API 密钥管理

将 API 密钥存储在环境变量或配置文件中，不要硬编码：

```python
import os

API_KEY = os.environ.get("OA_API_KEY")
```

### 请求限制

添加请求频率限制，避免滥用：

```python
import time

last_request_time = 0
REQUEST_INTERVAL = 1  # 1秒

def call_tool(self, tool_name: str, **kwargs):
    global last_request_time
    current_time = time.time()
    if current_time - last_request_time < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - (current_time - last_request_time))
    last_request_time = time.time()
    return self._execute_tool(tool_name, **kwargs)
```

### 错误处理

添加完善的错误处理机制：

```python
def call_tool(self, tool_name: str, **kwargs):
    try:
        return self._execute_tool(tool_name, **kwargs)
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "网络连接失败"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## 监控与日志

记录每次工具调用的信息：

```python
import logging

def call_tool(self, tool_name: str, **kwargs):
    logging.info(f"调用工具: {tool_name}")
    logging.info(f"参数: {kwargs}")
    start_time = time.time()
    result = self._execute_tool(tool_name, **kwargs)
    end_time = time.time()
    logging.info(f"耗时: {end_time - start_time:.2f}秒")
    logging.info(f"结果: {result}")
    return result
```

## 扩展指南

### 添加新的外部系统

1. 在 `available_tools` 中注册新工具
2. 实现对应的 `_execute_tool` 方法
3. 在文档中添加新工具的说明

### 批量调用

支持批量调用多个工具：

```python
def batch_call(self, tools: list) -> list:
    results = []
    for tool in tools:
        result = self.call_tool(tool["name"], **tool["parameters"])
        results.append(result)
    return results
```

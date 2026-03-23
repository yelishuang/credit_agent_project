"""
Agent 编排器 — 加载微调 Qwen2.5-14B，驱动多轮 tool_call 循环完成信贷审批。
"""

import json
import re
import logging
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "base_models", "Qwen2.5-14B-Instruct")
LORA_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "outputs", "qwen2.5_14b_lora", "20260323_150000")

MAX_TURNS = 6

SYSTEM_PROMPT = (
    "你是信用风险评估专家Agent。你拥有以下能力：查询用户数据、调用风险模型、检索风控知识库。\n"
    "请先思考分析步骤，再调用合适的工具，最后给出完整的风险评估报告。\n\n"
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
    '{"type": "function", "function": {"name": "query_user_credit_data", "description": "根据用户ID查询其信用数据（收入、负债、逾期等）", "parameters": {"type": "object", "properties": {"user_id": {"type": "integer", "description": "用户ID"}}, "required": ["user_id"]}}}\n'
    '{"type": "function", "function": {"name": "predict_risk_score", "description": "调用信用风险模型预测违约概率", "parameters": {"type": "object", "properties": {"features": {"type": "object", "description": "用户特征字典"}}, "required": ["features"]}}}\n'
    '{"type": "function", "function": {"name": "search_knowledge_base", "description": "在风控知识库中搜索相关政策法规和业务规则", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "检索关键词"}, "top_k": {"type": "integer", "description": "返回条数，默认3"}}, "required": ["query"]}}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": "<function-name>", "arguments": <args-json-object>}\n'
    "</tool_call>"
)

def parse_tool_calls(text: str) -> list[dict]:
    """从模型输出中解析 <tool_call> 标签。"""
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    calls = []
    for m in re.finditer(pattern, text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
        except json.JSONDecodeError as e:
            logger.warning(f"tool_call JSON 解析失败: {e}, 原文: {m.group(1)[:200]}")
    return calls


class CreditAgent:
    """信贷审批 Agent：加载模型，驱动多轮 tool_call 循环。"""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载基座模型 + LoRA adapter 并合并。"""
        logger.info(f"加载 tokenizer: {BASE_MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

        logger.info(f"加载基座模型: {BASE_MODEL_PATH}")
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )

        logger.info(f"加载 LoRA adapter: {LORA_ADAPTER_PATH}")
        self.model = PeftModel.from_pretrained(self.model, LORA_ADAPTER_PATH)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        logger.info("模型加载完成")

    def run(self, user_input: str, verbose: bool = True) -> str:
        """执行 Agent 主循环，返回最终回答文本。"""
        from src.agent.tool_executor import execute_tool

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        for turn in range(MAX_TURNS):
            # 生成
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=2048, do_sample=False,
                    temperature=1.0, top_p=1.0, repetition_penalty=1.05,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            messages.append({"role": "assistant", "content": response})

            # 解析 tool_call
            tool_calls = parse_tool_calls(response)
            if not tool_calls:
                if verbose:
                    print(f"\n{response}")
                return response

            # 打印思考 + 执行工具
            # 提取思考部分（<tool_call> 之前的文本）
            thought = response.split("<tool_call>")[0].strip()
            if verbose and thought:
                print(f"\n[思考] {thought}")

            tool_responses = []
            for tc in tool_calls:
                if verbose:
                    args_str = json.dumps(tc["arguments"], ensure_ascii=False)
                    print(f"[调用工具] {tc['name']}({args_str})")

                result = execute_tool(tc["name"], tc["arguments"])

                if verbose:
                    # 截断过长的输出
                    display = result if len(result) <= 500 else result[:500] + "..."
                    print(f"[工具返回] {display}")

                tool_responses.append(result)

            # 将所有工具结果包装为 <tool_response> 追加到对话
            resp_parts = [f"<tool_response>\n{r}\n</tool_response>" for r in tool_responses]
            messages.append({"role": "user", "content": "\n".join(resp_parts)})

        # 超过最大轮次 — 取最后一条 assistant 回复
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                final = msg["content"]
                break
        else:
            final = "达到最大对话轮次限制。"
        if verbose:
            print(f"\n{final}")
        return final

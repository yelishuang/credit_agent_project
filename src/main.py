#!/usr/bin/env python3
"""
CreditAgent CLI 入口 — 信贷审批 AI Agent 交互界面。

用法:
  python src/main.py --query "帮我审批客户ID=57543的贷款申请"   # 单次查询
  python src/main.py                                           # 交互模式
"""

import argparse
import logging
import os
import sys

# 离线模式
os.environ["HF_HUB_OFFLINE"] = "1"

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="CreditAgent 信贷审批 AI Agent")
    parser.add_argument("--query", type=str, default=None, help="单次查询（不指定则进入交互模式）")
    args = parser.parse_args()

    from src.agent.orchestrator import CreditAgent

    print("正在加载模型，请稍候...")
    agent = CreditAgent()
    agent.load_model()
    print("模型加载完成，Agent 就绪。\n")

    if args.query:
        agent.run(args.query, verbose=True)
    else:
        print("进入交互模式（输入 quit 或 exit 退出）")
        print("-" * 50)
        while True:
            try:
                user_input = input("\n用户> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not user_input or user_input.lower() in ("quit", "exit"):
                print("再见！")
                break
            agent.run(user_input, verbose=True)


if __name__ == "__main__":
    main()

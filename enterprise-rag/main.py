import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from scripts.graph_state import AgentGraph
from scripts.skill_registry import list_skills, export_metadata_to_json
from scripts.logger import get_logger, RequestContext


def main():
    logger = get_logger("main")
    logger.info("OfficeMate 企业智能问答助手启动")
    
    print("OfficeMate - 企业智能问答助手")
    print("正在初始化系统...")
    
    agent = AgentGraph()
    
    print("系统初始化完成！")
    print("输入 'help' 查看可用命令，输入 'exit' 退出")
    print("=" * 50)
    
    history = []
    
    while True:
        query = input("\n请输入您的问题：")
        
        if query.strip().lower() == "exit":
            logger.info("用户退出系统")
            print("感谢使用 OfficeMate，再见！")
            break
        
        if query.strip().lower() == "help":
            print("\n可用命令：")
            print("  exit - 退出系统")
            print("  help - 显示帮助信息")
            print("  skills - 查看所有可用技能")
            print("  export - 导出技能元数据")
            continue
        
        if query.strip().lower() == "skills":
            skills = list_skills()
            print(f"\n共找到 {len(skills)} 个技能：")
            for skill in skills:
                print(f"\n【{skill['name']}】")
                print(f"类型：{skill.get('skill_type', '未知')}")
                print(f"描述：{skill['description']}")
                print(f"标签：{', '.join(skill.get('tags', []))}")
            continue
        
        if query.strip().lower() == "export":
            export_metadata_to_json("skills_metadata.json")
            print("\n技能元数据已导出到 skills_metadata.json")
            continue
        
        print("\n正在处理您的问题...")
        
        try:
            result = agent.run(query, history)
            
            request_id = result.get("request_id", "N/A")
            logger.info(
                "请求处理完成",
                extra={"request_id": request_id, "query": query[:50]},
            )
            
            print(f"请求ID：{request_id}")
            
            if result.get("error"):
                print(f"\n错误：{result['error']}")
                continue
            
            routing_result = result.get("routing_result", {})
            print(f"意图类型：{routing_result.get('intent_type')}")
            print(f"检索器：{routing_result.get('retriever_type')}")
            print(f"选择理由：{routing_result.get('reasoning')}")
            
            answer = result.get("post_processed_answer") or result.get("answer")
            if answer:
                print("\n" + "=" * 50)
                print("回答：")
                print(answer)
                print("=" * 50)
                
                history.append({"user": query, "assistant": answer})
                
                post_process = input("\n是否需要后处理（摘要/翻译/高亮）？(y/n) ")
                if post_process.lower() == "y":
                    print("\n可选后处理：")
                    print("  1. 摘要")
                    print("  2. 翻译")
                    print("  3. 关键词高亮")
                    choice = input("请选择（1/2/3）：")
                    
                    if choice == "1":
                        summary = agent.post_processor.generate_summary(answer, max_length=150)
                        print("\n摘要：")
                        print(summary)
                    elif choice == "2":
                        lang = input("请输入目标语言（如English/Japanese）：")
                        translation = agent.post_processor.translate(answer, lang)
                        print(f"\n翻译为{lang}：")
                        print(translation)
                    elif choice == "3":
                        keywords = input("请输入需要高亮的关键词（逗号分隔）：").split(",")
                        highlighted = agent.post_processor.add_highlights(answer, keywords)
                        print("\n高亮结果：")
                        print(highlighted)
            else:
                print("未生成回答")
                
        except Exception as e:
            logger.error("请求处理失败", extra={"error": str(e), "query": query[:50]})
            print(f"处理过程中出现错误：{str(e)}")


if __name__ == "__main__":
    main()

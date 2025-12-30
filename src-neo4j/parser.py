from neo4j import GraphDatabase
import re
from typing import List, Dict, Any
import jieba
import requests
import json
from openai import OpenAI

class LawGraphQueryParser:
    def __init__(self, uri="bolt://localhost:7689", user="neo4j", password="password", database="neo4j", qwen_api_key=None):
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        api_key = qwen_api_key or os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            self.qwen_client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            self.qwen_client = None
        
    def close(self):
        self.driver.close()
    
    def parse_and_search(self, user_query: str, use_rag: bool = True, stream: bool = False) -> Dict[str, Any]:
        """解析用户查询并返回检索结果"""
        results = {
            "query": user_query,
            "matches": [],
            "related_questions": [],
            "legal_articles": [],
            "ai_answer": None
        }
        
        keywords = self._extract_keywords(user_query)
        
        if keywords:
            results["matches"] = self._search_by_keywords(keywords)
        
        results["related_questions"] = self._search_similar_questions(user_query, 20)
        results["legal_articles"] = self._get_related_articles(results["matches"])

        if use_rag and self.qwen_client:
            if stream:
                results["ai_answer_stream"] = self._generate_rag_answer_stream(user_query, results)
            else:
                results["ai_answer"] = self._generate_rag_answer(user_query, results)
        
        return results
    
    def _generate_rag_answer(self, user_query: str, retrieval_results: Dict) -> Dict[str, Any]:
        
        context = self._build_context(retrieval_results)
        
        prompt = self._build_messages(user_query, context)
        
        try:
            response = self._call_qwen_api(prompt)
            return {
                "success": True,
                "answer": response,
                "context_used": len(retrieval_results["matches"])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": None
            }

    def _generate_rag_answer_stream(self, user_query: str, retrieval_results: Dict,
                                   model: str = "qwen-plus"):
        """使用检索结果和Qwen生成答案（流式）"""
        context = self._build_context(retrieval_results)
        messages = self._build_messages(user_query, context)
        
        try:
            completion = self.qwen_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=2000,
                stream=True
            )
            
            # 返回生成器
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"\n[错误: {str(e)}]"

    def _build_context(self, retrieval_results: Dict) -> str:
        """从检索结果构建上下文"""
        context_parts = []
        
        # 添加匹配的问答对
        if retrieval_results["matches"]:
            context_parts.append("【相关法律问答】")
            for i, match in enumerate(retrieval_results["matches"][:3], 1):
                context_parts.append(f"\n问题{i}: {match['question']}")
                context_parts.append(f"回答{i}: {match['answer'][:500]}")  # 限制长度
                if match.get('articles'):
                    context_parts.append(f"相关法条: {', '.join(match['articles'][:3])}")
        
        # 添加相关法条
        if retrieval_results["legal_articles"]:
            context_parts.append("\n\n【相关法律条文】")
            for article in retrieval_results["legal_articles"][:5]:
                context_parts.append(
                    f"- {article['article_name']} ({article['law_name']}) "
                    f"[引用次数: {article['citation_count']}]"
                )
        
        return "\n".join(context_parts)

    def _build_messages(self, user_query: str, context: str) -> List[Dict[str, str]]:
        """构建对话消息"""
        system_prompt = """你是一个专业的法律顾问助手。请根据检索到的法律知识库内容回答用户问题。

回答要求：
1. 准确专业，引用具体法律条文
2. 如果检索内容不足，请明确指出
3. 语言通俗易懂，便于理解
4. 提供相关建议或注意事项
5. 回答有逻辑结构，分点说明"""

        user_content = f"""用户问题: {user_query}

检索到的相关内容:
{context}

请基于以上内容回答用户问题。"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    
    def _extract_keywords(self, query: str) -> List[str]:

        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', 
                 '人', '都', '一', '一个', '上', '也', '很', '到', '说', 
                 '要', '去', '你', '会', '着', '没有', '看', '好', '自己', 
                 '这', '怎么', '如何', '么', '吗', '呢'}

        words = jieba.lcut(query)
        keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
        
        return keywords[:5]
    
    def _search_by_keywords(self, keywords: List[str]) -> List[Dict]:

        cypher = """
        MATCH (q:Question)-[:QUESTION_KEYWORD]->(k:Keyword)
        WHERE k.keyword IN $keywords
        WITH q, COUNT(DISTINCT k) as keyword_match_count
        MATCH (q)-[:HAS_ANSWER]->(a:Answer)
        RETURN q.id as question_id,
            q.question_text as question,
            a.answer_id as answer_id,
            a.answer_text as answer,
            keyword_match_count
        ORDER BY keyword_match_count DESC
        LIMIT 10
        """
    
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, keywords=keywords)
            matches = [dict(record) for record in result]
        
            for match in matches:
                match['articles'] = self._get_articles_for_answer(match['answer_id'])
        
            return matches
    
    def _get_articles_for_answer(self, answer_id):
        cypher = """
        MATCH (a:Answer {answer_id: $answer_id})-[:CITES_ARTICLE]->(la:LegalArticle)
        RETURN la.article_name as article_name
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, answer_id=answer_id)
            return [record['article_name'] for record in result]
    
    def _search_similar_questions(self, query, limit):
        """文本相似度检索"""
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            limit = 5
    
        # 使用单个 MATCH 子句，用逗号分隔多个模式
        cypher = f"""
        MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)
        WHERE q.question_text CONTAINS $query_part
        RETURN q.id as question_id,
               q.question_text as question,
               a.answer_text as answer
        LIMIT {limit}
        """
    
        query_part = query[:10] if len(query) > 10 else query
    
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, query_part=query_part)
            return [dict(record) for record in result]
    
    def _get_related_articles(self, matches):
        """获取匹配问题的相关法条"""
        if not matches:
            return []
        
        question_ids = [m['question_id'] for m in matches]
        
        cypher = """
        MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)-[:CITES_ARTICLE]->(la:LegalArticle)
        WHERE q.id IN $question_ids
        WITH la, COUNT(DISTINCT q) as citation_count
        RETURN la.article_id as article_id,
               la.article_name as article_name,
               la.law_name as law_name,
               la.article_number as article_number,
               citation_count
        ORDER BY citation_count DESC
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, question_ids=question_ids)
            return [dict(record) for record in result]
    
    def search_by_article(self, article_name: str) -> List[Dict]:
        """根据法条检索问答"""
        cypher = """
        MATCH (la:LegalArticle)<-[:CITES_ARTICLE]-(a:Answer)<-[:HAS_ANSWER]-(q:Question)
        WHERE la.article_name CONTAINS $article_name 
        RETURN q.question_text as question,
               a.answer_text as answer,
               la.article_name as article
        LIMIT 20
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, article_name=article_name)
            return [dict(record) for record in result]
    
    def get_question_graph(self, question_id: str) -> Dict:
        """获取问题的完整图结构"""
        with self.driver.session(database=self.database) as session:
            # 获取基本信息
            cypher_base = """
            MATCH (q:Question {id: $question_id})-[:HAS_ANSWER]->(a:Answer)
            RETURN q.question_text as question, a.answer_text as answer
            """
            base_result = session.run(cypher_base, question_id=question_id).single()
            if not base_result:
                return {}
            
            result = dict(base_result)
            
            # 获取问题关键词
            cypher_qk = """
            MATCH (q:Question {id: $question_id})-[:QUESTION_KEYWORD]->(k:Keyword)
            RETURN k.keyword as keyword
            """
            result['question_keywords'] = [r['keyword'] for r in session.run(cypher_qk, question_id=question_id)]
            
            # 获取答案关键词
            cypher_ak = """
            MATCH (q:Question {id: $question_id})-[:HAS_ANSWER]->(a:Answer)-[:ANSWER_KEYWORD]->(k:Keyword)
            RETURN k.keyword as keyword
            """
            result['answer_keywords'] = [r['keyword'] for r in session.run(cypher_ak, question_id=question_id)]
            
            # 获取法条
            cypher_la = """
            MATCH (q:Question {id: $question_id})-[:HAS_ANSWER]->(a:Answer)-[:CITES_ARTICLE]->(la:LegalArticle)
            RETURN la.article_name as article
            """
            result['legal_articles'] = [r['article'] for r in session.run(cypher_la, question_id=question_id)]
            
            return result


# 使用示例
if __name__ == "__main__":
    parser = LawGraphQueryParser(qwen_api_key="XXXXX")
    
    try:
        query = "合同纠纷怎么处理"
        results = parser.parse_and_search(query)
        
        print(f"查询: {query}")
        print(f"关键词: {parser._extract_keywords(query)}\n")
        
        print("=== 匹配结果 ===")
        for i, match in enumerate(results["matches"][:3], 1):
            print(f"\n{i}. {match['question']}")
            print(f"   答案: {match['answer'][:100]}...")
            print(f"   匹配度: {match['keyword_match_count']}")
            if match['articles']:
                print(f"   法条: {match['articles'][:3]}")
        
        print("\n=== 相关法条 ===")
        for article in results["legal_articles"][:5]:
            print(f"- {article['article_name']} (引用{article['citation_count']}次)")
            
        if results["ai_answer"] and results["ai_answer"]["success"]:
            print("\n" + "=" * 60)
            print("AI生成的专业解答")
            print("=" * 60)
            print(results["ai_answer"]["answer"])
            print(f"\n(使用模型: {results['ai_answer']['model']}, "
                  f"基于 {results['ai_answer']['context_used']} 条检索结果)")
        elif results.get("ai_answer"):
            print(f"\nAI答案生成失败: {results['ai_answer']['error']}")
            
        print("\n\n" + "=" * 60)
        print("=" * 60)
        
        query2 = "劳动合同到期不续签需要补偿吗"
        print(f"\n查询: {query2}\n")
        
        results2 = parser.parse_and_search(query2, use_rag=True, stream=True)
        if "ai_answer_stream" in results2:
            for chunk in results2["ai_answer_stream"]:
                print(chunk, end="", flush=True)
            print("\n")
        
    finally:
        parser.close()

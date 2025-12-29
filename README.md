# Hw04 基于法律QA的知识图谱与问答系统搭建
## 项目介绍
  知识图谱与大模型在各大行业广泛运用，本次使用法律问答对数据，以及结合大模型技术来搭建一个智能问答系统

## 数据与模型介绍 
  采用的数据为:https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT
  这个数据的设计本意是用来进行监督微调，强化反馈的，这里我们拿他来抽取实体  
  由于开源公开的法律词典难以获得，使用jieba等库，没有经过专业的法律词汇的训练，因此结合大模型与jieba共同抽取实体：
  ``` {python}
import re
import json
import pandas as pd
import torch
import jieba
import jieba.analyse
from vllm import LLM, SamplingParams
from collections import defaultdict
from tqdm import tqdm
import os


class LegalKnowledgeBase:
    def __init__(self):
        self.vocab = {
            "法律名称": set(),
            "条款编号": set(),
            "法律概念": set()
        }
        self.article_index = {}
        self.keyword_index = defaultdict(list)

    def build_from_reference(self, jsonl_path):
        ref_pattern = r'《([^》]+)》第([0-9一二三四五六七八九十百零]+)条[：:](.+)'

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="解析Reference"):
                data = json.loads(line)
                for ref in data.get('reference', []):
                    match = re.search(ref_pattern, ref)
                    if match:
                        law_name = match.group(1)
                        article_num = match.group(2)
                        content = match.group(3).strip()

                        self.vocab["法律名称"].add(law_name)
                        full_article = f"《{law_name}》第{article_num}条"
                        self.vocab["条款编号"].add(full_article)

                        key = (law_name, article_num)
                        self.article_index[key] = {
                            "full_name": full_article,
                            "content": content,
                            "raw": ref
                        }

                        keywords = jieba.analyse.extract_tags(
                            content, topK=10, withWeight=False)
                        for kw in keywords:
                            self.keyword_index[kw].append(key)
                            self.vocab["法律概念"].add(kw)

        print(f"知识库构建完成:")
        print(f"  - 法律名称: {len(self.vocab['法律名称'])} 个")
        print(f"  - 条款数量: {len(self.article_index)} 条")
        print(f"  - 关键词数: {len(self.keyword_index)} 个")
        return self

    def search_articles(self, query, top_k=5):
        query_keywords = jieba.analyse.extract_tags(
            query, topK=5, withWeight=False)

        article_scores = defaultdict(int)
        for kw in query_keywords:
            for article_key in self.keyword_index.get(kw, []):
                article_scores[article_key] += 1

        ranked = sorted(article_scores.items(),
                        key=lambda x: x[1], reverse=True)
        return [self.article_index[k] for k, _ in ranked[:top_k]]


class SimpleLegalExtractor:

    def __init__(self, model_path):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            trust_remote_code=True
        )
        self.tokenizer = self.llm.get_tokenizer()

    def parse_json(self, text):
        try:
            return json.loads(text.strip())
        except:
            try:
                def fix_keywords(match):
                    keywords = [match.group(1)]
                    remaining = text[match.end():]
                    extra_kw = re.findall(
                        r',\s*"([^"]+)"(?=\s*[,}])', remaining[:100])
                    keywords.extend(extra_kw)
                    return f'"关键词": {json.dumps(keywords, ensure_ascii=False)}'

                text = re.sub(pattern, fix_keywords, text, count=1)

                if not text.strip().endswith('}'):
                    text = text.rsplit(',', 1)[0] + '}'
                return json.loads(text)
            except Exception as e:
                print(f"修复失败: {e}")
                try:
                    laws = re.findall(r'"法律名称":\s*"([^"]+)"', text)
                    articles = re.findall(r'"条款编号":\s*"([^"]+)"', text)
                    keywords = re.findall(
                        r'"([^"]+)"(?=\s*[,}])', text.split('"关键词"')[1] if '"关键词"' in text else "")

                    return {
                        "法律名称": laws,
                        "条款编号": articles,
                        "关键词": keywords[:7]  # 限制数量
                    }
                except:
                    pass
        return {}

    def extract_answer_entities(self, texts):
        prompts = []
        for text in texts:
            content = f"""从以下法律文本中提取信息，严格按JSON格式输出：

文本：
{text}

提取规则：
1. 法律名称：书名号《》内的完整法律名称
2. 条款编号：包含"第X条"的完整描述
3. 关键词：使用分词提取的核心概念（3-5个即可）

** 输出示例（严格遵守）：**
{{"法律名称": ["中华人民共和国劳动法"], "条款编号": ["第二十五条"], "关键词": ["劳动能力", "鉴定", "期限"]}}

注意：
- 所有字段值都必须是数组[]
- 关键词用逗号分隔并放在数组内
- 不要有多余的逗号
JSON："""

            messages = [
                {"role": "system", "content": "你是信息抽取助手，只输出JSON，不要解释。"},
                {"role": "user", "content": content}
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            )

        outputs = self.llm.generate(
            prompts,
            SamplingParams(
                temperature=0.1,
                max_tokens=256,
                stop=["}", "\n\n"]
            )
        )

        results = []
        for i, o in enumerate(outputs):
            raw_output = o.outputs[0].text
            if not raw_output.strip().endswith('}'):
                raw_output += '}'

            parsed = self.parse_json(raw_output)
            if not parsed:
                print(f"[警告] 第{i}条解析失败")
                print(f"原始输出: {raw_output[:200]}")
            results.append(parsed)

        return results

    def regex_extract(self, text):
        return {
            "法律名称": list(set(re.findall(r'《([^》]+)》', text))),
            "条款编号": list(set(re.findall(r'《[^》]+》第[0-9一二三四五六七八九十百零]+条', text))),
            "关键词": jieba.analyse.extract_tags(text, topK=5, withWeight=False)
        }


def normalize_entity_result(entities):
    if not entities:
        return {"法律名称": [], "条款编号": [], "关键词": []}

    normalized = {}
    for key in ["法律名称", "条款编号", "关键词"]:
        value = entities.get(key, [])
        if isinstance(value, str):
            normalized[key] = [value] if value else []
        elif isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = []

    return normalized


def run_pipeline(qa_path, ref_path, model_path, output_path, batch_size=100, start_from=0, debug_mode=False):
    kb = LegalKnowledgeBase().build_from_reference(ref_path)

    # 第二步：初始化抽取器
    extractor = SimpleLegalExtractor(model_path)

    # 第三步：读取QA数据
    df = pd.read_json(qa_path, lines=True)
    total_count = len(df)
    questions = df['input'].tolist()
    answers = df['output'].tolist()

    if debug_mode:
        print(f"调试模式：只处理第一个batch ({batch_size}条)")
        total_count = min(batch_size, total_count)

    # 第四步：处理数据
    final_results = []
    if start_from > 0 and os.path.exists(output_path):
        print(f"加载已有结果: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        print(f"已处理: {len(final_results)} 条")

    for batch_start in range(start_from, total_count, batch_size):
        batch_end = min(batch_start + batch_size, total_count)
        print(f"\n{'='*60}")
        print(f"处理批次: [{batch_start}, {batch_end}) / {total_count}")
        print(f"{'='*60}")

        # 当前批次数据
        batch_df = df.iloc[batch_start:batch_end]
        questions = batch_df['input'].tolist()
        answers = batch_df['output'].tolist()

        # 批量抽取答案实体
        print("正在抽取答案实体...")
        answer_entities = extractor.extract_answer_entities(answers)

        # 逐条处理
        batch_results = []

        for i in tqdm(range(len(questions)), desc=f"批次进度"):
            question = questions[i]
            answer = answers[i]

            # Question部分：关键词提取
            q_keywords = jieba.analyse.extract_tags(
                question, topK=5, withWeight=False)

            # Answer部分：LLM + 正则
            a_entities = normalize_entity_result(answer_entities[i])
            regex_result = extractor.regex_extract(answer)

            for key in ["法律名称", "条款编号", "关键词"]:
                if key in regex_result:
                    combined = a_entities.get(
                        key, []) + regex_result.get(key, [])
                    a_entities[key] = list(set(combined))

            # 检索相关条款
            related_articles = kb.search_articles(question, top_k=3)

            # 组装
            global_idx = batch_start + i
            item = {
                "id": batch_df.iloc[i].get('id', f'qa_{global_idx}'),
                "question": question,
                "answer": answer,
                "extracted": {
                    "question_keywords": q_keywords,
                    "answer_entities": a_entities,
                    "related_articles": [art["full_name"] for art in related_articles]
                }
            }
            batch_results.append(item)

        related_articles = kb.search_articles(question, top_k=3)

        # 组装结果
        item = {
            "id": df.iloc[i].get('id', f'qa_{i}'),
            "question": question,
            "answer": answer,
            "extracted": {
                "question_keywords": q_keywords,
                "answer_entities": a_entities,
                "related_articles": [art["full_name"] for art in related_articles]
            }
        }

        final_results.extend(batch_results)

        # 第五步：保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"已保存进度: {len(final_results)}/{total_count} 条")

        if debug_mode:
            print("\n调试模式：第一个batch处理完成，程序退出")
            break

    # 统计信息
    print(f"\n{'='*60}")
    print(f"全部处理完成!")
    print(f"总条数: {len(final_results)}")

    success_count = sum(1 for r in final_results
                        if r['extracted']['answer_entities'].get('法律名称')
                        or r['extracted']['answer_entities'].get('条款编号'))
    print(
        f"成功抽取实体: {success_count}/{len(final_results)} ({success_count/len(final_results)*100:.1f}%)")
    print(f"最终结果: {output_path}")


if __name__ == "__main__":
    run_pipeline(
        qa_path="./autodl-tmp/DISC-Law-SFT-Pair-QA-released.jsonl",
        ref_path="./autodl-tmp/DISC-Law-SFT-Triplet-QA-released.jsonl",
        model_path="./autodl-tmp/LawLLM-7B",
        output_path="final_legal_dataset.json",
        batch_size=100,
        start_from=0
    )

  ```

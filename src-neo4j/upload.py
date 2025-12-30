from neo4j import GraphDatabase
import csv
import os

class Neo4jSimpleUploader:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="your_password", database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def import_vertices(self, csv_dir):
        vertex_configs = [
            ("Question", "id", ["id", "question_text"]),
            ("Answer", "answer_id", ["answer_id", "answer_text"]),
            ("Keyword", "keyword", ["keyword"]),
            ("LegalArticle", "article_id", ["article_id", "article_name", "law_name", "article_number"])
        ]

        for label, primary_key, expected_fields in vertex_configs:
            csv_file = os.path.join(csv_dir, f"{label}.csv")
            if not os.path.exists(csv_file):
                print(f" {csv_file} not exist, skip ")
                continue

            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    print(f"{label}.csv is empty skip")
                    continue

                actual_fields = rows[0].keys()
                if not all(field in actual_fields for field in expected_fields):
                    print(f"{label}.csv not complete，expected {expected_fields}，truth {list(actual_fields)}")
                    continue

                with self.driver.session(database=self.database) as session:
                    for row in rows:
                        cypher = f"""
                        MERGE (n:{label} {{{primary_key}: $primary_value}})
                        SET n += $props
                        """
                        props = {k: v for k, v in row.items() if k != primary_key}  
                        session.run(cypher, primary_value=row[primary_key], props=props)

    def import_edges(self, csv_dir):
        edge_configs = [
            ("HAS_ANSWER", "Question", "id", "Answer", "answer_id"),
            ("QUESTION_KEYWORD", "Question", "id", "Keyword", "keyword"),
            ("ANSWER_KEYWORD", "Answer", "answer_id", "Keyword", "keyword"),
            ("CITES_ARTICLE", "Answer", "answer_id", "LegalArticle", "article_id")
        ]

        for edge_label, src_label, src_key, dst_label, dst_key in edge_configs:
            csv_file = os.path.join(csv_dir, f"{edge_label}.csv")
            if not os.path.exists(csv_file):
                print(f"{csv_file} not exist, skip")
                continue

            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    print(f"{edge_label}.csv empty skip")
                    continue

                if list(rows[0].keys()) != ['src', 'dst']:
                    print(f"{edge_label}.csv format error , should include src and dst columns")
                    continue

                cypher = f"""
                MATCH (a:{src_label} {{{src_key}: $src_id}})
                MATCH (b:{dst_label} {{{dst_key}: $dst_id}})
                MERGE (a)-[r:{edge_label}]->(b)
                """

                with self.driver.session(database=self.database) as session:
                    for row in rows:
                        session.run(cypher, src_id=row['src'], dst_id=row['dst'])

            print(f" {edge_label} successfully import（{len(rows)} ）")


if __name__ == "__main__":
    uploader = Neo4jSimpleUploader(
        uri="bolt://localhost:7689",
        user="neo4j",
        password="password", 
        database="neo4j"   
    )

    csv_directory = "tugraph_csv" 

    try:
        uploader.import_vertices(csv_directory)
        uploader.import_edges(csv_directory)
    except Exception as e:
        print(f"Import Error: {e}")
    finally:
        uploader.close()

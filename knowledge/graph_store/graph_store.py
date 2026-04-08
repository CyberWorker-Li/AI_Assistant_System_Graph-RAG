from typing import List, Tuple
import networkx as nx
from knowledge.vector_store.indexer import ChunkRecord
from knowledge.graph_store.triple_extractor import TripleExtractor

class KnowledgeGraphStore:
    def __init__(self, settings=None):
        self.graph = nx.DiGraph()
        self._neo4j_driver = None
        self._neo4j_db = None
        if settings is not None and getattr(settings, "enable_neo4j_graph", False):
            uri = str(getattr(settings, "neo4j_uri", "") or "").strip()
            user = str(getattr(settings, "neo4j_user", "") or "").strip()
            password = str(getattr(settings, "neo4j_password", "") or "").strip()
            database = str(getattr(settings, "neo4j_database", "") or "").strip() or None
            if uri and user and password:
                try:
                    from neo4j import GraphDatabase

                    self._neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
                    self._neo4j_db = database
                    self._neo4j_init_schema()
                except Exception:
                    self._neo4j_driver = None
                    self._neo4j_db = None

    def build_from_chunks(self, chunks: List[ChunkRecord], extractor: TripleExtractor, use_llm: bool = True):
        print(f"开始从 {len(chunks)} 个文本块中提取三元组...")
        all_triples = []
        for i, chunk in enumerate(chunks, 1):
            triples = extractor.extract(chunk.text, use_llm=use_llm)
            if triples:
                all_triples.extend(triples)
            if i % 10 == 0:
                print(f"已处理 {i}/{len(chunks)} 个文本块...")

        print(f"三元组提取完成，共找到 {len(all_triples)} 个。开始构建图谱...")
        rows = []
        for subj, pred, obj in all_triples:
            s = str(subj).strip()
            p = str(pred).strip()
            o = str(obj).strip()
            if not (s and p and o):
                continue
            self.graph.add_edge(s, o, label=p)
            if self._neo4j_driver is not None:
                rows.append({"s": s, "p": p, "o": o})

        if self._neo4j_driver is not None and rows:
            self._neo4j_upsert_triples(rows)

    def _find_matching_nodes(self, entity: str) -> List[str]:
        """寻找精确匹配或包含实体的节点"""
        if self.graph.has_node(entity):
            return [entity]
        # 模糊匹配：查找包含实体名称的所有节点
        matches = [node for node in self.graph.nodes if entity in node or node in entity]
        return matches[:3] # 最多返回3个最相关的

    def retrieve_subgraph_for_entities(self, entities: List[str], k: int = 2) -> List[Tuple[str, str, str]]:
        if not entities:
            return []
        if self._neo4j_driver is not None:
            try:
                return self._neo4j_retrieve_subgraph(entities, k=k)
            except Exception:
                pass

        relevant_triples = set()
        mapped_entities = {}
        for ent in entities:
            matches = self._find_matching_nodes(ent)
            if matches:
                mapped_entities[ent] = matches

        all_actual_nodes = [node for nodes in mapped_entities.values() for node in nodes]
        for node in all_actual_nodes:
            for u, _, data in self.graph.in_edges(node, data=True):
                relevant_triples.add((u, data["label"], node))
            for _, v, data in self.graph.out_edges(node, data=True):
                relevant_triples.add((node, data["label"], v))

        entity_names = list(mapped_entities.keys())
        if len(entity_names) >= 2:
            for i in range(len(entity_names)):
                for j in range(i + 1, len(entity_names)):
                    sources = mapped_entities[entity_names[i]]
                    targets = mapped_entities[entity_names[j]]
                    for s in sources:
                        for t in targets:
                            try:
                                for path in nx.all_simple_paths(self.graph, source=s, target=t, cutoff=k):
                                    for idx in range(len(path) - 1):
                                        u, v = path[idx], path[idx + 1]
                                        relevant_triples.add((u, self.graph.get_edge_data(u, v)["label"], v))
                            except (nx.NodeNotFound, nx.NetworkXNoPath):
                                continue

        return list(relevant_triples)

    def save_to(self, path: str) -> None:
        try:
            nx.write_gpickle(self.graph, path)
        except Exception:
            pass

    def load_from(self, path: str) -> None:
        try:
            self.graph = nx.read_gpickle(path)
        except Exception:
            self.graph = nx.DiGraph()

    def close(self) -> None:
        if self._neo4j_driver is not None:
            try:
                self._neo4j_driver.close()
            except Exception:
                pass

    def neo4j_ready(self) -> bool:
        return self._neo4j_driver is not None

    def neo4j_has_data(self) -> bool:
        if self._neo4j_driver is None:
            return False
        with self._neo4j_session() as s:
            rec = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()
            return int(rec["c"]) > 0 if rec and rec.get("c") is not None else False

    def neo4j_clear(self) -> None:
        if self._neo4j_driver is None:
            return
        with self._neo4j_session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    def _neo4j_session(self):
        if self._neo4j_db:
            return self._neo4j_driver.session(database=self._neo4j_db)
        return self._neo4j_driver.session()

    def _neo4j_init_schema(self) -> None:
        if self._neo4j_driver is None:
            return
        with self._neo4j_session() as s:
            try:
                s.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
            except Exception:
                pass
            try:
                s.run("CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            except Exception:
                pass

    def _neo4j_upsert_triples(self, rows: list[dict]) -> None:
        with self._neo4j_session() as s:
            s.run(
                "UNWIND $rows AS row "
                "MERGE (a:Entity {name: row.s}) "
                "MERGE (b:Entity {name: row.o}) "
                "MERGE (a)-[:REL {label: row.p}]->(b)",
                rows=rows,
            )

    def _neo4j_match_nodes(self, q: str) -> list[str]:
        q = str(q or "").strip()
        if not q:
            return []
        with self._neo4j_session() as s:
            rs = s.run(
                "MATCH (n:Entity) WHERE n.name CONTAINS $q OR $q CONTAINS n.name "
                "RETURN n.name AS name LIMIT 3",
                q=q,
            )
            return [str(r["name"]) for r in rs if r.get("name")]

    def _neo4j_retrieve_subgraph(self, entities: List[str], k: int = 2) -> List[Tuple[str, str, str]]:
        mapped = {}
        for ent in entities:
            matches = self._neo4j_match_nodes(ent)
            if matches:
                mapped[ent] = matches

        nodes = [n for ns in mapped.values() for n in ns]
        triples: set[tuple[str, str, str]] = set()

        if nodes:
            with self._neo4j_session() as s:
                rs = s.run(
                    "MATCH (a:Entity)-[r:REL]->(b:Entity) "
                    "WHERE a.name IN $nodes OR b.name IN $nodes "
                    "RETURN a.name AS s, r.label AS p, b.name AS o LIMIT 60",
                    nodes=nodes,
                )
                for r in rs:
                    sv, pv, ov = r.get("s"), r.get("p"), r.get("o")
                    if sv and pv and ov:
                        triples.add((str(sv), str(pv), str(ov)))

        ent_names = list(mapped.keys())
        if len(ent_names) >= 2:
            pairs = []
            for i in range(len(ent_names)):
                for j in range(i + 1, len(ent_names)):
                    a = mapped[ent_names[i]][0]
                    b = mapped[ent_names[j]][0]
                    pairs.append((a, b))
            for a, b in pairs[:6]:
                with self._neo4j_session() as s:
                    rs = s.run(
                        "MATCH (a:Entity {name:$a}), (b:Entity {name:$b}) "
                        "MATCH p=shortestPath((a)-[:REL*..$k]-(b)) "
                        "UNWIND relationships(p) AS r "
                        "RETURN startNode(r).name AS s, r.label AS p, endNode(r).name AS o LIMIT 60",
                        a=a,
                        b=b,
                        k=int(k),
                    )
                    for r in rs:
                        sv, pv, ov = r.get("s"), r.get("p"), r.get("o")
                        if sv and pv and ov:
                            triples.add((str(sv), str(pv), str(ov)))

        return list(triples)

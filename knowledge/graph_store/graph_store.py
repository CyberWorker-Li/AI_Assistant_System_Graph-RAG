from typing import List, Tuple
import networkx as nx
from knowledge.vector_store.indexer import ChunkRecord
from knowledge.graph_store.triple_extractor import TripleExtractor

class KnowledgeGraphStore:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_from_chunks(self, chunks: List[ChunkRecord], extractor: TripleExtractor, use_llm: bool = True):
        """
        Builds the knowledge graph by extracting triples from a list of text chunks.
        """
        print(f"开始从 {len(chunks)} 个文本块中提取三元组...")
        all_triples = []
        for i, chunk in enumerate(chunks, 1):
            # We use the more powerful LLM-based extractor here
            triples = extractor.extract(chunk.text, use_llm=use_llm)
            if triples:
                all_triples.extend(triples)
            if i % 10 == 0:
                print(f"已处理 {i}/{len(chunks)} 个文本块...")
        
        print(f"三元组提取完成，共找到 {len(all_triples)} 个。开始构建图谱...")
        for subj, pred, obj in all_triples:
            # 为避免图过于庞大，可以做一些清洗和标准化
            s = str(subj).strip()
            p = str(pred).strip()
            o = str(obj).strip()
            if s and p and o:
                self.graph.add_edge(s, o, label=p)

    def _find_matching_nodes(self, entity: str) -> List[str]:
        """寻找精确匹配或包含实体的节点"""
        if self.graph.has_node(entity):
            return [entity]
        # 模糊匹配：查找包含实体名称的所有节点
        matches = [node for node in self.graph.nodes if entity in node or node in entity]
        return matches[:3] # 最多返回3个最相关的

    def retrieve_subgraph_for_entities(self, entities: List[str], k: int = 2) -> List[Tuple[str, str, str]]:
        """
        Retrieves a subgraph connecting the given entities.
        """
        if not entities:
            return []

        relevant_triples = set()
        
        # 1. 映射实体到图中的实际节点
        mapped_entities = {}
        for ent in entities:
            matches = self._find_matching_nodes(ent)
            if matches:
                mapped_entities[ent] = matches

        # 2. 查找所有匹配节点的直接邻居
        all_actual_nodes = [node for nodes in mapped_entities.values() for node in nodes]
        for node in all_actual_nodes:
            # (u) -> [node]
            for u, _, data in self.graph.in_edges(node, data=True):
                relevant_triples.add((u, data['label'], node))
            # [node] -> (v)
            for _, v, data in self.graph.out_edges(node, data=True):
                relevant_triples.add((node, data['label'], v))

        # 3. 查找实体对之间的路径
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
                                        u, v = path[idx], path[idx+1]
                                        relevant_triples.add((u, self.graph.get_edge_data(u, v)['label'], v))
                            except (nx.NodeNotFound, nx.NetworkXNoPath):
                                continue

        return list(relevant_triples)

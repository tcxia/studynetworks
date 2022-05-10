import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from typing import Union, List, Dict, Tuple, Type
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from sklearn.metrics.pairwise import cosine_similarity


class SimCSE(object):
    def __init__(self,
                 model_name_or_path,
                 device=None,
                 num_cells=100,
                 num_cells_in_search=10,
                 pooler=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        if pooler is not None:
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            self.pooler = "cls_before_pooler"
        else:
            self.pooler = "cls"

    def encode(self,
               sentence: Union[str, List[str]],
               device=None,
               return_numpy=False,
               normalize_to_unit=True,
               keepdim=False,
               batch_size=64,
               max_length=128):
        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (
                1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id * batch_size:(batch_id + 1) *
                             batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError

                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1,
                                                              keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence and not keepdim:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()

        return embeddings

    def similarity(self, queries, keys, device=None):
        query_vecs = self.encode(queries, device=device, return_numpy=True)

        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True)
        else:
            key_vecs = keys

        single_query, single_key = len(query_vecs.shape) == 1, len(
            key_vecs.shape) == 1
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)

        similarities = cosine_similarity(query_vecs, key_vecs)

        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        return similarities


    def build_index(self, sentence_or_file_path, use_faiss=None, faiss_fast=False, device=None, batch_size=64):
        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True
            except:
                use_faiss = False

        if isinstance(sentence_or_file_path, str):
            sentences = []
            with open(sentence_or_file_path, "r") as f:
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentence_or_file_path = sentences
        
        embeddings = self.encode(sentence_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)
        self.index = {"sentences": sentence_or_file_path}

        if use_faiss:
            quantizer = faiss.IndexFastIP(embeddings.shape[1])
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentence_or_file_path)))
            else:
                index = quantizer
            
            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(20 * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    pass
            else:
                pass
        
            if faiss_fast:
                index.train(embeddings.astype(np.float32))
                index.nprobe = min(self.num_cells_in_search, len(sentence_or_file_path))
                self.is_faiss_index = True

        else:
            index = embeddings
            self.is_faiss_index = False
        
        self.index["index"] = index


    def add_to_index(self, sentence_or_file_path, device=None, batch_size=64):
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)
        
        if self.is_faiss_index:
            self.index["index"].add(embeddings.astype(np.float32))
        else:
            self.index["index"] = np.concatenate((self.index["index"], embeddings))
        self.index["sentences"] += sentences_or_file_path

    def search(self, queries, device=None, threshold=0.6, top_k=5):
        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device)
                    combined_results.append(results)
                return combined_results
            
            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["sentences"][idx], score) for idx, score in id_and_score]
            return results
        else:
            query_vecs = self.encode(queries, device=device, normalize_to_unit=True, keepdim=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)
            
            def pack_single_result(dist, idx):
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results
            
            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])



if __name__ == "__main__":
    example_senteces = [
        'An animal is biting a persons finger.',
        'A woman is reading.',
        'A man is lifting weights in a garage.',
        'A man plays the violin.',
        'A man is eating food.',
        'A man plays the piano.',
        'A panda is climbing.',
        'A man plays a guitar.',
        'A woman is slicing a meat.',
        'A woman is taking a picture.'
    ]

    example_queries = [
        'A man is playing music.',
        'A woman is making a photo.'
    ]

    model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    simcse = SimCSE(model_name)

    similarities = simcse.similarity(example_queries, example_senteces)
    print(similarities)

    simcse.build_index(example_senteces, use_faiss=False)
    results = simcse.search(example_queries)
    for i, result in enumerate(results):
        for sentence, score in result:
            print("{} (cosine similarity: {:.4f})".format(sentence, score))

    simcse.build_index(example_senteces, use_faiss=True)
    results = simcse.search(example_queries)
    for i, result in enumerate(results):
        for sentence, score in result:
            print("{} (cosine similarity: {:.4f})".format(sentence, score))
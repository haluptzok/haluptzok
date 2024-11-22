#!/bin/python3

import argparse
import math
import os
import random
import re
import sys
from statistics import mean
import pickle
import string
from typing import Dict, List
from collections import defaultdict, Counter
import heapq

# Link to writeup: https://drive.google.com/file/d/1QLZQjfTSub5T76uQWSl6c6AstaBPSWOT/view?usp=sharing


class Bm25:
    """
    Implements the BM25 algorithm, a popular text retrieval function that ranks a set of documents based
    on the query terms appearing in each document, irrespective of their proximity.
    It uses two parameters, k1 and b, for tuning.

    Attributes:
        k1 (float): A free parameter, usually chosen as 1.2, that controls term frequency saturation.
        b (float): A free parameter, usually 0.75, that controls the scaling by document length.
        stopwords (set): A set of stopwords to be ignored during text processing.

    Methods:
        tokenize(text): Tokenizes the given text by removing punctuation and stopwords, converting it to lowercase, and splitting into words.
        process_docs(docs): Processes a list of documents.
        search(query, k): Searches the processed documents for the given query and returns the top k relevant document ids.
    """

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

        # Taken from nltk
        self.stopwords = set()
        with open("stopwords.txt", "r") as f:
            for line in f:
                self.stopwords.add(line.strip())

    def tokenize(self, text: str):
        return [
            word for word in text.lower().translate(
                str.maketrans("", "", string.punctuation)).split()
            if word not in self.stopwords
        ]

    def process_docs(self, docs: List[str]):
        """
        Processes a list of documents to prepare for BM25 scoring.

        Args:
            docs (List[str]): A list of documents (texts) to be processed.

        Returns:
            None: This method updates the internal state of the BM25 object to store necessary information for scoring documents.
        """

        # Array of maps.  Each document has a map, each map is from a token to token frequency
        self.doc_processed = []
        self.doc_tokencount = []  # list of token counts for each document
        self.token_to_doc_count = {}  # map from token to document count for each token
        total_tokens = 0

        # docs = docs[:10]

        for i in range(len(docs)):
            doc_tokens = self.tokenize(docs[i])
            # print("doc_tokens", len(doc_tokens))
            total_tokens += len(doc_tokens)
            # print(f"Total tokens: {total_tokens}")
            doc_map = {}
            for token in doc_tokens:
                if token in doc_map:
                    doc_map[token] += 1
                else:
                    doc_map[token] = 1
            # print("doc_map", doc_map)
            self.doc_tokencount.append(len(doc_tokens))
            self.doc_processed.append(doc_map)
            # walk through the hash table and increment count for each word in this document

            for token in doc_map.keys():
                if token in self.token_to_doc_count:
                    self.token_to_doc_count[token] += 1
                else:
                    self.token_to_doc_count[token] = 1

        self.avgdl = total_tokens / len(docs)
        # print("avgdl", self.avgdl)
        # self.docs = docs

    def search(self, query: str, k: int) -> List[int]:
        """
        Searches the processed documents based on the given query using the BM25 algorithm.
        Returns the top k document ids that are most relevant to the query.

        Note that we want to compute over only the unique query tokens.

        Args:
            query (str): The query string based on which documents are to be retrieved.
            k (int): The number of top documents to return.

        Returns:
            List[int]: A list of document ids representing the top k documents relevant to the query.
        """

        # print(query, k)
        # return idcs not docs

        query_tokens = self.tokenize(query)
        doc_score_list = []

        for i_doc in range(len(self.doc_processed)):
            doc_map_t_c = self.doc_processed[i_doc]
            doc_score = 0.0
            for query_token in query_tokens:
                idf_token = math.log(1.0 + (
                    (len(self.doc_processed) -
                     self.token_to_doc_count.get(query_token, 0) + 0.5) /
                    (self.token_to_doc_count.get(query_token, 0) + 0.5)))
                token_score = (doc_map_t_c.get(query_token, 0) * (
                    self.k1 + 1)) / (doc_map_t_c.get(query_token, 0) + self.k1 *
                                    (1 - self.b + (self.b *
                                     (self.doc_tokencount[i_doc] / self.avgdl))))
                token_score *= idf_token
                doc_score += token_score

            doc_score_list.append((doc_score, i_doc))

        # Array of doc_scores in doc_score_list, return the top k.

        # If I was smart and had time - heap - and just keep the largest k elements.
        # heapq.nlargest()
        # sorted all and pick the first k

        doc_score_list.sort(reverse=True)
        # print(doc_score_list)

        return [doc_score_list[i][1] for i in range(k)]


def evaluate(searcher: Bm25,
             queries: List[str],
             qrels: Dict[int, List[int]],
             k=10):
    """
    Evaluates the effectiveness of a search algorithm using a set of queries and corresponding relevance judgments (qrels).
    It calculates the mean reciprocal rank (MRR) of the search results.

    Args:
        searcher (Bm25): An instance of the Bm25 class.
        queries (list): A list of query strings to be evaluated.
        qrels (dict): A dictionary with query ids as keys and lists of relevant document ids as values.
        k (int): The number of top documents to consider for each query (default is 10).

    Returns:
        float: The mean reciprocal rank (MRR) for the given set of queries.
    """
    ranks = []
    for qid, query in enumerate(queries):
        pids = searcher.search(query, k=k)
        rank = next((1 / (i + 1)
                     for i, pid in enumerate(pids) if pid in qrels[qid]), 0)
        ranks.append(rank)
    return mean(ranks)


def load_dataset():
    with open("dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    docs = dataset["docs"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]
    return (docs, queries, qrels)


def main(args):
    docs, queries, qrels = load_dataset()
    searcher = Bm25()
    searcher.process_docs(docs)
    acc = evaluate(searcher, queries, qrels, args.k)
    print(f"MRR@{args.k} = {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25")
    parser.add_argument("-k",
                        type=int,
                        default=10,
                        help="Number of documents to return")
    args = parser.parse_args()
    main(args)

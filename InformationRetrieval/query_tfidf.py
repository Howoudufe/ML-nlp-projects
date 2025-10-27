import math
from collections import defaultdict
from string_processing import (
    process_tokens,
    tokenize_text,
)
from query import (
    get_query_tokens,
    count_query_tokens,
    query_main,
)


def get_doc_to_norm(index, doc_freq, num_docs):
    """Pre-compute the norms for each document vector in the corpus using tfidf.

    Args:
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        num_docs (int): number of documents in the corpus

    Returns:
        dict(int: float): a dictionary mapping doc_ids to document norms
    """

    # TODO: Implement this function using tfidf
    # Hint: This function is similar to the get_doc_to_norm function in query.py
    #       but should use tfidf instead of term frequency

    # That is, the norm should be tf-idf instead of tf, comparing to the function in query.py
    # doc_norm: mapping from docid(int) to the norms(float)

    doc_norm = defaultdict(float)

    # Loop over each token
    for (term, postings) in index.items():
        # Calculate the IDF
        idf = math.log2(num_docs / (1 + doc_freq[term]))
        # Loop over each doc
        # Calculate square of norm for all docs
        for (docid, tf) in postings:
            tfidf = tf * idf
            doc_norm[docid] += tfidf ** 2

    # Take square root
    for docid in doc_norm:
        doc_norm[docid] = math.sqrt(doc_norm[docid])


    return doc_norm


def run_query(query_string, index, doc_freq, doc_norm, num_docs):
    """ Run a query on the index and return a sorted list of documents. 
    Sorted by most similar to least similar.
    Documents not returned in the sorted list are assumed to have 0 similarity.

    Args:
        query_string (str): the query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        doc_norm (dict(int : float)): a map from doc_ids to pre-computed document norms
        num_docs (int): number of documents in the corpus

    Returns:
        list(tuple(int, float)): a list of document ids and the similarity scores with the query
        sorted so that the most similar documents to the query are at the top.
    """

    # TODO: Implement this function using tfidf
    # Hint: This function is similar to the run_query function in query.py
    #       but should use tfidf instead of term frequency

    # pre-process the query string
    qt = get_query_tokens(query_string)
    query_token_counts = count_query_tokens(qt) # (unique token, term frequency)

    # calculate the norm of the query vector
    query_tfidf = defaultdict(float)
    query_norm = 0
    for (term, tf) in query_token_counts:
        # ignore term if not in index (to be comparable to doc_norm)
        # note that skipping this will not change the rank of retrieved docs
        if term not in index:
            continue
        if term in doc_freq:
            idf = math.log2(num_docs / (1 + doc_freq[term]))
            query_tfidf[term] = tf * idf
            query_norm += (tf * idf) ** 2

    query_norm = math.sqrt(query_norm)

    # calculate cosine similarity for all relevant documents
    doc_to_score = defaultdict(float)
    for term, tfidf_query in query_tfidf.items():
        # Ignore query terms not in the index
        if term not in index:
            continue
        # add to similarity for documents that contain current query word
        for docid, tf in index[term]:
                idf = math.log2(num_docs / (1 + doc_freq[term]))
                tfidf_doc = tf * idf
                doc_to_score[docid] += (tfidf_query * tfidf_doc) / (doc_norm[docid] * query_norm)

    sorted_docs = sorted(doc_to_score.items(), key=lambda x: -x[1])

    return sorted_docs


if __name__ == '__main__':
    queries = [
        # 'Is nuclear power plant eco-friendly?',
        'How to stay safe during severe weather?',
    ]
    query_main(queries=queries, query_func=run_query, doc_norm_func=get_doc_to_norm)
    

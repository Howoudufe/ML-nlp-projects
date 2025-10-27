import pickle
from string_processing import (
    process_tokens,
    tokenize_text,
)


def intersect_query(doc_list1, doc_list2):
    # TODO: you might like to use a function like this in your run_boolean_query implementation
    # for full marks this should be the O(n + m) intersection algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks

    # doc_list1, doc_list2 are list of document IDs. List[int]
    
    res = []
    i, j = 0, 0
    # Loop over two lists
    while i < len(doc_list1) and j < len(doc_list2):
        if doc_list1[i] == doc_list2[j]: # The shared doc found
            res.append(doc_list1[i])
            i += 1
            j += 1
        elif doc_list1[i] < doc_list2[j]:
            i += 1
        else:
            j += 1

    return res


def union_query(doc_list1, doc_list2):
    # TODO: you might like to use a function like this in your run_boolean_query implementation
    # for full marks this should be the O(n + m) union algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks

    res = []
    i, j = 0, 0
    while i < len(doc_list1) and j < len(doc_list2):
        if doc_list1[i] == doc_list2[j]:
            res.append(doc_list1[i])
            i += 1
            j += 1
        elif doc_list1[i] < doc_list2[j]:
            res.append(doc_list1[i])
            i += 1
        else:
            res.append(doc_list2[j])
            j += 1

    # Add remaining elements, if exist
    while i < len(doc_list1):
        res.append(doc_list1[i])
        i += 1
    while j < len(doc_list2):
        res.append(doc_list2[j])
        j += 1


    return res


def run_boolean_query(query_string, index):
    """Runs a boolean query using the index.

    Args:
        query_string (str): boolean query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists

    Returns:
        list(int): a list of doc_ids which are relevant to the query
    """

    # TODO: implement this function

    tokens = query_string.split()
    # The list to store the ID of relevant docs
    relevant_docs = []
    i = 0

    while i < len(tokens):
        term = tokens[i]
        # term = tokens[i].lower()

        if term.upper() == "AND":
            i += 1
            next_term = tokens[i]
            if next_term in index:
                next_docs = [doc_id for doc_id, _ in index[next_term]]
                # intersect with next term
                relevant_docs = intersect_query(relevant_docs, next_docs)
        elif term.upper() == "OR":
            i += 1
            next_term = tokens[i]
            if next_term in index:
                next_docs = [doc_id for doc_id, _ in index[next_term]]
                relevant_docs = union_query(relevant_docs, next_docs)
        else:
            if term in index:
                term_docs = [doc_id for doc_id, _ in index[term]]
                if not relevant_docs:
                    # Initializes relevant_docs using the document IDs of the first term
                    relevant_docs = term_docs
                else:
                    relevant_docs = union_query(relevant_docs, term_docs)
        # Move to next token
        i += 1
      

    return relevant_docs


if __name__ == '__main__':
    # load the stored index
    (index, doc_freq, doc_ids, num_docs) = pickle.load(open("stored_index.pkl", "rb"))

    print("Index length:", len(index))
    if len(index) != 808777:
        print("Warning: the length of the index looks wrong.")
        print("Make sure you are using `process_tokens_original` when you build the index.")
        raise Exception()

    # the list of queries asked for in the assignment text
    # queries = [
    #     "Workbooks",
    #     "Australasia OR Airbase",
    #     "Warm AND WELCOMING",
    #     "Global AND SPACE AND economies",
    #     "SCIENCE OR technology AND advancement AND PLATFORM",
    #     "Wireless OR Communication AND channels OR SENSORY AND INTELLIGENCE",
    # ]

    queries = [
        "Workbooks",
        "workbooks", # test
        "physical AND therapists",
        "SCIENCE OR technology AND advancement AND PLATFORM",
    ]

    # Run each queries and print results
    # for query in queries:
    #     relevant_docs = run_boolean_query(query, index)
    #     # print(f"Query: {query} \nResults: {[doc_ids[key] for key in relevant_docs]}\n")
    #     print(f"Query: {query} \nResults: {[doc_ids[key] for key in relevant_docs if key in doc_ids]}\n")

    # run each of the queries and print the result
    ids_to_doc = {docid: path for (path, docid) in doc_ids.items()}
    for query_string in queries:
        print("Query: ",query_string)
        print("Results: ")
        doc_list = run_boolean_query(query_string, index) # List of Doc IDs 
        # Converts the doc IDs to their file paths and sorts the results
        res = sorted([ids_to_doc[docid] for docid in doc_list])
        for path in res:
            print(path)


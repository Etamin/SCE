from tree_sitter_language_pack import get_parser
from collections import Counter
import math
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from contextlib import ExitStack
from nltk.translate.bleu_score import SmoothingFunction


def _normalize_vector(v):
    """
    Normalizes a NumPy vector to represent a probability distribution.
    Ensures non-negative values and sums to 1.
    Adds a small epsilon to prevent log(0) issues.
    """
    if np.any(v < 0):
        raise ValueError("Vector elements must be non-negative for probability distribution.")
    
    # Add a small epsilon to handle potential zeros for logarithm calculations later
    # This prevents division by zero or log(0) which results in -inf
    epsilon = 1e-10
    v = v + epsilon
    
    # Normalize to sum to 1
    normalized_v = v / np.sum(v)
    return normalized_v

def kullback_leibler_divergence(p, q):
    """
    Calculates the Kullback-Leibler Divergence D_KL(P || Q) between two
    normalized probability distributions P and Q.

    Args:
        p (np.array): First normalized probability distribution.
        q (np.array): Second normalized probability distribution.

    Returns:
        float: The KL-Divergence value.
    """
    # Ensure inputs are numpy arrays
    p = np.asarray(p)
    q = np.asarray(q)

    # Check for same length
    if p.shape != q.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Check for normalization (sum to ~1) - good practice but _normalize_vector handles it if not
    if not np.isclose(np.sum(p), 1.0) or not np.isclose(np.sum(q), 1.0):
        print("Warning: Input distributions are not normalized. Normalizing internally.")
        p = _normalize_vector(p)
        q = _normalize_vector(q)
        
    # Calculate D_KL(P || Q)
    # The term p * log(p / q) becomes 0 if p is 0, which is handled correctly by numpy's log
    # provided q is not zero (which our epsilon in _normalize_vector helps with).
    kl_div = np.sum(p * np.log(p / q))
    return kl_div

def jensen_shannon_divergence(p, q):
    """
    Calculates the Jensen-Shannon Divergence JSD(P || Q) between two
    NumPy vectors representing probability distributions.

    Args:
        p (np.array): First NumPy vector. Will be normalized internally.
        q (np.array): Second NumPy vector. Will be normalized internally.

    Returns:
        float: The JSD value.
    """
    # Ensure inputs are numpy arrays
    p = np.asarray(p)
    q = np.asarray(q)

    # Check for same length
    if p.shape != q.shape:
        raise ValueError("Input arrays must have the same shape.")

    # 1. Normalize the vectors
    p_norm = _normalize_vector(p.copy()) # Use .copy() to avoid modifying original array if it's already np array
    q_norm = _normalize_vector(q.copy())

    # 2. Calculate the average probability distribution M
    m = 0.5 * (p_norm + q_norm)

    # 3. Calculate KL Divergence from P to M and Q to M
    kl_pm = kullback_leibler_divergence(p_norm, m)
    kl_qm = kullback_leibler_divergence(q_norm, m)

    # 4. Combine KLDs to get JSD
    js_div = 0.5 * (kl_pm + kl_qm)
    return js_div

def extract_subtrees(root_node, max_depth):
    """
    Traverse AST up to max_depth and collect subtree tuples.
    Returns a Counter mapping subtree_repr -> count.
    """
    counter = Counter()

    def dfs(node, depth):
        if depth > max_depth:
            return
        # Represent subtree by (node_type, tuple(child_types...))
        repr_tuple = (node.type, tuple(child.type for child in node.children))
        counter[repr_tuple] += 1
        for child in node.children:
            dfs(child, depth + 1)

    dfs(root_node, 1)
    return counter

def extract_subtrees_withvalue(root_node, source_code: bytes, max_depth: int):
    """
    Traverse an AST up to max_depth and collect subtree representations.
    For internal nodes, represent as (node_type, tuple(child_types...)).
    For leaf nodes (child_count==0), append the actual source text: 
      (node_type, node_text).
    Returns a Counter mapping repr_tuple -> count.
    
    Args:
      root_node: a tree_sitter.Node for the AST root.
      source_code: the original source as a bytes object (UTF-8).
      max_depth: maximum subtree depth to extract (root is depth=1).
    """
    counter = Counter()

    def dfs(node, depth):
        # If we've exceeded max_depth, stop recursing.
        if depth > max_depth:
            return
        
        # If this node is a leaf, capture its text (slice source by byte range)
        if node.child_count == 0:
            # Extract the exact snippet this leaf covers
            start, end = node.start_byte, node.end_byte
            leaf_text = source_code[start:end].decode('utf8', errors='ignore')
            repr_tuple = (node.type, (leaf_text,))
            counter[repr_tuple] += 1
            return
        
        # Otherwise, internal node: represent by (type, tuple(child types...))
        child_types = tuple(child.type for child in node.children)
        repr_tuple = (node.type, child_types)
        counter[repr_tuple] += 1

        # Recurse on each child
        for child in node.children:
            dfs(child, depth + 1)

    dfs(root_node, 1)
    return counter


def compute_sce_norm_struct(lang,sql_A, sql_B, max_depth=30, eps=1e-10):
    """
    Compute normalized SCE via dividing by log2(|U|).
    Inputs: code strings sql_A, sql_B.
    """
    parser = get_parser(lang)
    root_A = parser.parse(bytes(sql_A, "utf8")).root_node
    root_B = parser.parse(bytes(sql_B, "utf8")).root_node

    # 1. Extract subtree frequencies up to max_depth
    sql_A_byte = sql_A.encode('utf8')
    sql_B_byte = sql_B.encode('utf8')
    freqA = extract_subtrees(root_node=root_A, max_depth=max_depth)
    freqB = extract_subtrees(root_node=root_B, max_depth=max_depth)

    # 2. Build support U = keys in either freqA or freqB
    U = set(freqA) | set(freqB)
    nA = sum(freqA.values())
    nB = sum(freqB.values())

    # 3. Compute probability distributions P and Q over U
    P = []
    Q = []
    for s in U:
        p = freqA[s] / nA if freqA[s] > 0 else eps
        q = freqB[s] / nB if freqB[s] > 0 else eps
        P.append(p)
        Q.append(q)

    # 4. Compute cross-entropy H(P, Q)
    cross_entropy = -sum(p * math.log2(q) for _, p, q in zip(U,P, Q))

    U_len=len(U)
    # 5. Compute entropy H(P)
    entropy_P = -sum(p * math.log2(p) for p in P)
    entropy_Q = -sum(q * math.log2(q) for q in Q)
    entropy_Max= max(entropy_P,entropy_Q)
    
    # 6. Compute normalized SCE
    # _entropy if cross_entropy > 0 else 0.0
    return sce_norm

def compute_cte_jsd_struct(lang,sql_A, sql_B, max_depth=30, eps=1e-10):
    """
    Compute JSD (bounded [0,1]).
    """
    parser = get_parser(lang)
    root_A = parser.parse(bytes(sql_A, "utf8")).root_node
    root_B = parser.parse(bytes(sql_B, "utf8")).root_node
    freqA = extract_subtrees(root_node=root_A,max_depth=max_depth)
    freqB = extract_subtrees(root_node=root_B,max_depth=max_depth)

    U = set(freqA) | set(freqB)
    nA = sum(freqA.values())
    nB = sum(freqB.values())
    # print(freqA)
    # print(freqB)
    # Build probability vectors aligned on sorted(U) for reproducibility
    labels = sorted(U)
    P_vec = np.array([freqA[s]/nA for s in labels], dtype=float)
    Q_vec = np.array([freqB.get(s, 0)/nB for s in labels], dtype=float)



    # Smooth zero entries to avoid issues; ensure Q_vec[i]>0 if P_vec[i]>0
    Q_vec = np.maximum(Q_vec, eps)

    # Renormalize Q_vec to sum to 1
    Q_vec = Q_vec / Q_vec.sum()

    # Compute JSD (base-2) => returns sqrt(JS divergence), so square it
    jsd_val = jensen_shannon_divergence(P_vec, Q_vec) 
    if np.isnan(jsd_val):
        # If JSD computation fails (e.g., due to NaN), return 1.0
        print("Warning: JSD computation resulted in NaN, returning 1.0")
        jsd_val = 1.0


    return 1-jsd_val

def compute_sce_norm_value(lang,sql_A, sql_B, max_depth=30, eps=1e-10):
    """
    Compute normalized SCE via dividing by log2(|U|).
    Inputs: code strings sql_A, sql_B.
    """
    parser = get_parser(lang)
    root_A = parser.parse(bytes(sql_A, "utf8")).root_node
    root_B = parser.parse(bytes(sql_B, "utf8")).root_node

    # 1. Extract subtree frequencies up to max_depth
    sql_A_byte = sql_A.encode('utf8')
    sql_B_byte = sql_B.encode('utf8')
    freqA = extract_subtrees_withvalue(root_node=root_A, source_code=sql_A_byte,max_depth=max_depth)
    freqB = extract_subtrees_withvalue(root_node=root_B, source_code=sql_B_byte,max_depth=max_depth)

    # 2. Build support U = keys in either freqA or freqB
    U = set(freqA) | set(freqB)
    nA = sum(freqA.values())
    nB = sum(freqB.values())

    # 3. Compute probability distributions P and Q over U
    P = []
    Q = []
    for s in U:
        p = freqA[s] / nA if freqA[s] > 0 else eps
        q = freqB[s] / nB if freqB[s] > 0 else eps
        P.append(p)
        Q.append(q)

    # 4. Compute cross-entropy H(P, Q)
    cross_entropy = -sum(p * math.log2(q) for _, p, q in zip(U,P, Q))

    U_len=len(U)
    # 5. Compute entropy H(P)
    entropy_P = -sum(p * math.log2(p) for p in P)
    entropy_Q = -sum(q * math.log2(q) for q in Q)
    entropy_Max= max(entropy_P,entropy_Q)
    
    # 6. Compute normalized SCE
    sce_norm = entropy_Q / cross_entropy if cross_entropy > 0 else 0.0

    return sce_norm

def compute_cte_jsd_value(lang,sql_A, sql_B, max_depth=30, eps=1e-10):
    """
    Compute Cross Topological JSD (bounded [0,1]).
    """
    parser = get_parser(lang)
    root_A = parser.parse(bytes(sql_A, "utf8")).root_node
    root_B = parser.parse(bytes(sql_B, "utf8")).root_node
    sql_A_byte = sql_A.encode('utf8')
    sql_B_byte = sql_B.encode('utf8')
    freqA = extract_subtrees_withvalue(root_node=root_A, source_code=sql_A_byte,max_depth=max_depth)
    freqB = extract_subtrees_withvalue(root_node=root_B, source_code=sql_B_byte,max_depth=max_depth)
    
    U = set(freqA) | set(freqB)
    nA = sum(freqA.values())
    nB = sum(freqB.values())
    
    # Build probability vectors aligned on sorted(U) for reproducibility
    labels = sorted(U)

    P_vec = np.array([freqA[s]/nA for s in labels], dtype=float)
    Q_vec = np.array([freqB.get(s, 0)/nB for s in labels], dtype=float)

    # Smooth zero entries to avoid issues; ensure Q_vec[i]>0 if P_vec[i]>0
    Q_vec = np.maximum(Q_vec, eps)
    # Renormalize Q_vec to sum to 1
    Q_vec = Q_vec / Q_vec.sum()
    

    # Compute JSD (base-2) => returns sqrt(JS divergence), so square it
    # jsd_val = jensenshannon(P_vec, Q_vec, base=2.0) ** 2
    jsd_val = jensen_shannon_divergence(P_vec, Q_vec) 
    if np.isnan(jsd_val):
        print("Warning: JSD computation resulted in NaN, returning 1.0")

        jsd_val = 1.0
    return 1-jsd_val
# SQL queries
code1 = """
if x >= 0:
    sign = "non-negative"
else:
    sign = "negative"
print(sign)
"""

# code2 = """
# def sum(x, y):
#     result = (x + y) ^ 2 - 1
#     return result
# """

code2 = """
if x >= 0:
    sign = "non-negative"
    print(sign)
else:
    sign = "negative"
    print(sign)
"""
lang = "python"
print(compute_cte_jsd_struct(lang,code1, code2))
print(compute_sce_norm_struct(lang,code1, code2))
print(compute_cte_jsd_value(lang,code1, code2))
print(compute_sce_norm_value(lang,code1, code2))

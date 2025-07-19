# src/core/traditional_compressors.py

import heapq
from collections import Counter
import bitarray

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = []
    for char, freq in freq_dict.items():
        heapq.heappush(heap, Node(char, freq))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        internal = Node(None, left.freq + right.freq)
        internal.left = left
        internal.right = right
        heapq.heappush(heap, internal)
    
    return heap[0]

def build_codes(root, current_code, codes):
    if root is None:
        return
    
    if root.char is not None:
        codes[root.char] = current_code
        return
    
    build_codes(root.left, current_code + '0', codes)
    build_codes(root.right, current_code + '1', codes)

def huffman_compress(text):
    if not text:
        return bitarray.bitarray(), {}
    
    freq_dict = Counter(text)
    root = build_huffman_tree(freq_dict)
    codes = {}
    build_codes(root, '', codes)
    ba = bitarray.bitarray()
    for char in text:
        ba.extend(codes[char])
    
    return ba, codes

def huffman_decompress(encoded, codes):
    if not encoded:
        return ""
    reverse_codes = {v: k for k, v in codes.items()}
    decoded = ""
    current_code = ""
    for bit in encoded:
        current_code += '0' if bit == 0 else '1'
        if current_code in reverse_codes:
            decoded += reverse_codes[current_code]
            current_code = ""
    
    return decoded

def lzw_compress(uncompressed):
    """Compress a string to a list of output symbols."""
    if not uncompressed:
        return []
    
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    
    if w:
        result.append(dictionary[w])
    
    return result

def lzw_decompress(compressed):
    """Decompress a list of output symbols to a string."""
    if not compressed:
        return ""
    
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    result = [chr(compressed[0])]
    w = result[0]
    for k in compressed[1:]:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError("Bad compressed k: %d" % k)
        
        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        w = entry
    return "".join(result)


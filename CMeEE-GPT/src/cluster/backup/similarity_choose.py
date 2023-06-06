from src.utils.settings import get_json_data
import jieba
from hashlib import sha1
import numpy as np

def get_shinglesInDoc(text):
    words = list(jieba.cut(text, cut_all=True))
    shinglesInDoc = set()
    for i in range(0, len(words) - 2):
        shingle = words[i] + " " + words[i+1] + " " + words[i+2]
        shingleID = int.from_bytes(sha1(shingle.encode('utf-8')).digest(), byteorder='big')
        shinglesInDoc.add(shingleID)
    return shinglesInDoc

def calculate_sim(sig1, sig2):
    intersection = 0
    union = len(sig2)
    for k in range(len(sig2)):
        if sig2[k] == sig1[k]:
            intersection += 1

    something = intersection / union
    return something

class Sim_shots:
    def __init__(self):
        self.data = get_json_data("train")
        self.numDocs = len(self.data)
        self.num_hash = 30
        docsAsShingleSets = {}
        docNames = []
        for i in range(0, self.numDocs):
            text = self.data[i]["text"]
            docID = i
            docNames.append(docID)
            docsAsShingleSets[docID] = get_shinglesInDoc(text)
        
        self.nextPrime = 4294967311
        self.a_list = [2647644122, 3144724950, 1813742201, 3889397089, 850218610, 4228854938, 3422847010, 1974054350, 1398857723, 3861451095]
        self.b_list = [2834859619, 3834190079, 3272971987, 1421011856, 1598897977, 1288507477, 1224561085, 3278591730, 1664131571, 3749293552]
        
        print( '\nGenerating MinHash signatures for all documents...')

        self.signatures = []
        for docID in docNames:
            shingleIDSet = docsAsShingleSets[docID]
            signature = self.get_signatures(shingleIDSet)
            self.signatures.append(signature)
    
    def get_signatures(self, shinglesInDoc):
        signature = []
        for i in range(self.num_hash):
            minHash = self.nextPrime
            for shingleID in shinglesInDoc:
                hashValue = (self.a_list[i] * shingleID + self.b_list[i]) % self.nextPrime
                if hashValue < minHash:
                    minHash = hashValue
            signature.append(minHash)
        return signature
    
    def get_shots(self, text, nums):
        text_shingles = get_shinglesInDoc(text)
        text_signatures = self.get_signatures(text_shingles)
        SimMatric = []
        for i in range(0, self.numDocs):
            # Get the MinHash signature for document i.
            signature1 = self.signatures[i]
            something = calculate_sim(signature1, text_signatures)
            SimMatric.append((i, something))
        SimMatric = sorted(SimMatric,key=lambda x: x[1], reverse=True)
        print(SimMatric[:100])
        return SimMatric[:nums]
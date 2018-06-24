##  TF-IDF Extractor
#UTF-8
'''

'''
import math
import collections

class Kwdsextractor():
    def __init__(self):
        self.text = {}              #text 的理想数据类型是 已经分好词的词列表的字典
        self.stopwords = set(['，','。','的','“',"”","！","？"])      #停用词
        self.wordsets = set()               #所有词的集合

        self.tfmap = collections.defaultdict(lambda: collections.defaultdict(int))
        self.tfidfmap = collections.defaultdict(lambda: collections.defaultdict(int))
        self.idfmap = collections.defaultdict(int)
        
    def tf(self,tname):	#计算tf值
        
        for word in set(self.text[tname]):
            self.tfmap[tname][word] = self.text[tname].count(word)/len(self.text[tname])

    def idf(self):      #计算idf值
        for itemkey in self.text.keys():
            self.wordsets |= set(self.text[itemkey])
        for word in self.wordsets:
            count = 0
            for member in self.text.keys():
                if word in self.text[member]:
                    count += 1
            self.idfmap[word] = abs(math.log(self.D/(count+1)))
            
    def tfidf(self):    #计算tf-idf值并存储到哈希表
        self.idf()
        for tname in self.text.keys():
            self.tf(tname)
            for word in self.wordsets.intersection(set(self.text[tname])):
                self.tfidfmap[tname][word] = self.tfmap[tname][word] * self.idfmap[word]

    def load(self,value):     #demo
        self.text = value
        self.D = len(self.text)
        self.tfidf()                                        #计算公式

    def loadSingleText(self,value,textname):
        self.text[textname] = value
        self.D += 1
        wlst = set(value)
        for word in wlst:
            count = 0
            for member in self.text.keys():
                if word in self.text[member]:
                    count += 1
            self.idfmap[word] = abs(math.log(self.D/(count+1)))
        self.tf(textname)
        for word in wlst:
            self.tfidfmap[textname][word] = self.tfmap[textname][word] * self.idfmap[word]
      
    def extract(self,textname,defaultnumofword = 10):       #提词函数，
        wordsets = set()
        wordsets = set(self.text[textname]).difference(self.stopwords)             #去停用词
        wordpairs = [(word,self.tfidfmap[textname][word]) for word in wordsets]                         #(词、tfidf值)的元组列表
        freqwordlist = [word for word,value in sorted(wordpairs,key = lambda x:x[1],reverse = True)]    #根据tdidf排序后的词
        return freqwordlist[:defaultnumofword]
        
        
                
            
            
        
    

# -*- coding: utf-8 -*-
#训练用的txt文档中，如果句子前有'*'标识，表示为人为的摘要标签
#
import sys,os
import re
import collections
from pyltp import Segmentor
from tfidf import Kwdsextractor


class Trainer():
        def __init__(self):
            #self.text用于存放文章，键名为文章名（默认为txt文件的命名）
            self.text = {}
            #特征存储
            self.features = collections.defaultdict(lambda:collections.defaultdict(float))                                      
            #统计值存储      
            self.sttstcs = collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(int))) 

            self.path = r"E:\Myprojects\Obtainer\trainsets"
            self.finalnames = []                                #txt文件名，（文档名，+后缀的文档名）的元组列表
            self.model_path = r"E:\Myprojects\LTP\ltp_data\cws.model"
                                                                #分词模型路径
            self.stopwords = set(['，','。','的','“',"”","！","？","都","有","在","将","是"])      
                                                                #停用词
            self.keytuple = ('keyrel','lenrel','posrel','titrel')
            self.labelcount = 0                                 #统计标签
            self.sentcount = 0                                  #统计句子总数

            #参数集
            self.titlekeywds = 2                                #权衡标题关键字

        def train(self):
            self.load()                         

        def load(self):
            self.getFilename()
            self.segmentor = Segmentor()
            self.segmentor.load(self.model_path)
            self.getText()
            self.loadtfidf()                                    #关键词提取器
            self.trainmain()
            paras = self.summary()
            return paras
        #
        def getFilename(self):                                  #读取txt文件名
            form = re.compile(r'([^\\]+)\.txt')             
            for root,dirs,files in os.walk(self.path):
                for filename in files:
                    self.finalnames.append((re.match(form,filename).group(1),filename))      

        def getText(self):                                      #将txt转化为分好词的列表
            for textname,filename in self.finalnames:
                text = []                                       #段落的列表
                abspath = os.path.join(self.path,filename)      #txt文件路径
                f = open(abspath)
                lines = f.readlines()                           #读取一个txt
                for line in lines:
                    if line != '\n':                            #去掉空行
                        string = line.strip().lstrip('\t')
                        sentlist = list(self.sentPreProcess(line))
                                                                #把段落以句子结束符分割
                        #print(sentlist)        #test
                        prcssdlist = []
                        for sent in sentlist:
                            prcssdlist.append(list(self.segment(sent))) 
                                                                #分好词的句子列表
                        text.append(prcssdlist)                 #得到深度为3(段-句-词)的列表
                self.text[textname] = text
                #print(self.text)               
                f.close()
        
        def sentPreProcess(self,string):                        #将字符串按句子切分
            wordlist = re.split(r"(。”|！”|？”|。|！|？)",string) #re.split分割多个结束符
            for i,item in enumerate(wordlist):
                if item in set(['。”','！”','？”','。','！','？']):
                    wordlist[i-1] += wordlist[i]                #将标点赋给上一句合并
                    del wordlist[i]
            wordlist = filter(None,wordlist)
            return wordlist

        def segment(self,string):                               #LTP分词器
            wordlistobj = self.segmentor.segment(string)
            return wordlistobj

        #part of train  
        def loadtfidf(self):                                    #tf-idf提词器的数据格式接口
            self.extractor = Kwdsextractor()
            textcopy = collections.defaultdict(list)
            for key in self.text.keys():
                for graph in self.text[key]:
                    for sent in graph:
                        textcopy[key].extend(sent)              #将text展开并拷进副本
            self.extractor.load(textcopy)
        #
        def getClusterScore(self,value,wordlist):               #获取句子的关键字簇得分
            if value:
                if value[0] == '*':
                    length = len(value) - 1
                else:
                    length = len(value)
                count = 0
                if length != 0:
                    for word in value:
                        if word in wordlist:
                            count += 1
                    score = count * count / length              #count^2/len
                #print(score)
                return (score,length)                           #返回句子的得分和长度
            else:
                return (0,0)

        def judgePosOfSent(self,text,xpos,ypos):                #判断句子的位置，有七种值
            cap = len(self.text[text][xpos])
            if cap > 1:
                if ypos == 0:
                    return self.judgePosif1stPara(xpos)+'start'
                elif ypos == cap - 1:
                    return self.judgePosif1stPara(xpos)+'end'
                else:
                    return self.judgePosif1stPara(xpos)+'others'

            if cap == 1:
                return 'unique'

        def judgePosif1stPara(self,pos):
            if pos == 0:
                return '1st'
            else:
                return ''

        def jugdeRelOfTiWords(self,value,titleWords):           #判断句子的标题相关度
            count = 0
            for word in value:
                if word in titleWords:
                    count += 1
            if count >= self.titlekeywds:
                return True
            else:
                return False

        def postProcess(self,pos,maxScore,maxLen,minLen):       #句子簇、长得分后处理
            #
            if 2/3 * maxScore < self.features[pos]['cscore'] <= maxScore:
                self.features[pos]['keyrel'] = "High"
            elif 1/3 * maxScore < self.features[pos]['cscore'] <= 2/3 * maxScore:
                self.features[pos]['keyrel'] = "Medium"
            else:
                self.features[pos]['keyrel'] = "Low"
            #
            if (minLen + 2/3 * (maxLen - minLen)) < self.features[pos]['length'] <= maxLen:
                self.features[pos]['lenrel'] = "High"
            elif (minLen + 1/3 * (maxLen - minLen)) < self.features[pos]['length'] <= (minLen + 2/3 * (maxLen - minLen)):
                self.features[pos]['lenrel'] = "Medium"
            else:
                self.features[pos]['lenrel'] = "Low"

        def labelOfY(self,pos):
            for item in self.keytuple:
                self.sttstcs['Y'][item][str(self.features[pos][item])] += 1

        def labelOfN(self,pos):
            for item in self.keytuple:
                self.sttstcs['N'][item][str(self.features[pos][item])] += 1

        def labelOfYProba(self,sentcount,labelcount):               #统计先验概率
            self.sentcount += sentcount
            self.labelcount += labelcount

        #主流程        
        def trainmain(self):
            #每一个文档的处理
            for textname in self.text.keys():
                freqwords = set(self.extractor.extract(textname))
                titleWords = set(self.segment(textname)).difference(self.stopwords)
                '''
                count = 0
                sumofScore = 0.0
                sumofLen = 0
                '''
                count = 0
                tempcscmax = 0
                templenmax = 0
                templenmin = 30
                #先对每个句子预处理
                for i in range(len(self.text[textname])):           #段落序号
                    for j in range(len(self.text[textname][i])):    #句子序号
                        pos = str(i) + '-' + str(j)                 #句子的坐标
                        temp = self.getClusterScore(self.text[textname][i][j],freqwords)
                        self.features[pos]['cscore'] = temp[0]
                        self.features[pos]['length'] = temp[1]
                        self.features[pos]['posrel'] = self.judgePosOfSent(textname,i,j)
                        self.features[pos]['titrel'] = self.jugdeRelOfTiWords(self.text[textname][i][j],titleWords)
                        if temp[0] > tempcscmax:                    #获取最大得分
                            tempcscmax = temp[0]
                        if self.features[pos]['length'] != 0:
                            if temp[1] > templenmax:                #获取最大句长
                                templenmax = temp[1]
                            if temp[1] < templenmin:                #获取最小句长
                                templenmin = temp[1]
                            count += 1


                #print(tempcscmax,templenmax,templenmin)

                #进行统计
                labelcount = 0
                for i in range(len(self.text[textname])):       
                    for j in range(len(self.text[textname][i])):
                        pos = str(i) + '-' + str(j)
                        self.postProcess(pos,tempcscmax,templenmax,templenmin)
                        if self.text[textname][i][j]:
                            if self.text[textname][i][j][0] == '*':
                                labelcount += 1
                                self.labelOfY(pos)
                                #print(self.text[textname][i][j])
                            else:
                                self.labelOfN(pos)
                self.labelOfYProba(count,labelcount) 


        def summary(self):
            probOfY = self.labelcount / self.sentcount              #先验概率
            sumofScrY = 0
            sumofScrN = 0
            sumofLenY = 0
            sumofLenN = 0
            sumOfPosY = 0
            sumOfPosN = 0
            for item in ('High','Medium','Low'):
                sumofScrY += self.sttstcs['Y']['keyrel'][item]
                sumofScrN += self.sttstcs['N']['keyrel'][item]
                sumofLenY += self.sttstcs['Y']['lenrel'][item]
                sumofLenN += self.sttstcs['N']['lenrel'][item]

            for item in ('start','end','others','1ststart','1stend','1stothers','unique'):
                sumOfPosY += self.sttstcs['Y']['posrel'][item]
                sumOfPosN += self.sttstcs['N']['posrel'][item]

            # Y 
            probKeyOfYh = self.sttstcs['Y']['keyrel']['High'] / sumofScrY
            probKeyOfYm = self.sttstcs['Y']['keyrel']['Medium'] / sumofScrY
            probKeyOfYl = self.sttstcs['Y']['keyrel']['Low'] / sumofScrY
            probLenOfYh = self.sttstcs['Y']['lenrel']['High'] / sumofLenY
            probLenOfYm = self.sttstcs['Y']['lenrel']['Medium'] / sumofLenY
            probLenOfYl = self.sttstcs['Y']['lenrel']['Low'] / sumofLenY
            probTitOfYt = self.sttstcs['Y']['titrel']['True'] / (self.sttstcs['Y']['titrel']['True'] + self.sttstcs['Y']['titrel']['False'])
            probPosOfYs1 = self.sttstcs['Y']['posrel']['1ststart'] / sumOfPosY
            probPosOfYe1 = self.sttstcs['Y']['posrel']['1stend'] / sumOfPosY 
            probPosOfYo1 = self.sttstcs['Y']['posrel']['1stothers'] / sumOfPosY
            probPosOfYs = self.sttstcs['Y']['posrel']['start'] / sumOfPosY
            probPosOfYe = self.sttstcs['Y']['posrel']['end'] / sumOfPosY 
            probPosOfYo = self.sttstcs['Y']['posrel']['others'] / sumOfPosY
            probPosOfYu = self.sttstcs['Y']['posrel']['unique'] / sumOfPosY
            
            # N
            probKeyOfNh = self.sttstcs['N']['keyrel']['High'] / sumofScrN
            probKeyOfNm = self.sttstcs['N']['keyrel']['Medium'] / sumofScrN
            probKeyOfNl = self.sttstcs['N']['keyrel']['Low'] / sumofScrN
            probLenOfNh = self.sttstcs['N']['lenrel']['High'] / sumofLenN 
            probLenOfNm = self.sttstcs['N']['lenrel']['Medium'] / sumofLenN 
            probLenOfNl = self.sttstcs['N']['lenrel']['Low'] / sumofLenN 
            probTitOfNt = self.sttstcs['N']['titrel']['True'] / (self.sttstcs['N']['titrel']['True'] + self.sttstcs['N']['titrel']['False'])            
            probPosOfNs1 = self.sttstcs['N']['posrel']['1ststart'] / sumOfPosN
            probPosOfNe1 = self.sttstcs['N']['posrel']['1stend'] / sumOfPosN 
            probPosOfNo1 = self.sttstcs['N']['posrel']['1stothers'] / sumOfPosN
            probPosOfNs = self.sttstcs['N']['posrel']['start'] / sumOfPosN
            probPosOfNe = self.sttstcs['N']['posrel']['end'] / sumOfPosN 
            probPosOfNo = self.sttstcs['N']['posrel']['others'] / sumOfPosN
            probPosOfNu = self.sttstcs['N']['posrel']['unique'] / sumOfPosN

            return [
            probOfY,                            #先验概率
            probKeyOfYh,                        #1
            probKeyOfYm,
            probKeyOfYl,
            probLenOfYh,
            probLenOfYm,
            probLenOfYl,
            probTitOfYt,
            probPosOfYs1,
            probPosOfYe1,
            probPosOfYo1,
            probPosOfYs,
            probPosOfYe,
            probPosOfYo,
            probPosOfYu,                        #14

            probKeyOfNh,                        #15
            probKeyOfNm,
            probKeyOfNl,
            probLenOfNh,
            probLenOfNm,
            probLenOfNl,
            probTitOfNt,
            probPosOfNs1,
            probPosOfNe1,
            probPosOfNo1,
            probPosOfNs,
            probPosOfNe,
            probPosOfNo,
            probPosOfNu                         #28
            ]



class Parser(Trainer):
    def __init__(self):
        #用于存放文本列表
        #self.text
        #用于存放已标签的句子
        self.abstract = []

        self.rootpath = r"E:\Myprojects\Obtainer"
        self.model_path = r"E:\Myprojects\LTP\ltp_data\cws.model"
                                                                #分词模型路径
        self.stopwords = set(['，','。','的','“',"”","！","？","都","有","在","将","是"])      
                                                                #停用词
        self.keytuple = ('keyrel','lenrel','posrel','titrel')
        self.features = collections.defaultdict(lambda: collections.defaultdict(float))
        self.sentcount = 0                                      #统计句子总数

        #参数集
        self.titlekeywds = 2                                    #权衡标题关键字
        self.init()

    def getParameters(self,value):                              #获取参数
        """
        value 的格式是
        [probOfY,       # 0 
        probKeyOfYt,probLenOfYt,probTitOfYt,probPosOfYs1,probPosOfYe1,probPosOfYo1,probPosOfYs,probPosOfYe,probPosOfYo,probPosOfYu,     #1-10
        probKeyOfNt,probLenOfNt,probTitOfNt,probPosOfNs1,probPosOfNe1,probPosOfNo1,probPosOfNs1,probPosOfNe,probPosOfNo,probPosOfNu]    #11-20
        共21项
        """
        self.parameters = tuple(value)

    def init(self):
        trainer = Trainer()
        self.getParameters(trainer.load())
        self.segmentor = Segmentor()
        self.segmentor.load(self.model_path)
        self.text = trainer.text.copy()
        self.loadtfidf()

        #概率参数表  --------------
        self.indexYmap = {
            'keyrel':{  'High': self.parameters[1], 
                        'Medium': self.parameters[2],
                        'Low':self.parameters[3]},
            'lenrel':{  'High': self.parameters[4], 
                        'Medium': self.parameters[5],
                        'Low':self.parameters[6]},
            'titrel':{True: self.parameters[7] , False: 1 - self.parameters[7]},
            'posrel':{
                '1ststart':self.parameters[8],
                '1stend':self.parameters[9],
                '1stothers':self.parameters[10],
                'start':self.parameters[11],
                'end':self.parameters[12],
                'others':self.parameters[13],
                'unique':self.parameters[14]}
        }
        self.indexNmap = {
        'keyrel':{  'High': self.parameters[15], 
                    'Medium': self.parameters[16],
                    'Low':self.parameters[17]},
        'lenrel':{  'High': self.parameters[18], 
                    'Medium': self.parameters[19],
                    'Low':self.parameters[20]},
        'titrel':{True: self.parameters[21] , False: 1 - self.parameters[21]},
        'posrel':{
        '1ststart':self.parameters[22],
        '1stend':self.parameters[23],
        '1stothers':self.parameters[24],
        'start':self.parameters[25],
        'end':self.parameters[26],
        'others':self.parameters[27],
        'unique':self.parameters[28]}
        }

    def loadtext(self,textname):
        file = "testsets"
        filename = textname + '.txt'
        filepath = os.path.join(self.rootpath,file)
        abspath = os.path.join(filepath,filename)
        f = open(abspath)
        lines = f.readlines()
        text = []
        for line in lines:
            if line:                                            #去掉空行
                sentlist = list(self.sentPreProcess(line))
                                                                #把段落以句子结束符分割
                #print(sentlist)        #test
                prcssdlist = []
                for sent in sentlist:
                    prcssdlist.append(list(self.segment(sent))) 
                                                                #分好词的句子列表
                text.append(prcssdlist)                         #得到深度为3(段-句-词)的列表       
        f.close()
        return text

    #
    def unfoldText(self,value):                                 #tf-idf提词器的数据格式接口
        textcopy = []
        for graph in value:
            for sent in graph:
                textcopy.extend(sent)                           #将text展开并拷进副本
        return textcopy
    #
    def extract(self,textname):                                 #提取摘要

        self.text[textname] = self.loadtext(textname)           #按数据格式读取文档

        titleWords = set(self.segment(textname)).difference(self.stopwords)
        self.extractor.loadSingleText(self.unfoldText(self.text[textname]),textname)
        freqwords = set(self.extractor.extract(textname))

        tempcscmax = 0
        templenmax = 0
        templenmin = 30
        count = 0
        #先对每个句子预处理
        for i in range(len(self.text[textname])):               #段落序号
            for j in range(len(self.text[textname][i])):        #句子序号
                pos = str(i) + '-' + str(j)                     #句子的坐标
                temp = self.getClusterScore(self.text[textname][i][j],freqwords)
                self.features[pos]['cscore'] = temp[0]
                self.features[pos]['length'] = temp[1]
                self.features[pos]['posrel'] = self.judgePosOfSent(textname,i,j)
                self.features[pos]['titrel'] = self.jugdeRelOfTiWords(self.text[textname][i][j],titleWords)
                if temp[0] > tempcscmax:                        #获取最大得分
                    tempcscmax = temp[0]
                if self.features[pos]['length'] != 0:
                    if temp[1] > templenmax:                    #获取最大句长
                        templenmax = temp[1]
                    if temp[1] < templenmin:                    #获取最小句长
                        templenmin = temp[1]
                    count += 1


        #逐句计算
        output = []
        for i in range(len(self.text[textname])):       
            for j in range(len(self.text[textname][i])):
                pos = str(i) + '-' + str(j)
                self.postProcess(pos,tempcscmax,templenmax,templenmin)
                if self.computeProb(pos):
                    output.append(''.join(self.text[textname][i][j]))
        #print(output)
        outputString = ''.join(output)
        self.outputFile(textname,outputString)
        print("Done!")


    def computeProb(self,pos):                                  #根据概率判断是否为摘要句
        probYOfFeature = 1
        probNOfFeature = 1
        for item in self.keytuple:
            #if item == 'keyrel':
            #print((self.indexYmap[item][self.features[pos][item]],self.indexNmap[item][self.features[pos][item]]))
            probYOfFeature *= self.indexYmap[item][self.features[pos][item]]
            probNOfFeature *= self.indexNmap[item][self.features[pos][item]]
        #print("####")

        probOfY = self.parameters[0] * probYOfFeature
        probOfN = (1 -self.parameters[0]) * probNOfFeature
        #print(probOfY,probOfN)
        if probOfY > probOfN:
            #print("Yes!")
            return True
        else:
            return False

    def outputFile(self,textname,string):                       #输出txt文件
        file = "testsets"
        filename = textname + '-abstract.txt'
        filepath = os.path.join(self.rootpath,file)
        abspath = os.path.join(filepath,filename)
        f = open(abspath,'w')
        f.write('    ')
        f.write(string)
        f.close()


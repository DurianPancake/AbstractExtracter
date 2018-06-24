# AbstractExtractor
An abstract extractor based on Naive Bayes model and Ltp segmentor
使用须知：
1、由于使用了LTP的分词器，所以需要用到ltp已经训练好的模型，cws.model文件
  其路径需要修改self.model_path
2、Test用的训练材料见rar,其路径为self.path。由于是监督式学习，训练材料的标签为在句前加上‘*’符号。

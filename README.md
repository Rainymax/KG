使用 python 解析 zhishime.json 文件，并将解析出的 dict 保存为文件

利用合适的方法实现头实体的检索模块，例如正向最大匹配或命名实体识别

根据头实体检索相应的关系和尾实体，使用预训练模型获得的词向量计算问题和每一个关系的余弦相似度

根据预测的关系取出相应的答案，并计算答案和正确答案的余弦相似度，以及在知识图谱中找到的 relation 与问题中的 relation 的余弦相似度
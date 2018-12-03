
#coding: utf-8
import os
import time
import random
import jieba
import nltk
import sklearn
import xlrd
from xlutils.copy import copy
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


# 从文件中获取词典
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'rb') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set

#从训练集文件中得到词典
def GetTrainSet(file_path):
    train_data_list = [] #词集
    train_class_list = [] #类别集，本程序只设计两种情况——自动化专利（1）和非自动化专利（0）

    worksheet = xlrd.open_workbook(file_path)  # 打开excel文件
    sheet_names = worksheet.sheet_names()  # 获取excel中所有工作表名
    sheet1 = worksheet.sheets()[0] #获取第一个sheet
    nrows = sheet1.nrows  # 表示获取Sheet1中所有行

    for i in range(1, nrows):
        train_class_list.append(sheet1.row(i)[24].value)
        ## --------------------------------------------------------------------------------
        ## jieba分词
        # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
        content = sheet1.row(i)[19].value
        content = content.replace("\n", "")
        content = content.replace(" ", "")
        word_cut = jieba.cut(content, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
        word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
        train_data_list.append(word_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list))[0] #如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。

    return train_data_list, train_class_list, all_words_list

def GetTestSet(file_path):
    test_data_list = [] #词集

    worksheet = xlrd.open_workbook(file_path)  # 打开excel文件
    sheet_names = worksheet.sheet_names()  # 获取excel中所有工作表名
    sheet1 = worksheet.sheets()[0] #获取第一个sheet
    nrows = sheet1.nrows  # 表示获取Sheet1中所有行

    for i in range(1, nrows):
        ## --------------------------------------------------------------------------------
        ## jieba分词
        # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
        content = sheet1.row(i)[4].value
        content = content.replace("\n", "")
        content = content.replace(" ", "")
        word_cut = jieba.cut(content, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
        word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
        test_data_list.append(word_list)

    return test_data_list


#从文件中获取待分类专利集
def GetPatentSet(file_path):
    test_data_list = [] #词集

    worksheet = xlrd.open_workbook(file_path)  # 打开excel文件
    sheet_names = worksheet.sheet_names()  # 获取excel中所有工作表名
    sheet1 = worksheet.sheets()[0] #获取第一个sheet
    nrows = sheet1.nrows  # 表示获取Sheet1中所有行

    for i in range(1, nrows):
        ## --------------------------------------------------------------------------------
        ## jieba分词
        # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
        content = sheet1.row(i)[19].value
        content = content.replace("\n", "")
        content = content.replace(" ", "")
        word_cut = jieba.cut(content, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
        word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
        test_data_list.append(word_list)

    return test_data_list

def TextProcessing3(file_path):
    data_list = [] #词集
    class_list = [] #1/0

    worksheet = xlrd.open_workbook(file_path)  # 打开excel文件
    sheet_names = worksheet.sheet_names()  # 获取excel中所有工作表名
    sheet1 = worksheet.sheets()[0] #获取第一个sheet
    nrows = sheet1.nrows  # 表示获取Sheet1中所有行

    for i in range(1, nrows):
        class_list.append(sheet1.row(i)[5].value)
        ## --------------------------------------------------------------------------------
        ## jieba分词
        # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
        content = sheet1.row(i)[3].value
        content = content.replace("\n", "")
        content = content.replace(" ", "")
        word_cut = jieba.cut(content, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
        word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
        data_list.append(word_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list))[0] #如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。

    return all_words_list, data_list, class_list

def TextProcessing2(file_path, test_size=0.2):
    data_list = [] #词集
    class_list = [] #文件集

    worksheet = xlrd.open_workbook(file_path)  # 打开excel文件
    sheet_names = worksheet.sheet_names()  # 获取excel中所有工作表名
    sheet1 = worksheet.sheets()[0] #获取第一个sheet
    nrows = sheet1.nrows  # 表示获取Sheet1中所有行

    for i in range(1, nrows):
        class_list.append(sheet1.row(i)[5].value)
        ## --------------------------------------------------------------------------------
        ## jieba分词
        # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
        content = sheet1.row(i)[3].value
        content = content.replace("\n", "")
        content = content.replace(" ", "")
        word_cut = jieba.cut(content, cut_all=False)  # 精确模式，返回的结构是一个可迭代的genertor
        word_list = list(word_cut)  # genertor转化为list，每个词unicode格式
        data_list.append(word_list)

    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    data_class_list = list(zip(data_list, class_list)) #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    random.shuffle(data_class_list) #随机列表排序
    #（此处训练集为后80%，测试集为前20%）
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list))[0] #如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def TextProcessing(folder_path, test_size=0.2):
    folder_list = os.listdir(folder_path) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
    data_list = [] #词集
    class_list = [] #文件集

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 类内循环
        j = 1
        for file in files:
            if j > 100: # 每类text样本数最多100
                break
            with open(os.path.join(new_folder_path, file), 'rb') as fp:
               raw = fp.read()
            # print raw
            ## --------------------------------------------------------------------------------
            ## jieba分词
            # jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数，不支持windows
            word_cut = jieba.cut(raw, cut_all=False) # 精确模式，返回的结构是一个可迭代的genertor
            word_list = list(word_cut) # genertor转化为list，每个词unicode格式
            # jieba.disable_parallel() # 关闭并行分词模式
            # print word_list
            ## --------------------------------------------------------------------------------
            data_list.append(word_list)
            class_list.append(folder.decode('utf-8'))
            j += 1

    ## 划分训练集和测试集
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    data_class_list = list(zip(data_list, class_list)) #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    random.shuffle(data_class_list) #随机列表排序
    #（此处训练集为后80%，测试集为前20%）
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict:
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list))[0] #如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000: # feature_words的维度1000
            break
        # print all_words_list[t]
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words] #特征词汇中，如果出现在文本中，为1 ，否则为0，features的长度等于feature_words
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list = [], flag='sklearn'):
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        # print ''
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        test_predict = classifier.predict(test_feature_list)
        print(test_predict)
        # test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    # return test_accuracy
    return test_predict


if __name__ == '__main__':
    # print ("start")
    #
    # ## 文本预处理
    # file_path = './专利.xlsx'
    # all_words_list, train_data_list, train_class_list = TextProcessing3(file_path)
    #
    # #测试对比
    # # print("实际值：", end='')
    # # print(test_class_list)
    #
    # # 生成stopwords_set
    # stopwords_file = './stopwords_cn.txt'
    # stopwords_set = MakeWordsSet(stopwords_file)
    #
    # #获取待分类专利
    # test_data_list, test_class_list, test_all_word_list = GetTrainSet('./训练文件.xlsx')
    #
    # ## 文本特征提取和分类
    # # flag = 'nltk'
    # flag = 'sklearn'
    # deleteNs = range(0, 1000, 20)
    # # deleteNs = range(0, 1, 1)
    # test_accuracy_list = []
    # for deleteN in deleteNs:
    #     print (deleteN, end=' ')
    #     # feature_words = words_dict(all_words_list, deleteN)
    #     feature_words = words_dict(all_words_list, deleteN, stopwords_set)
    #     train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, flag)
    #     test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
    #     test_accuracy_list.append(test_accuracy)
    # print (test_accuracy_list)
    #
    # # 结果评价
    # plt.figure()
    # plt.plot(deleteNs, test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.savefig('result.png')
    #
    # feature_words2 = words_dict(all_words_list, 0, stopwords_set)
    #
    #
    # print ("finished")



    print("start")
    file_path = './专利.xlsx'
    all_words_list, data_list, class_list = TextProcessing3(file_path)
    print(class_list)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    folder_path = "./testFold"
    file_list = os.listdir(folder_path)
    for file in file_list:
        print(file)
        # fp = os.path.join(folder_path, file)
        fp = folder_path + "/" + file
        print(fp)
        # 获取待分类专利
        patent_set = GetPatentSet(fp)

        ## 文本特征提取和分类
        # flag = 'nltk'
        flag = 'sklearn'
        # feature_words = words_dict(all_words_list, deleteN)
        feature_words = words_dict(all_words_list, 0, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(data_list, patent_set, feature_words, flag)
        test_predict = TextClassifier(train_feature_list, test_feature_list, class_list, [], flag)

        # 结果写入文件
        book = xlrd.open_workbook(fp)
        wb = copy(book)
        ws = wb.get_sheet(0)
        for i in range(1, len(test_predict)):
            ws.write(i, 34, test_predict[i - 1])
        wb.save(fp)


    # #获取待分类专利
    # patent_set = GetPatentSet('./1-5000.xls')
    #
    # ## 文本特征提取和分类
    # # flag = 'nltk'
    # flag = 'sklearn'
    # # feature_words = words_dict(all_words_list, deleteN)
    # feature_words = words_dict(all_words_list, 0, stopwords_set)
    # train_feature_list, test_feature_list = TextFeatures(data_list, patent_set, feature_words, flag)
    # test_predict = TextClassifier(train_feature_list, test_feature_list, class_list, [], flag)
    #
    # #结果写入文件
    # book = xlrd.open_workbook('./1-5000.xls')
    # wb = copy(book)
    # ws = wb.get_sheet(0)
    # for i in range(1, len(test_predict)):
    #     ws.write(i, 34, test_predict[i-1])
    # wb.save('./1-5000.xls')

    print("finished")


#自测
    # print ("start")
    #
    # ## 文本预处理
    # file_path = './专利.xlsx'
    # all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing2(file_path, test_size=0.2)
    #
    # #测试对比
    # print("实际值：", end='')
    # print(test_class_list)
    #
    # # 生成stopwords_set
    # stopwords_file = './stopwords_cn.txt'
    # stopwords_set = MakeWordsSet(stopwords_file)
    #
    # #获取待分类专利
    # patent_set = GetPatentSet('./1-5000.xls')
    #
    # ## 文本特征提取和分类
    # # flag = 'nltk'
    # flag = 'sklearn'
    # # deleteNs = range(0, 1000, 20)
    # deleteNs = range(0, 1, 1)
    # test_accuracy_list = []
    # for deleteN in deleteNs:
    #     print (deleteN, end=' ')
    #     # feature_words = words_dict(all_words_list, deleteN)
    #     feature_words = words_dict(all_words_list, deleteN, stopwords_set)
    #     train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, flag)
    #     test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
    #     test_accuracy_list.append(test_accuracy)
    # print (test_accuracy_list)
    #
    # # 结果评价
    # plt.figure()
    # plt.plot(deleteNs, test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.savefig('result.png')
    #
    # feature_words2 = words_dict(all_words_list, 0, stopwords_set)
    #
    #
    # print ("finished")



    # print ("start")
    #
    # ## 文本预处理
    # folder_path = './Database/SogouC/Sample'
    # all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    #
    # # 生成stopwords_set
    # stopwords_file = './stopwords_cn.txt'
    # stopwords_set = MakeWordsSet(stopwords_file)
    #
    # ## 文本特征提取和分类
    # # flag = 'nltk'
    # flag = 'sklearn'
    # deleteNs = range(0, 1000, 20)
    # test_accuracy_list = []
    # for deleteN in deleteNs:
    #     # feature_words = words_dict(all_words_list, deleteN)
    #     feature_words = words_dict(all_words_list, deleteN, stopwords_set)
    #     train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, flag)
    #     test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
    #     test_accuracy_list.append(test_accuracy)
    # print (test_accuracy_list)
    #
    # # 结果评价
    # plt.figure()
    # plt.plot(deleteNs, test_accuracy_list)
    # plt.title('Relationship of deleteNs and test_accuracy')
    # plt.xlabel('deleteNs')
    # plt.ylabel('test_accuracy')
    # plt.savefig('result.png')
    #
    # print ("finished")
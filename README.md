#mailFilter V2.0

详细介绍请转到我博客里的这篇博文：https://fuhailin.github.io/2018/12/04/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%9E%83%E5%9C%BE%E9%82%AE%E4%BB%B6%E5%88%86%E7%B1%BB%E5%AE%9E%E6%88%98/

### 原理简介

基于贝叶斯推断的垃圾邮件过滤器。通过8000封正常邮件和8000封垃圾邮件“训练”过滤器:
解析所有邮件，提取每一个词,然后，计算每个词语在正常邮件和垃圾邮件中的出现频率。

1. 当收到一封未知邮件时，在不知道的前提下，我们假定它是垃圾邮件和正常邮件的概率各
   为50%，p(s) = p(n) = 50%

2. 解析该邮件，提取每个词，计算该词的p(s|w)，也就是受该词影响，该邮件是垃圾邮件的概率

					p(sw)             p(w|s)p(s)
		p(s|w) = -----------  =   ----------------------
					p(w)        p(s)p(w|s) + p(n)p(w|n)

3. 提取该邮件中p(s|w)最高的15个词，计算联合概率。

					p(s|w1)p(s|w2)...p(s|w15)
		p = ---------------------------------------------------------------
			p(s|w1)p(s|w2)...p(s|w15) + (1-p(s|w1))(1-p(s|w2)...(1-p(s|w15)))

4. 设定阈值 p > 0.9 :垃圾邮件
            p < 0.9 :正常邮件

> 注:如果新收到的邮件中有的词在史料库中还没出现过，就假定p(s|w) = 0.4

### 中文邮件数据集

`trec06c`是一个公开的垃圾邮件语料库，由国际文本检索会议提供，分为英文数据集（trec06p）和中文数据集（trec06c），其中所含的邮件均来源于真实邮件保留了邮件的原有格式和内容。

文件下载地址：[trec06c](https://plg.uwaterloo.ca/~gvcormac/treccorpus06/)
百度网盘备份链接：[trec06c](https://pan.baidu.com/s/1LEEN1aDbR22D_b1u07yTQw)

### 使用

1. 解压trec06c.tgz到`./input`文件夹
2. 启动一个终端，模拟邮件服务器

		cd mailFilter
		python server.py


3. 等到出现 "Waiting for clients..."，启动另一终端，模拟邮件发送端

		cd mailFilter
		python client.py

**使用的是Python 3.6版本**

### 参考资料
[http://www.ruanyifeng.com/blog/2011/08/bayesian_inference_part_two.html](http://www.ruanyifeng.com/blog/2011/08/bayesian_inference_part_two.html)
[http://en.wikipedia.org/wiki/Bayesian_spam_filtering](http://en.wikipedia.org/wiki/Bayesian_spam_filtering)

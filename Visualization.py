import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyecharts import options as opts
from pyecharts.charts import WordCloud #词云库
from pyecharts.globals import SymbolType

import re # 正则表达式库
import collections # 词频统计库
import jieba # 结巴分词
from PIL import Image # 图像处理库

# 设置画图类型与尺寸
sns.set_palette(sns.color_palette('deep'))
sns.set(rc = {'figure.figsize': (9, 5)})
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

#读取文件，查看前五行
df = pd.read_csv('bestsellers with categories.csv')
df.head()

#查看数据特征
df.shape
# 列数据信息
df.info()

# 检查缺失值
df.isnull().sum()

# 数据整体特征，describe函数参数包括数量，平均值，最小/大值，标准差，4分位数
df.describe()

#查看所有价格并排序
np.sort(df['Price'].unique())

# 上一步中价格为0的数据为免费读物，查看这些行
df[df['Price']==0]

# 查看书籍评分的分布，这里使用seaborn核密度估计并可视化kdeplot
sns.kdeplot(df['User Rating'], shade = True)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency');

#考虑到书籍类型有两种，小说Fiction与非小说Non-Fiction，我们用pie查看哪种类型更受欢迎
plt.pie(df['Genre'].value_counts(), autopct = '%1.2f%%', labels = df['Genre'].value_counts().index)
plt.title('Genre Distribution');

# 查看价格分布，这里使用histplot直方图
plt.title('Price Distribution')
sns.histplot(x = 'Price', hue = 'Genre', data = df);

# 查看评论分布，同样使用histplot直方图
plt.title('Reviews Distribution')
sns.histplot(x = 'Reviews', hue = 'Genre', data = df);

# 查看作者的上榜次数，选取TOP10
most_books = df['Author'].value_counts().head(10)
most_books

# 使用pie图可视化结果
most_books.plot(kind = 'pie', autopct = '%1.1f%%', figsize = (7, 7));

# 查看每个评分的书籍数量，使用条形图
sns.barplot(df['User Rating'].value_counts().index, df['User Rating'].value_counts())
plt.title('Number of Books Each Rating Received')
plt.xlabel('Ratings')
plt.ylabel('Counts')
plt.xticks(rotation = 45) #直接进行一个倾斜的设置

# 看一下哪一年的读者比较刁钻，评分数量按年进行比较，使用折线图
df.groupby('Year')['User Rating'].sum().plot(marker = 'o', c = 'g')
plt.title('Year vs Average Rating')
plt.xlabel('Year')
plt.ylabel('No. of Ratings')

# 看一看哪一年的评价最高，使用折线图
df.groupby('Year')['Reviews'].sum().plot(marker = 'o', c = 'g')
plt.title('Year Vs Average Reviews')
plt.xlabel('Year')
plt.ylabel('No. of Reviews');

#看一看大家这个对书籍价格有没有更容易接受，价格按年分布
df.groupby('Year')['Price'].sum().plot(marker = 'o', c = 'g')
plt.title('Variation of Price Over the Years')
plt.xlabel('Year')
plt.ylabel('Price');

#查看评论数最多的书，使用横向条形图
top_reviews = df.nlargest(20, ['Reviews'])
sns.barplot(top_reviews['Reviews'], top_reviews['Name']);

#查看评论数最少的书，同样使用横向条形图
lowest_reviews = df.nsmallest(10, ['Reviews'])
sns.barplot(lowest_reviews['Reviews'], lowest_reviews['Name']);

#查看评价最差的前十本书
worst = df.sort_values('User Rating').head(10)
worst

# 将他们可视化，使用横向条形图，只显示7本是因为书名有重复（不同年份出版）
plt.title('Worst Rated Books')
sns.barplot(y = worst['Name'], x = worst['User Rating']);

# 来看看最贵的那一批书，一本书80刀是认真的？
plt.title('Expensive books in Amazon bestseller list')
top_expensive = df.drop(df[df['Price'] < 1].index).sort_values('Price', ascending = False).head(10)
sns.barplot(y = top_expensive['Name'], x = top_expensive['Price']);

# 最便宜的一批书，当然要排除价格为0的读物
plt.title('Cheapest books in Amazon bestseller list')
cheapest = df[-df['Price'].isin([0])].sort_values('Price').head(10)
sns.barplot(y = cheapest['Name'], x = cheapest['Price']);

#当然我们最关心免费读物中哪些是最值得购买的，按读者评论排序显示
df.drop(df[df['Price'] > 0].index).sort_values('User Rating', ascending = False).head(10)

# 对于最畅销的作家，我们可以用词密度图来展示他们的受欢迎程度
df = pd.read_csv('bestsellers with categories.csv',usecols=[1])
#将名字列输出到txt文件
df.to_csv('result.txt',index=False,sep=' ',encoding='utf_8_sig')

# 读取文件
fn = open('result.txt','rt') # 打开文件
string_data = fn.read() # 读出整个文件
fn.close() # 关闭文件

# 文本预处理
pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"') # 定义正则表达式匹配模式
string_data = re.sub(pattern, '', string_data) # 将符合模式的字符去除

# 文本分词
seg_list_exact = jieba.cut(string_data, cut_all = False) # 精确模式分词
object_list = []
remove_words = [u'的', u'，',u'和', u'是', u'随着', u'对于', u'对',u'等',u'能',u'都',u'。',u' ',u'、',u'中',u'在',u'了',
                u'通常',u'如果',u'我们',u'需要',u'艒'] # 自定义去除词库

for word in seg_list_exact: # 循环读出每个分词
    if word not in remove_words: # 如果不在去除词库中
        object_list.append(word) # 分词追加到列表

# 词频统计
word_counts = collections.Counter(object_list) # 对分词做词频统计
word_counts_top40 = word_counts.most_common(40) # 获取前40最高频的词
print (word_counts_top40) # 输出检查

c = (
    WordCloud()
    .add("", word_counts_top40, word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title="WordCloud-shape-diamond"))
    #.render("wordcloud_diamond.html")
)
c.render_notebook()
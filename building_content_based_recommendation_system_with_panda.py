import pandas as pd

movies_data = 'movies.csv'
ratings_data = 'movie_ratings.csv'

# 定义额外的 NaN标识符.
missing_values = ['na', '--', '?', '-', 'None', 'none', 'non']
movies_df = pd.read_csv(movies_data, na_values=missing_values)
ratings_df = pd.read_csv(ratings_data, na_values=missing_values)


#使用正则表达式查找存储在括号中的年份。我们指定了括号，这样我们就不会与标题中有年份的电影发生冲突。
movies_df['year'] = movies_df.title.str.extract('(\d\d\d\d)', expand=False)

# 从“标题”栏中删除年份.
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

# 使用lambda函数，应用strip函数来删除可能出现的任何结尾空白字符。
movies_df['title'] = movies_df.title.apply(lambda x: x.strip())

# 每个流派是由一个|分隔，所以我们只需调用split函数 |.
movies_df['genres'] = movies_df.genres.str.split('|')

# movies_df.info 判断
movies_df.isna().sum()

# 用0填充年NaN值
movies_df.year.fillna(0, inplace=True)

# 将列year从obj转换为int16，将movieId从int64转换为int32以节省内存。
movies_df.year = movies_df.year.astype('int16')
movies_df.movieId = movies_df.movieId.astype('int32')
# movies_df.dtypes

# 首先，先对movies_df 做一个copy。
movies_with_genres = movies_df.copy(deep=True)

#print(movies_with_genres.head(2))
#让我们遍历movies_df，然后将电影类型附加为 1或 0的列。如果该列包含当前索引中类型的电影，则为1，否则为0
x = []
for index, row in movies_df.iterrows():
    x.append(index)
    for genre in row['genres']:
        movies_with_genres.at[index, genre] = 1

# 确认每一行都已被迭代并被处理
#print(len(x) == len(movies_df))
# movies_with_genres.head(3)

# 用0填充NaN值，以显示电影没有该列的类型。
movies_with_genres = movies_with_genres.fillna(0)
#movies_with_genres.head(3)

# print shape和前五行评级数据。
# ratings_df.head()

# Dropping the timestamp column
ratings_df.drop('timestamp', axis=1, inplace=True)
# Confirming the drop
# ratings_df.head(3)

# 让我们确认额定值data_set中的每一列都存在正确的数据类型
#print(ratings_df.dtypes)
#print(ratings_df.isna().sum())

#注意:可以从下面的字典列表中添加或删除电影
#一定要用大写字母写，如果电影是以The开头的，就像《复仇者联盟》一样，那就这样写:《复仇者联盟》。
#创建劳伦斯的个人资料 0到5分，0分和5分满分，看下面劳伦斯的电影评分。
Lawrence_movie_ratings = [
    {'title': 'Predator', 'rating': 4.9},
    {'title': 'Final Destination', 'rating': 4.9},
    {'title': 'Mission Impossible', 'rating': 4},
    {'title': "Beverly Hills Cop", 'rating': 3},
    {'title': 'Exorcist, The', 'rating': 4.8},
    {'title': 'Waiting to Exhale', 'rating': 3.9},
    {'title': 'Avengers, The', 'rating': 4.5},
    {'title': 'Omen, The', 'rating': 5.0}
]
Lawrence_movie_ratings = pd.DataFrame(Lawrence_movie_ratings)
#print(Lawrence_movie_ratings.head())

# 从movies_df中提取电影id，并使用电影id更新lawrence_movie_ratings。
Lawrence_movie_Id = movies_df[movies_df['title'].isin(Lawrence_movie_ratings['title'])]

# 将Lawrence电影Id和评级合并到lawrence_movie_ratings数据框架中.
# 此操作通过标题列隐式合并两个数据帧.
Lawrence_movie_ratings = pd.merge(Lawrence_movie_Id, Lawrence_movie_ratings)

# 删除我们不需要的信息，比如年份和类型
Lawrence_movie_ratings = Lawrence_movie_ratings.drop(['genres', 'year'], 1)

# 劳伦斯的最终文件
#print(Lawrence_movie_ratings)

# 通过输出两者都存在的影片来过滤选择，Lawrence_movie_ratings和movies_with_genre。
Lawrence_genres_df = movies_with_genres[movies_with_genres.movieId.isin(Lawrence_movie_ratings.movieId)]
# Lawrence_genres_df

# 首先，将index重置为default并删除现有索引。
Lawrence_genres_df.reset_index(drop=True, inplace=True)

# 接下来，去掉多余的列
Lawrence_genres_df.drop(['movieId', 'title', 'genres', 'year'], axis=1, inplace=True)

# 我们来确认一下数据的形状，以便于做矩阵乘法。
#print('Shape of Lawrence_movie_ratings is:', Lawrence_movie_ratings.shape)
#print('Shape of Lawrence_genres_df is:', Lawrence_genres_df.shape)

# 我们来求劳伦斯评级列的劳伦斯- genres_df转置的点积.  做乘积
Lawrence_profile = Lawrence_genres_df.T.dot(Lawrence_movie_ratings.rating)

# 将索引设置为movieId。
movies_with_genres = movies_with_genres.set_index(movies_with_genres.movieId)

# 删除四个不必要的列。
movies_with_genres.drop(['movieId', 'title', 'genres', 'year'], axis=1, inplace=True)
#print(movies_with_genres)
# 将类型数乘以权重，然后取加权平均值。   计算相似度，再去做归一化
recommendation_table_df = (movies_with_genres.dot(Lawrence_profile) / Lawrence_profile.sum())

# 将值从大到小排序
recommendation_table_df.sort_values(ascending=False, inplace=True)


# 首先，我们复制原始的movies_df
copy = movies_df.copy(deep=True)

# 然后将它的索引设置为movieId
copy = copy.set_index('movieId', drop=True)

# 接下来，我们列出我们在上面定义的前20个推荐的电影id
top_20_index = recommendation_table_df.index[:20].tolist()

# 最后，我们将这些索引从复制的movies df中切片并保存到一个变量中
recommended_movies = copy.loc[top_20_index, :]

# 现在我们可以按喜好降序显示前20部电影
print('推荐的电影列表：',recommended_movies)
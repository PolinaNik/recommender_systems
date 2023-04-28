import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data):
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.data = data
        self.result = self.data.groupby('user_id')['item_id'].unique().reset_index()
        self.result.columns = ['user_id', 'actual']
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.sparse_user_matrix = self.sparse_matrix(self.user_item_matrix)
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        self.model = self.fit(self, self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data):
        popularity = data.groupby('item_id')['quantity'].sum().reset_index()
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()

        # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
        data.loc[~data['item_id'].isin(top_5000), 'item_id'] = 999999
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробоват ьдругие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def sparse_matrix(matrix):
        # переведем в формат saprse matrix

        return csr_matrix(matrix).tocsr()

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(self, user_item_matrix, n_factors=150, regularization=0.05, iterations=5, num_threads=8):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(bm25_weight(user_item_matrix.T).T)

        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        recs = self.own_recommender.recommend(userid=self.userid_to_id[user],  # userid - id от 0 до N
                               user_items=csr_matrix(self.user_item_matrix).tocsr(),  # на вход user-item matrix
                               N=5,  # кол-во рекомендаций
                               filter_already_liked_items=False,
                               filter_items=[self.itemid_to_id[999999]],
                               recalculate_user=True)
        res = [self.id_to_itemid[rec] for rec in recs[0]]
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        recs = self.model.recommend(userid=self.userid_to_id[user],  # userid - id от 0 до N
                                    user_items=self.sparse_user_matrix,  # на вход user-item matrix
                                    N=5,  # кол-во рекомендаций
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[999999]],
                                    recalculate_user=False)

        res = [self.id_to_itemid[rec] for rec in recs[0]]
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

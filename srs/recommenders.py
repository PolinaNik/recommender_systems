import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекомендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.Dataframe
        Матрица взаимодействий user-item
    weighting: string
        Тип взвешивания, один из вариантов: None, 'bm25', 'tfidf'
    fake_id: int
        идентификатор, которым заменялись редкие объекты. Можно передать None,
        если такого объекта нет
    """

    def __init__(self, data, weighting=None, fake_id=999999):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != fake_id]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        if fake_id is not None:
            self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != fake_id]
        #
        self.fake_id = fake_id
        self.user_item_matrix = self._prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        self.user_item_matrix = csr_matrix(self.user_item_matrix).tocsr()
        self.user_item_matrix_for_pred = self.user_item_matrix
        if weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix).T.tocsr()
        if weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix).T.toscr()
        self.model = self.fit(self, self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data):
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
    def _prepare_dicts(user_item_matrix):
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
        model.fit(user_item_matrix)

        return model

    def _update_dict(self, user_id):
        """Если появился новый user/item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит похожий товар на item_id"""

        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[0][1]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если рекомендаций < N, то дополняем их топ популярными"""

        if len(recommendations) < N:
            top_popular = self.overall_top_purchases[:N]
            top = [rec for rec in top_popular['item_id'] if rec not in recommendations]
            recommendations.extend(top)
            recommendations = recommendations[:N]
            return recommendations
        else:
            return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стандартные библиотеки implicit"""

        filter_items = [] if self.fake_id is not None else [self.itemid_to_id[self.fake_id]]
        recs = model.recommend(userid=self.userid_to_id[user],
                               user_items=self.user_item_matrix_for_pred[self.userid_to_id[user]],
                               N=N,
                               filter_already_liked_items=False,
                               filter_items=filter_items,
                               recalculate_user=True)
        mask = recs[1].argsort()[::-1]
        res = [self.id_to_itemid[rec] for rec in recs[0][mask]]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Кол-во рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стандартные библиотеки implicit"""

        self._update_dict(user_id=user)
        recs = self.model.recommend(userid=self.userid_to_id[user],  # userid - id от 0 до N
                                    user_items=self.user_item_matrix_for_pred[self.userid_to_id[user]],
                                    N=N,  # кол-во рекомендаций
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[self.fake_id]],
                                    recalculate_user=False)

        mask = recs[1].argsort()[::-1]
        res = [self.id_to_itemid[rec] for rec in recs[0][mask]]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Кол-во рекомендаций != {}'.format(N)
        return res

    def get_own_recommendations(self, user, Nget=5):
        """Рекомендуем товары среди тех, которые пользователь уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=5)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ N-купленных юзером товаров"""

        pop_list = self.top_purchases[self.top_purchases['user_id'] == self.userid_to_id[user]].item_id.to_list()[:N]
        res = []
        for item in pop_list:
            sim_item = self._get_similar_item(item)
            res.append(sim_item)

        assert len(res) == N, 'Кол-во рекомендаций != {}'.format(N)
        return res

    def get_similar_user_recommendation(self, user, N=5):
        """Рекомендуем топ N товаров, среди купленных похожими юзерами"""

        similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)
        res = list([])
        for user in similar_users[0][1:]:
            items = self.get_als_recommendations(user)
            unique = list([item for item in items if item not in res])
            if len(unique) != 0:
                res.append(unique[0])
            else:
                top_popular = self.overall_top_purchases[:N]
                top = list([rec for rec in top_popular['item_id'] if rec not in res])
                res.append(top[0])

        assert len(res) == N, 'Кол-во рекомендаций != {}'.format(N)
        return res

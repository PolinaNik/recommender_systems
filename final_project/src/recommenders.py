import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:

    def __init__(self, data, weighting=None, fake_id=999999):

        # топ покупок каждого пользователя
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)

        if fake_id is not None:
            self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != fake_id]

        # топ покупок по всему датасету
        self.overall_top_parcheses = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_parcheses.sort_values('quantity', ascending=False, inplace=True)

        if fake_id is not None:
            self.overall_top_parcheses = self.overall_top_parcheses[self.overall_top_parcheses['item_id'] != fake_id]
        self.overall_top_parcheses = self.overall_top_parcheses['item_id'].tolist()

        self.fake_id = fake_id
        self.user_item_matrix = self._prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        self.user_item_matrix = csr_matrix(self.user_item_matrix).tocsr()
        self.user_item_matrix_for_pred = self.user_item_matrix

        if weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T.tocsr()

        if weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T.tocsr()

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def _prepare_matrix(data_train):

        user_item_matrix = pd.pivot_table(data_train,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0)

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):

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
        """
        Обучает модель, которая рекомендует товары
        среди товаров, купленных юзером
        """

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(user_item_matrix)
        # own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix,
            n_factors=20,
            regularization=0.01,
            iterations=15,
            num_threads=4):

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        # model.fit(user_item_matrix)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):

        """Если появился новый пользователь, то обновляем словари"""

        max_id = max(list(self.userid_to_id.values()))
        max_id += 1

        self.userid_to_id.update({user_id: max_id})
        self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):

        """Находим товар похожий на item_id"""

        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        top_rec = recs[1][0]  # берём второй товар

        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):

        """Если количество рекомендаций меньше топ N, то дополняем топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_parcheses[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):

        """Рекомендуем топ-N товаров"""

        if self.fake_id in self.itemid_to_id:
            filt = [self.itemid_to_id[999999]]
        else:
            filt = None

        res = model.recommend(userid=self.userid_to_id[user],
                              user_items=self.user_item_matrix[self.userid_to_id[user]],
                              N=N,
                              filter_already_liked_items=False,
                              filter_items=filt,
                              recalculate_user=True)

        mask = res[1].argsort()[::-1]
        res = [self.id_to_itemid[rec] for rec in res[0][mask]]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)

        return res

    def get_als_recommendations(self, user, N=5):

        if self.fake_id in self.itemid_to_id:
            filt = [self.itemid_to_id[999999]]
        else:
            filt = None

        if np.isin(user, self.userid_to_id) is False:
            self._update_dict(user_id=user)

        try:
            recs = self.model.recommend(userid=self.userid_to_id[user],  # userid - id от 0 до N
                                        user_items=self.user_item_matrix_for_pred[self.userid_to_id[user]],
                                        N=N,  # кол-во рекомендаций
                                        filter_already_liked_items=False,
                                        filter_items=filt,
                                        recalculate_user=False)

            mask = recs[1].argsort()[::-1]
            res = [self.id_to_itemid[rec] for rec in recs[0][mask]]
            res = self._extend_with_top_popular(res, N=N)

            assert len(res) == N, 'Кол-во рекомендаций != {}'.format(N)
            return res

        except Exception:
            res = []
            self._extend_with_top_popular(res, N=N)

            assert len(res) == N, 'Количество рекомендаций != {}'.format(N)

            return res

    def get_own_recommendations(self, user, N=5):

        """Рекомендуем товары, которые пользователь уже купил"""

        if np.isin(user, self.userid_to_id) is False:
            self._update_dict(user_id=user)

        try:
            return self._get_recommendations(user, model=self.own_recommender, N=N)

        except Exception:
            res = []
            self._extend_with_top_popular(res, N=N)

            assert len(res) == N, 'Количество рекомендаций != {}'.format(N)

            return res

    def get_similar_items_recommendation(self, user, N=5):

        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        if np.isin(user, self.userid_to_id) is False:
            self._update_dict(user_id=user)

        try:
            top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)
            res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()

            assert len(res) == N, 'Количество рекомендаций != {}'.format(N)

            return res

        except Exception:
            res = []
            self._extend_with_top_popular(res, N=N)

            return res

    def get_similar_users_recommendation(self, user, N=5):

        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        if np.isin(user, self.userid_to_id) is False:
            self._update_dict(user_id=user)

        res = []

        try:

        # Находим топ-N похожих пользователей
            similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)
            similar_users = [rec[0] for rec in similar_users]
            similar_users = similar_users[1:]  # удалим юзера из запроса

            for user in similar_users:
                res.extend(self.get_own_recommendations(user, N=1))

            res = self._extend_with_top_popular(res, N=N)

            assert len(res) == N, 'Количество рекомендаций != {}'.format(N)

            return res
        
        except Exception:

            res = []
            self._extend_with_top_popular(res, N=N)

            return res
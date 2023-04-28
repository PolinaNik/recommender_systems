import numpy as np


"""HIT RATE"""
def hit_rate(recommended_list, bought_list):
    "был ли хотя бы 1 релевантный товар среди рекомендованных"

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)
    hit_rate = int(flags.sum() > 0)

    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    "был ли хотя бы 1 релевантный товар среди топ-k рекомендованных"

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list[:k], bought_list, )
    hit_rate = int(flags.sum() > 0)

    return hit_rate


"""PRECISION"""
def precision(recommended_list, bought_list):
    "доля релевантных товаров среди рекомендованных"

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)
    precision = flags.sum() / len(recommended_list)

    return precision


"""Обычно k в precision@k достаточно невелико (5-20) и определяется из бизнес-логики. 
Например, 5 товаров в e-mail рассылке, 20 ответов на первой странице google и т.д"""


def precision_at_k(recommended_list, bought_list, k=5):
    "(# of recommended items that are relevant) / (# of recommended items)"

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    recommended_list = recommended_list[:k]

    flags = np.isin(recommended_list, bought_list)
    precision = flags.sum() / len(recommended_list)

    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    "(revenue of recommended items @k that are relevant) / (revenue of recommended items @k)"

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)

    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]

    flags = np.isin(recommended_list, bought_list)
    precision = (flags * prices_recommended).sum() / prices_recommended.sum()

    return precision


"""RECALL"""
def recall(recommended_list, bought_list):
    "доля рекомендованных товаров среди релевантных = Какой % купленных товаров был среди рекомендованных"

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)
    recall = flags.sum() / len(bought_list)

    return recall


"в recall@k число k обычно достаточно большое (50-200), больше чем покупок у среднестатистического юзера"""
def recall_at_k(recommended_list, bought_list, k=5):
    "(# of recommended items @k that are relevant) / (# of relevant items)"

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    recommended_list = recommended_list[:k]
    flags = np.isin(recommended_list, bought_list)
    recall = flags.sum() / len(bought_list)

    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    "(revenue of recommended items @k that are relevant) / (revenue of relevant items)  "

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    prices_bought = np.array(prices_bought)

    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
    flags = np.isin(recommended_list, bought_list)
    recall = (flags * prices_recommended).sum() / prices_bought.sum()

    return recall


"""МЕТРИКИ РАНЖИРОВАНИЯ"""


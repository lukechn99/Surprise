"""
The :mod:`surprise.accuracy` module provides tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mse
    mae
    fcp
    novelty
    unexpectedness
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import numpy as np
from heapq import heappush, nlargest
from math import log
import pandas as pd
from six import iteritems
# from sklearn.metrics import ndcg_score
# from scipy import sparse

# def ndcg(surprise_predictions, k=50):
#     total_ndcg = []
#     uidset = list(set([p.uid for p in surprise_predictions]))
#     iidset = list(set([p.iid for p in surprise_predictions]))

#     est = pd.DataFrame(0, index=iidset, columns=uidset)
#     true = pd.DataFrame(0, index=iidset, columns=uidset)

#     for prediction in surprise_predictions:
#         est.loc[prediction.iid, prediction.uid] = prediction.est
#         true.loc[prediction.iid, prediction.uid] = prediction.r_ui

#     total_ndcg.append(ndcg_score(true, est, k=k))
#     avg_ndcg = np.mean(total_ndcg)
#     return avg_ndcg

def unexpectedness_aux(user_id, top_k, ratings, uidset, iidset):
    '''
    roughly, we're following the equation -log( p(i,j)/p(i)p(j) ) / log(p(i,j))
    The p(i) is going to be calculated for each item in the user's item corpus
    calculated ahead of time in the pre-processing step

    user_id: userId
    top_5: List[Tuple(est, itemId)]
    ratings: pd.df
    '''
    total_pmi = []

    # pre-process the stats for items the user has rated
    user_rated_items = np.array(ratings.index.to_list())[ratings[user_id] > 0]

    # item : probability(item) map
    pi = {}

    # item-to-p(i) indices map. this is used to calculate p(i,j) later
    pii = {}

    for item_id in user_rated_items:
        # probability of item i considering all items
        pi[item_id] = np.count_nonzero(ratings.loc[:,item_id])/ratings.shape[1]
        indices = []
        for user_id in uidset:
            if ratings.loc[user_id,item_id] != 0:
                indices.append(ratings.loc[user_id,item_id])
        pii[item_id] = indices

    # calculate unexpectedness for each recommended item
    for item in top_k:
        item_id = item[1]
        pj = np.count_nonzero(ratings.loc[:,item_id])/ratings.shape[1]   # float

        # compare this item with every item that the user has rated
        for comp_item_id in user_rated_items:
            # optimize with this later: https://stackoverflow.com/questions/63317109/python-get-column-name-by-non-zero-value-in-row
            # extract item_ids that have nonzero ratings for this user
            jitems = []
            for r, name in zip(ratings.loc[:,item_id], ratings.index):
                if r != 0:
                    jitems.append(int(name))

            # p_ij = len(set(p_ii[item_id2][0].tolist() + ratings.loc[:,item_id].values.flatten().tolist()))/ratings.shape[1]
            pij = len(set(pii[comp_item_id][0].tolist()) & set(jitems))/ratings.shape[1]
            total_pmi.append(-1 * log(max(pij / max((pi[comp_item_id] * pj), 1), 1), 2) / log(max(pij, 1.1), 2))

    avg_pmi = np.mean(total_pmi)
    return avg_pmi

def unexpectedness(surprise_predictions, verbose=True, k=50):
    '''
    We will measure unexpectedness of the top 5 recommendations. 
    To do this, we will take each of the recommendations, and calculate its unexpectedness
    when observing the body of items rated by the user
    More specifically, we are taking the user's item corpus I = {i1, i2, ..., in}
    '''
    # unexpectedness will be a collection of the unexpectedness score for each user with repeats of users for ilfm
    total_unexpectedness = []

    # find top five for each user
    uidset = list(set([p.uid for p in surprise_predictions]))
    iidset = list(set([p.iid for p in surprise_predictions]))
    ratings = pd.DataFrame(index=uidset, columns=iidset)
    for prediction in surprise_predictions:
        ratings.loc[prediction.uid, prediction.iid] = prediction.est
    ratings.fillna(0, inplace=True)

    users_top = {}
    # initialize a heap for each user
    for user in uidset:
        users_top[user] = []

    # push all predictions onto user heaps
    for prediction in surprise_predictions:
        heappush(users_top[prediction.uid], (prediction.est, prediction.iid))

    # calculate the unexpectedness for each user
    for user in uidset:
        total_unexpectedness.append(unexpectedness_aux(user, nlargest(k, users_top[user]), ratings, uidset, iidset))

    avg_unexpectedness = np.mean(total_unexpectedness)
    return avg_unexpectedness

def novelty(surprise_predictions, verbose=True, k=50):
    '''
    Similar to how we take nDCG@k, where k is some value like 5, Vargas' novelty
    calculation is to be taken by calculating novelty for each inverted prediction 
    matrix based on the top k values
    surprise_predictions is a list of lists. Each outer list represents a prediction
    matrix 
    '''
    total_nov = []
    uidset = list(set([p.uid for p in surprise_predictions]))
    iidset = list(set([p.iid for p in surprise_predictions]))
    ratings = pd.DataFrame(index=uidset, columns=iidset)
    for prediction in surprise_predictions:
      ratings.loc[prediction.uid, prediction.iid] = prediction.est
    ratings.fillna(0, inplace=True)

    # find top 5 from each "predictions"
    users_top = {}
    for user in uidset:
      users_top[user] = []

    for p in surprise_predictions:
      heappush(users_top[p.uid], (p.est, p.iid))
    # calculate novelty
    for user in uidset:
      top_k = nlargest(k, users_top[user])
      for i in range(len(top_k)):
        ith_item_score, ith_item = top_k[i]
        # this is a slightly modified novelty function that removes the constant and simplifies rel() and adjusted log() for zero indexing
        nov = (1/max(1, log(i+1,2))) * 2**(float(ith_item_score)-5) * (1-np.count_nonzero(ratings.loc[:, ith_item])/len(uidset))
      total_nov.append(nov)
    avg_nov = np.mean(total_nov)
    return avg_nov

def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mse(predictions, verbose=True):
    """Compute MSE (Mean Squared Error).

    .. math::
        \\text{MSE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse_ = np.mean([float((true_r - est)**2)
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MSE: {0:1.4f}'.format(mse_))

    return mse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose=True):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in iteritems(predictions_u):
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp))

    return fcp

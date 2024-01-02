import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import numba
from numba import cuda, jit, prange
from math import ceil
from Bio import pairwise2
from time import time
import cupy as cp


def normalize_chain(dicto):
    """
    Normalizes dictionary values. Used for the Markov Chain normalization.

    Args:
        dicto: dictionary to ..

    Returns:
        ..d dictionary
    """
    total = 1 / float(np.sum(list(dicto.values())))
    for k, v in dicto.items():
        dicto[k] = v * total
    return dicto


def get_diags(a):
    """
    Extracts all the diagonals from the matrix.

    Args:
        a: numpy array to process

    Returns:
        a list of diagonals
    """
    diags = [a.diagonal(i) for i in range(-a.shape[0] + 1, a.shape[1])]
    return [n.tolist() for n in diags]


def resolution_to_points(r, c_max, c_min):
    """
    Calculates how many points are needed to divide range for given resolution
    :param r: resolution
    :param c_max: maximum value
    :param c_min: minimum value
    :return: the number of points
    """
    c_range = c_max - c_min
    c_points = c_range / r + 1
    return ceil(c_points)


def moving_average(a, n=2):
    """
    Implements fast moving average
    :param a: input data array
    :param n: window size
    :return: processed data array
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def _matchfinder(start_idx, gs, data_len):
    """
    Finds the shortest not repeating sequences according to the Lempel-Ziv algorithm
    :param start_idx: starting point in the array from which search will be started
    :param gs: symbol series
    :param data_len: data length
    :return: current starting index and the shortest non-repeating subsequence length
    """
    max_subsequence_matched = 0
    for i in range(0, start_idx):
        j = 0
        end_distance = data_len - start_idx
        while (start_idx + j < data_len) and (i + j < start_idx) and (gs[i + j] == gs[start_idx + j]):
            j += 1
        if j == end_distance:
            return start_idx, 0
        elif j > max_subsequence_matched:
            max_subsequence_matched = j
    return start_idx, max_subsequence_matched + 1


@cuda.jit
def _matchfinder_gpu(gs, data_len, output):
    """
    Finds the shortest not repeating sequences according to the Lempel-Ziv algorithm. Algorithm adaptation for GPU.
    :param gs: symbol series
    :param data_len: data length
    :param output: output array
    """
    pos = cuda.grid(1)
    max_subsequence_matched = 0
    finish_bool = False
    if pos < data_len:
        for i in range(0, pos):
            j = 0
            end_distance = data_len - pos
            while (pos + j < data_len) and (i + j < pos) and (gs[i + j] == gs[pos + j]):
                j += 1
            if j == end_distance:
                finish_bool = True
                break
            elif j > max_subsequence_matched:
                max_subsequence_matched = j
        if finish_bool:
            output[pos] = end_distance + 1  # CHANGED with XU paper
        else:
            output[pos] = max_subsequence_matched + 1


def matchfinder(gs, gpu=True):
    """
    Finds the shortest not repeating sequences according to the Lempel-Ziv algorithm
    :param gs: symbol series
    :return: the length of the shortest non-repeating subsequences at each step of sequence
    """
    gs = gs.dropna()
    data_len = len(gs)
    gs = np.array(gs.values)
    output = np.zeros(data_len)
    output[0] = 1
    if gpu:
        threadsperblock = 256
        blockspergrid = ceil(data_len / threadsperblock)
        _matchfinder_gpu[threadsperblock, blockspergrid](gs, data_len, output)
    return output


@jit(nopython=True)
def _repeatfinder_dense(s1, s2):
    output = np.zeros(len(s2))
    for pos1 in range(0, len(s2)):
        max_s = 0
        for pos2 in range(0, len(s1)):
            j = 0
            while s1[pos2 + j] == s2[pos1 + j]:
                j += 1
                if pos2 + j == len(s1) or pos1 + j == len(s2):
                    break
            if j > max_s:
                max_s = j
        output[pos1] = max_s
    return max(output) - 1 / len(s2) - 1


@jit(nopython=True)
def _repeatfinder_sparse(s1, s2):
    matrix = [[0 for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] += 1
                else:
                    matrix[i][j] = matrix[i - 1][j - 1] + 1
            else:
                matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1])
    cs = matrix[-1][-1]
    return cs - 1 / len(s2) - 1


def _repeatfinder_equally_sparse(s1, s2):
    matrix = np.zeros((len(s1), len(s2)))  # prepare matrix for results
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:  # if matched symbols are at the start of the sequence
                    matrix[i][j] += 1  # if symbols matched - add 1
                else:
                    matrix[i][j] = 1  # if symbols matched - add 1
    s2_indi = (np.vstack(
        [np.arange(matrix.shape[1]) for x in range(matrix.shape[0])]) + 1) * matrix  # convert matched 1's into indices
    s2diags = get_diags(s2_indi)  # get all diagonals
    nonzero_transitions = [[y for y in x if y != 0] for x in s2diags]  # filter out empty lists
    nonzero_transitions = [[(x[y], x[y + 1]) for y in range(len(x) - 1)] for x in
                           nonzero_transitions]  # convert to transitions
    translateds2 = [[(s2[int(y[0]) - 1], s2[int(y[1]) - 1]) for y in x] for x in nonzero_transitions if
                    len(x) > 0]  # convert to matched symbols
    # search for self-transitions and transitions
    try:
        selft = sum([x[0] == x[1] for x in sorted(translateds2, key=len)[-1]])
    except:
        selft = 0
    try:
        nonselft = sum([x[0] != x[1] for x in sorted(translateds2, key=len)[-1]])
    except:
        nonselft = 0
    return sum([selft, nonselft]) / (len(s2) - 1)


def _global_align(s1, s2):
    one = list(s1)
    two = list(s2)
    alignment = \
        pairwise2.align.globalms(one, two, 1, -1, -1, 0, penalize_end_gaps=False, one_alignment_only=True,
                                 gap_char=['-'])[
            0]
    return alignment.score / (len(two) - 1)


def _iterative_global_align(s1, s2):
    one = list(s1)  # preapare lists of symbols
    two = list(s2)
    cut = two  # assign currently processed sequence
    all_match = []
    to_search = []
    while True:
        best_match = \
            pairwise2.align.globalms(one, cut, 1, -1, -1, 0, penalize_end_gaps=False,
                                     one_alignment_only=True, gap_char=['-'])[
                0]
        zipped = [(x, y) for x, y in zip(best_match[0], best_match[1])]  # combinations of matched symbols
        out_of_match = [1 if x[0] != x[1] and x[0] == '-' else 0 for x in zipped]  # search for mismatched symbols
        out_diff = np.diff(out_of_match)  # find gaps
        starts = [x for x in range(len(out_diff)) if out_diff[x] == 1]  # find starts of gaps
        ends = [x for x in range(len(out_diff)) if out_diff[x] == -1]  # find ends of gaps
        lengths = [y - x for x, y in zip(starts, ends)]  # find lengths of gaps
        first_ends = [1 if x == '-' else 0 for x in best_match[0]]  # find mismatched starts and ends of sequence
        first_ends_diff = np.diff(first_ends)  # find range of mismatched starts and ends
        first_ends_starts = [x for x in range(len(first_ends_diff)) if first_ends_diff[x] == 1]
        first_ends_ends = [x for x in range(len(first_ends_diff)) if first_ends_diff[x] == -1]
        if len(first_ends_ends) > 0 and 0 not in first_ends[:first_ends_ends[0]]:
            begin = best_match[1][:first_ends_ends[0] + 1]
        else:
            begin = []
        if len(first_ends_starts) > 0 and 0 not in first_ends[first_ends_starts[-1] + 1:]:
            end = best_match[1][first_ends_starts[-1] + 1:]
        else:
            end = []
        lengths += [len(begin), len(end)]
        maxleng = np.max(lengths)  # check which mismatch is the longest
        longest = np.where(lengths == maxleng)  # take the mismatched part
        if longest[0][0] == len(lengths) - 1:  # check if its a gap in the middle or at the ends
            to_search.append(end)
        elif longest[0][0] == len(lengths) - 2:
            to_search.append(begin)
        else:
            for n in longest[0]:
                n = n
                if n == len(lengths) - 1:
                    to_search.append(end)
                elif n == len(lengths) - 2:
                    to_search.append(begin)
                else:
                    to_search.append(best_match[1][starts[n] + 1:ends[n] + 1])
        if len(to_search) == 0:  # if there are no sequences to match - stop
            if best_match[2] > 0:  # if there was positive score in the last match - add it to the list
                all_match.append(best_match[2] - 1)  # add score to the list of scores (-1 for transitions)
            break
        cut = to_search.pop(0)  # pop out the sequence for search
        cut = [x for x in cut if isinstance(x, float)]  # take only symbols
        if best_match[2] <= 0:  # if there is zero score already - stop
            break
        all_match.append(best_match[2] - 1)  # add score to the list of scores (-1 for transitions)
        if len(cut) <= 1:
            break  # if sequence for search does not consist of at least two symbols - stop
    return sum(all_match) / (len(two) - 1)


def extract_diaginfo(a):
    shorter_size = a.shape[0] - 1
    embed_array = np.zeros((a.shape[0], a.shape[1] + shorter_size * 2), dtype=int)
    embed_array[:, shorter_size:embed_array.shape[1] - shorter_size] = a
    extracted = np.array(
        [np.diagonal(embed_array, offset=i) for i in range(-embed_array.shape[0] + 1, embed_array.shape[1])
         if np.sum(np.diagonal(embed_array, offset=i)) != 0])
    return extracted


@jit(nopython=True, cache=True)
def extract_diaginfo_jit(a):
    shorter_size = a.shape[0] - 1
    rows, cols = a.shape
    new_cols = cols + shorter_size * 2
    embed_array = np.zeros((rows, new_cols), dtype=np.int32)
    embed_array[:, shorter_size:embed_array.shape[1] - shorter_size] = a

    max_diag_length = min(embed_array.shape[0], embed_array.shape[1])
    diag_length = np.arange(-a.shape[0] + 1, a.shape[1]).size
    extracted = np.zeros((diag_length, max_diag_length), dtype=np.int32)
    extracted_count = 0

    for i in range(-a.shape[0] + 1, a.shape[1]):
        start_col = shorter_size + i
        start_row = 0
        diagonal = np.empty((1, max_diag_length))
        for j in range(max_diag_length):
            diagonal[0, j] = embed_array[start_row + j, start_col + j]
        if np.sum(diagonal) != 0:
            extracted[extracted_count, :] = diagonal
            extracted_count += 1
    extracted = extracted[:extracted_count, :]
    return extracted


def get_last_nonzero(a, return_index=False):
    nonzero_a = np.flipud(np.argwhere(a))
    last_ind = nonzero_a[np.unique(nonzero_a[:, 0], return_index=True)[1]]
    if return_index:
        return last_ind[:, 1]
    else:
        return a[last_ind[:, 0], last_ind[:, 1]]


@jit(nopython=True, cache=True)
def get_last_nonzero_jit(a, return_index=False):
    lasts = np.empty(a.shape[0], dtype=np.int64)
    for i in range(a.shape[0]):
        cur_last = 0
        for j in range(a[i].shape[0]):
            if a[i][j] != 0:
                if return_index:
                    cur_last = j
                else:
                    cur_last = a[i][j]
            lasts[i] = cur_last
    return lasts


def get_rolls(a):
    """
    Repeats all the rows of a matrix below with a +1 roll to the right. Repeats till last columns rolls over the end
    of the matrix
    :param a: matrix to roll
    :return: rolled matrix with sum(0) rows removed
    """
    tiles = np.tile(a, (a.shape[1], 1))
    rows, column_indices = np.ogrid[:tiles.shape[0], :tiles.shape[1]]
    rolls = np.repeat(np.arange(a.shape[1]), a.shape[0])
    column_indices = column_indices - rolls[:, np.newaxis]
    tiles = tiles[rows, column_indices]
    tiles[column_indices < 0] = 0
    tiles = tiles[~np.all(tiles == 0, axis=1), :]
    return tiles


@jit(nopython=True, cache=True)
def get_rolls_jit(a):
    """
    Repeats all the rows of a matrix below with a +1 roll to the right. Repeats till last columns rolls over the end
    of the matrix
    :param a: matrix to roll
    :return: rolled matrix with sum(0) rows removed
    """
    tiles = np.zeros((a.shape[1] * a.shape[0], a.shape[1]), dtype=np.int32)
    for i in range(a.shape[1]):
        tiles[i * a.shape[0]:(i + 1) * a.shape[0], :] = a

    column_indices = np.empty((1, tiles.shape[1]), dtype=np.int32)
    for j in range(tiles.shape[1]):
        column_indices[0, j] = j
    rolls_prepare = np.arange(a.shape[1])
    rolls = np.empty(rolls_prepare.shape[0] * a.shape[0], dtype=np.int32)
    for i in range(a.shape[1]):
        rolls[i * a.shape[0]:(i + 1) * a.shape[0]] = rolls_prepare[i]
    column_indices = column_indices - rolls.reshape(-1, 1)
    new_tiles = np.zeros_like(tiles, dtype=np.int32)
    nonzero = 0
    for i in range(column_indices.shape[0]):
        org_j = -1
        j_sum = 0
        for j in column_indices[i, :]:
            org_j += 1
            if j < 0:
                continue
            new_tiles[i, org_j] = tiles[i, j]
            j_sum += tiles[i, j]
        if j_sum > 0:
            nonzero += 1
    filtered_tiles = np.zeros((nonzero, new_tiles.shape[1]), dtype=np.int32)
    i = 0
    for j in new_tiles:
        if np.sum(j) > 0:
            filtered_tiles[i, :] = j
            i += 1
    return filtered_tiles


@cuda.jit
def get_rolls_cuda(a, new_tiles, translation_pattern):
    pos = cuda.grid(1)
    if pos < a.shape[0]:
        for i in range(translation_pattern.shape[0]):
            org_j = -1
            for j in translation_pattern[i, :]:
                org_j += 1
                if j < 0:
                    continue
                new_tiles[pos * a.shape[1] + i, org_j] = a[pos, j]


def run_get_rolls_cuda(a):
    threads_per_block = 128
    blocks_per_grid = (a.shape[0] + (threads_per_block - 1)) // threads_per_block
    new_tiles = np.zeros((a.shape[0] * a.shape[1], a.shape[1]), dtype=a.dtype)
    translation_pattern = np.arange(a.shape[1]) - np.arange(a.shape[1])[:, np.newaxis]

    a_gpu = cuda.to_device(a)
    new_tiles_gpu = cuda.to_device(new_tiles)
    translation_pattern_gpu = cuda.to_device(translation_pattern)

    get_rolls_cuda[blocks_per_grid, threads_per_block](a_gpu, new_tiles_gpu, translation_pattern_gpu)

    new_tiles = new_tiles_gpu.copy_to_host()
    new_tiles = new_tiles[new_tiles.sum(axis=1) > 0]
    return new_tiles


def _equally_sparse_match(s1, s2, overreach=True, roll=True):  # TODO: SPEEDUP SECOND FOLD & GPU
    matrix = np.array([[c1 == c2 for c2 in s2] for c1 in s1],
                      dtype=np.bool_)  # where s1 (vertical) matches s2 (horizontal)
    if not matrix.any():
        return None

    # s1_indi = np.tile(np.arange(1,matrix.shape[0]+1),(matrix.shape[1],1)).T * matrix
    s2_indi = np.tile(np.arange(1, matrix.shape[1] + 1),
                      (matrix.shape[0], 1)) * matrix  # convert matches into indeces of s2
    symbols = np.tile(s2, (matrix.shape[0], 1)) * matrix  # convert matches into matched
    # symbols at their postions

    # s1rolls = get_rolls(extract_diaginfo(s1_indi))
    if roll:
        s2rolls = get_rolls(extract_diaginfo(s2_indi))  # extract diagonals with s2 matched indecies
        syrolls = get_rolls(extract_diaginfo(symbols))  # extract diagonals with matched symbols
    else:
        s2rolls = extract_diaginfo(s2_indi)
        syrolls = extract_diaginfo(symbols)

    last_s1 = s2rolls.shape[1] - get_last_nonzero(s2rolls,
                                                  return_index=True)  # get the last index of matched diagonal in s1 #TODO: get last of all combs
    last_s2 = get_last_nonzero(s2rolls) - 1  # get the last index of matched diagonal in s2
    # of matched symbol in s1 from the end (1 being first)

    index_of_next_symbol = last_s1 + last_s2  # what symbol (index) from s2 will be after the match from s1 and s2
    if overreach:
        s2 = np.hstack([s2, s1])
    reach_mask = index_of_next_symbol < s2.size  # check if that index is in s2 and produce masking
    syrolls = syrolls[reach_mask]
    index_of_next_symbol = index_of_next_symbol[reach_mask]  # apply mask
    next_symbols = np.array(s2)[index_of_next_symbol.astype(int)]  # get all next symbols

    return syrolls - 1, next_symbols - 1


@jit(nopython=True, cache=True)
def _equally_sparse_match_jit(s1, s2, overreach=True, roll=True):
    rows = len(s1)
    cols = len(s2)
    matrix = np.empty((rows, cols), dtype=np.bool_)
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = s1[i] == s2[j]

    if not matrix.any():
        return None

    s2_indi = np.zeros_like(matrix, dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            s2_indi[i, j] = (j + 1) * matrix[i, j]

    symbols = np.zeros_like(matrix, dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            symbols[i, j] = s2[j] * matrix[i, j]

    if roll:
        s2rolls = get_rolls_jit(extract_diaginfo_jit(s2_indi))  # extract diagonals with s2 matched indecies
        syrolls = get_rolls_jit(extract_diaginfo_jit(symbols))  # extract diagonals with matched symbols
    else:
        s2rolls = extract_diaginfo_jit(s2_indi)
        syrolls = extract_diaginfo_jit(symbols)

    last_s1 = s2rolls.shape[1] - get_last_nonzero_jit(s2rolls,
                                                      return_index=True)  # get the last index of matched diagonal in s1 #TODO: get last of all combs
    last_s2 = get_last_nonzero_jit(s2rolls) - 1  # get the last index of matched diagonal in s2
    # of matched symbol in s1 from the end (1 being first)

    index_of_next_symbol = last_s1 + last_s2  # what symbol (index) from s2 will be after the match from s1 and s2
    if overreach:
        new_s2 = np.empty(s2.shape[0] + s1.shape[0], dtype=np.int64)
        for i in range(s2.shape[0]):
            new_s2[i] = s2[i]
        for i in range(s1.shape[0]):
            new_s2[i + s2.shape[0]] = s1[i]
        s2 = new_s2
    reach_mask = index_of_next_symbol < s2.size  # check if that index is in s2 and produce masking
    syrolls = syrolls[reach_mask]
    index_of_next_symbol = index_of_next_symbol[reach_mask]  # apply mask
    next_symbols = np.empty_like(index_of_next_symbol, dtype=np.int32)
    for i in range(index_of_next_symbol.shape[0]):
        next_symbols[i] = s2[index_of_next_symbol[i]]  # get all next symbols
    return syrolls - 1, next_symbols - 1


def fano_inequality(distinct_locations, entropy):
    """
    Implementation of the Fano's inequality. Algorithm solves it and returns the solution.
    :param distinct_locations:
    :param entropy:
    :return:
    """
    func = lambda x: (-(x * np.log2(x) + (1 - x) * np.log2(1 - x)) + (1 - x) * np.log2(
        distinct_locations - 1)) - entropy
    return fsolve(func, .9999)[0]


def to_labels(trajectories_frame):
    """
    Adds labels column based on repeating geometries or coordinates
    :param trajectories_frame: TrajectoriesFrame object class
    :return: TrajectoriesFrame with labels column
    """
    to_tranformation = trajectories_frame[trajectories_frame.geometry.is_valid]
    try:
        to_tranformation['labels'] = to_tranformation[to_tranformation._geom_cols[0]].astype(str) + ',' + \
                                     to_tranformation[to_tranformation._geom_cols[1]].astype(str)
        trajectories_frame['labels'] = trajectories_frame[to_tranformation._geom_cols[0]].astype(str) + ',' + \
                                       trajectories_frame[to_tranformation._geom_cols[1]].astype(str)
    except:
        to_tranformation['labels'] = to_tranformation.lat.astype(str) + ',' + to_tranformation.lon.astype(str)
        trajectories_frame['labels'] = trajectories_frame.lat.astype(str) + ',' + trajectories_frame.lon.astype(str)
    unique_coors = pd.DataFrame(pd.unique(to_tranformation['labels']))
    unique_coors.index = unique_coors.loc[:, 0]
    unique_coors.loc[:, 0] = range(len(unique_coors))
    sub_dict = unique_coors.to_dict()[0]
    trajectories_frame.astype({'labels': str})
    trajectories_frame['labels'] = trajectories_frame['labels'].map(sub_dict)
    return trajectories_frame


def _equally_sparse_match_old(s1, s2):
    matrix = np.zeros((len(s1), len(s2)))  # prepare matrix for results
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:  # if matched symbols are at the start of the sequence
                    matrix[i][j] += 1  # if symbols matched - add 1
                else:
                    matrix[i][j] = 1  # if symbols matched - add 1
    s2_indi = (np.vstack(
        [np.arange(matrix.shape[1]) + 1 for x in range(matrix.shape[0])])) * matrix  # convert matched 1's into indices
    s1_indi = (np.hstack(
        [np.expand_dims(np.arange(matrix.shape[0]), axis=1) + 1 for x in range(matrix.shape[1])])) * matrix
    s2diags = get_diags(s2_indi)  # get all diagonals
    s1diags = get_diags(s1_indi)
    if sum([sum(x) for x in s2diags]) == 0:
        return None
    nonzero_s2 = [[y - 1 for y in x if y != 0] for x in s2diags if sum(x) > 0]  # filter out empty lists
    nonzero_s1 = [[len(s1) - y + 1 for y in x if y != 0] for x in s1diags if sum(x) > 0]  # filter out empty lists
    # nonzero_s2 = [x for x in nonzero_s2 if len(x) >= 2]
    # nonzero_s1 = [x for x in nonzero_s1 if len(x) >= 2]
    matches = []
    for x, y in zip(nonzero_s1, nonzero_s2):
        if y[-1] + x[-1] < len(s2):
            matched_pattern = np.zeros(len(s1) + len(s2)) - 1
            for z, w in zip(y, x):
                matched_pattern[int(-w)] = s2[int(z)]
            next_symbol = s2[int(y[-1] + x[-1])]
            matches.append((matched_pattern, next_symbol))
    return matches


def remove_subset_rows(arr):
    """Removes rows from arr that are subsets of other rows."""
    unique_rows = {}

    # Iterate over each row in arr
    for row in arr:
        # Convert the row to a tuple
        row_tuple = tuple(row)

        # Check if the row is a subset of any of the unique rows
        is_subset = False
        for unique_tuple in unique_rows:
            if set(row_tuple).issubset(set(unique_tuple)):
                is_subset = True
                break

        # If the row is not a subset of any of the unique rows, add it to the dictionary
        if not is_subset:
            unique_rows[row_tuple] = row

    # Convert the dictionary values back to a numpy array
    unique_arr = np.array(list(unique_rows.values()))
    return unique_arr


@jit(nopython=True)
def sparse_predict_jit(context, model, candidates, counts, recency, lengths,
                       recency_weights=None, length_weights=None,
                       from_dist=False,
                       org_recency_weights=None, org_length_weights=None, completeness_weights=None,
                       uniqueness_weights=None, count_weights=None):
    model_size = model.shape[1]
    n_matches = model.shape[0]
    context = context.astype(np.int64)
    pad_size = model_size - context.shape[0]
    if pad_size > 0:
        context = np.hstack((np.zeros(pad_size, dtype=np.int64), context))
    elif pad_size < 0:
        context = context[-model_size:]

    # MATCHING
    context_size = context.shape[0]
    matches = np.zeros((n_matches, context_size), dtype=np.bool_)
    match_mask = np.zeros(n_matches, dtype=np.bool_)
    for i in range(n_matches):
        for j in range(context_size):
            if model[i][j] == context[j]:
                matches[i][j] = True
                match_mask[i] = True
    if not np.any(match_mask):
        unique_candidates = np.unique(candidates)
        candidates_counts = np.zeros(unique_candidates.max() + 1)
        for j in candidates:
            candidates_counts[j] += 1
        SMC = np.argmax(candidates_counts)
        candidates_counts = (candidates_counts / np.sum(candidates_counts))
        return SMC, candidates_counts

    # Weights
    matches = matches[match_mask]
    candidates = candidates[match_mask]
    counts = counts[match_mask]
    recency = recency[match_mask]
    lengths = lengths[match_mask]

    matches_sum = np.sum(matches, axis=1, dtype=np.float64)
    if uniqueness_weights is not None:
        unq_weights = weight_reversed_jit(counts, uniqueness_weights)
    else:
        unq_weights = np.ones_like(matches_sum)

    if org_recency_weights is not None:
        org_recency = weight_reversed_jit(recency, org_recency_weights)
    else:
        org_recency = np.ones_like(matches_sum)

    if org_length_weights is not None:
        org_lengths = weight_jit(lengths, org_length_weights)
    else:
        org_lengths = np.ones_like(matches_sum)

    if completeness_weights is not None:
        completeness_measure = matches_sum / lengths
        completeness_weigh = weight_jit(completeness_measure, completeness_weights)
    else:
        completeness_weigh = np.ones_like(matches_sum)

    if count_weights is not None:
        count_weights = weight_jit(counts, count_weights)
    else:
        count_weights = np.ones_like(matches_sum)

    if recency_weights is not None:
        recency = weight_recency_jit(matches, recency_weights)
    else:
        recency = np.ones_like(matches_sum)

    if length_weights is not None:
        lengths = weight_jit(matches_sum, length_weights)
    else:
        lengths = np.ones_like(matches_sum)

    matches_sum = matches_sum * org_recency
    matches_sum = matches_sum * org_lengths
    matches_sum = matches_sum * recency
    matches_sum = matches_sum * lengths
    matches_sum = matches_sum * completeness_weigh
    matches_sum = matches_sum * unq_weights
    matches_sum = matches_sum * count_weights

    # PREDICTION
    unique_candidates = np.unique(candidates)
    candidates_probs = np.zeros(unique_candidates.max() + 1)
    for j in range(candidates.shape[0]):
        candidates_probs[candidates[j]] += matches_sum[j]
    candidates_probs /= np.sum(candidates_probs)
    SMC = np.argmax(candidates_probs)
    return SMC, candidates_probs


@jit(nopython=True)
def weight_recency_jit(vect, recency_weights):
    last_nonzero = np.empty(vect.shape[0])
    nonzero_elements = np.nonzero(vect)
    for j in range(nonzero_elements[0].shape[0] - 1):
        if nonzero_elements[0][j] != nonzero_elements[0][j + 1]:
            last_nonzero[nonzero_elements[0][j]] = vect.shape[1] - nonzero_elements[1][j]
    last_nonzero[-1] = vect.shape[1] - nonzero_elements[1][-1]
    if recency_weights in ['inverted', 'IW']:
        recency_func = lambda x: 1 / x
        recency = np.array(list(map(recency_func, last_nonzero)))
    elif recency_weights in ['inverted squared', 'IWS']:
        recency_func = lambda x: 1 / x ** 2
        recency = np.array(list(map(recency_func, last_nonzero)))
    elif recency_weights in ['inverted qubic', 'IWQ']:
        recency_func = lambda x: 1 / x ** 3
        recency = np.array(list(map(recency_func, last_nonzero)))
    elif recency_weights in ['linear', 'quadratic', 'L', 'Q']:
        last_nonzero = vect.shape[1] - last_nonzero + 1
        if recency_weights in ['linear', 'L']:
            recency = last_nonzero / vect.shape[1]
        else:
            recency = (last_nonzero / vect.shape[1]) ** 2
    return recency


@jit(nopython=True)
def weight_jit(vect, comp_weights):
    vect = vect.astype(np.float64)
    if comp_weights in ['inverted', 'IW']:
        # weights_func = lambda x: 1 / x
        lengths = 1 / vect
    elif comp_weights in ['inverted squared', 'IWS']:
        # weights_func = lambda x: 1 / x ** 2
        lengths = 1 / (vect ** 2)
    elif comp_weights in ['linear', 'L']:
        # weights_func = lambda x: x
        lengths = vect
    elif comp_weights in ['quadratic', 'Q']:
        # weights_func = lambda x: x ** 2
        lengths = vect ** 2
    # lengths = np.array(list(map(weights_func, vect)), dtype=np.float64)
    lengths = scale_vector(lengths)
    return lengths


@jit(nopython=True)
def weight_reversed_jit(vect, unq_weights):
    if unq_weights in ['inverted', 'IW']:
        weights_func = lambda x: 1 / x
        unqs = np.array(list(map(weights_func, vect)))
        unqs = scale_vector(unqs)
    elif unq_weights in ['inverted squared', 'IWS']:
        weights_func = lambda x: 1 / x ** 2
        unqs = np.array(list(map(weights_func, vect)))
        unqs = scale_vector(unqs)
    elif unq_weights in ['inverted qubic', 'IWQ']:
        weights_func = lambda x: 1 / x ** 3
        unqs = np.array(list(map(weights_func, vect)))
        unqs = scale_vector(unqs)
    elif unq_weights in ['linear', 'L']:
        vect_reversed = (vect.max() - vect) + 1
        unqs = scale_vector(vect_reversed)
    elif unq_weights in ['quadratic', 'Q']:
        vect_reversed = (vect.max() - vect) + 1
        unqs = scale_vector(vect_reversed ** 2)
    return unqs


@jit(nopython=True)
def scale_vector(v):
    minim = np.min(v)
    maxim = np.max(v)
    if minim == maxim:
        return np.full(v.shape[0], 0.5)
    else:
        return (v - minim) / (maxim - minim)


def sort_lines(mat):
    n, m = mat.shape
    for i in range(m):
        kind = 'stable' if i > 0 else None
        mat = mat[np.argsort(mat[:, m - 1 - i], kind=kind)]
    return mat


@jit(parallel=True, nopython=True)
def unq_counts_model(sorted_mat):
    n, m = sorted_mat.shape
    assert m >= 0
    isUnique = np.zeros(n, np.bool_)
    uniqueCount = 1
    if n > 0:
        isUnique[0] = True
    for i in prange(1, n):
        isUniqueVal = False
        for j in range(m):
            isUniqueVal |= sorted_mat[i, j] != sorted_mat[i - 1, j]
        isUnique[i] = isUniqueVal
        uniqueCount += isUniqueVal
    uniqueValues = np.empty((uniqueCount, m), np.int64)
    duplicateCounts = np.zeros(len(uniqueValues), np.int64)

    cursor = 0
    for i in range(n):
        cursor += isUnique[i]
        for j in range(m):
            uniqueValues[cursor - 1, j] = sorted_mat[i, j]
        duplicateCounts[cursor - 1] += 1
    return uniqueValues, duplicateCounts


def cupy_sort_lines(mat):
    cupy_mat = cp.array(mat)
    return cupy_mat[cp.lexsort(cupy_mat.T[::-1, :])].get()

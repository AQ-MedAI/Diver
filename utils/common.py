import numpy as np
import json, pickle
import os
import sys
import csv
csv.field_size_limit(sys.maxsize)
import sys
from tqdm import tqdm
import requests
from multiprocessing import Pool


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


# def select_field(features, field):
#     return [
#         [
#             choice[field]
#             for choice in feature.choices_features
#         ]
#         for feature in features
#     ]

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(obj, fp)


def load_json(input_file):
    with open(input_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data


def load_jsonl(input_file):
    data_list = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            data_list.append(json.loads(line))
    return data_list


# def save_jsonl(obj, path):
#     assert isinstance(obj, list)
#     with open(path, "w", encoding="utf-8") as fp:
#         for _elem in obj:
#             dump_str = json.dumps(_elem) + os.linesep
#             fp.write(dump_str)

# 为了和reasonir格式兼容
def save_json_dict_format(obj, path):
    assert isinstance(obj, list)
    with open(path, "w", encoding="utf-8") as fp:
        for idx, text in obj:
            elem = {"id": str(idx), "query": text}
            fp.write(json.dumps(elem, ensure_ascii=False) + "\n")


def save_jsonl_with_offset(obj, path):
    assert isinstance(obj, list)
    offset_list = []
    with open(path, "w", encoding="utf-8") as fp:
        for _elem in obj:
            offset_list.append(fp.tell())
            dump_str = json.dumps(_elem) + os.linesep
            fp.write(dump_str)
    assert len(obj) == len(offset_list)
    return offset_list


def load_jsonl_with_offset(offset, path):
    with open(path, encoding="utf-8") as fp:
        fp.seek(offset)
        return json.loads(fp.readline())


def get_line_offsets(path, encoding="utf-8"):
    offset_list = []
    with open(path, encoding=encoding, mode="r") as fp:
        line = True
        while line:
            begin_tell = fp.tell()
            line = fp.readline()
            if line.strip():
                offset_list.append(begin_tell)
    return offset_list

import time
def get_line_offsets_verbose(path, encoding="utf-8"):
    start_time = time.time()
    offset_list = []
    idx_line = 0
    with open(path, encoding=encoding, mode="r") as fp:
        line = True
        while line:
            idx_line += 1
            if idx_line % 100000 == 0:
                print(f"progress {time.time()-start_time}: {idx_line}")
            begin_tell = fp.tell()
            line = fp.readline()
            if line.strip("\n"):
                offset_list.append(begin_tell)
    return offset_list


def save_pkll_with_offset(obj, path):
    assert isinstance(obj, list)
    offset_list = []
    with open(path, "wb") as fp:
        for _elem in obj:
            offset_list.append(fp.tell())
            fp.write(pickle.dumps(_elem))
        last_offset = fp.tell()
    assert len(obj) == len(offset_list)
    pair_offset_list = []
    for _idx in range(len(offset_list)):
        if _idx < len(offset_list) - 1:
            pair_offset_list.append([offset_list[_idx], offset_list[_idx+1] - offset_list[_idx]])
        else:
            pair_offset_list.append([offset_list[_idx], last_offset - offset_list[_idx]])
    return pair_offset_list


def load_pkll_with_offset(offset, path):
    assert len(offset) == 2  # 1. seek 2. for size
    with open(path, "rb") as fp:
        fp.seek(offset[0])
        line = fp.read(offset[1])
        return pickle.loads(line)


def load_pickle(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
        return data


def save_pickle(obj, path, protocol=None):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp, protocol=protocol)


def save_list_to_file(str_list, file_path, use_basename=False, verbose=False):
    with open(file_path, "w", encoding="utf-8") as fp:
        for path_str in tqdm(str_list, disable=(not verbose), desc="Save list to file"):
            fp.write(os.path.basename(path_str) if use_basename else path_str)
            fp.write(os.linesep)


def load_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def load_list_from_file(file_path, encoding="utf-8"):
    data = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding=encoding) as fp:
            for line in fp:
                data.append(line.strip())
    return data


def get_data_path_list(data_dir, suffix=None):  # , recursive=False
    path_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            temp = os.path.join(root, file)
            if isinstance(suffix, str) and not temp.endswith(suffix):
                continue
            path_list.append(temp)
    return path_list


def file_exists(path):
    return os.path.exists(path) and os.path.isfile(path)


def dir_exists(path):
    return os.path.exists(path) and os.path.isdir(path)


def shards_iterator(file_list):
    fp_list = [open(p) for p in file_list]
    ptr = 0
    while len(fp_list) > 0:
        fp = fp_list[ptr]
        line = fp.readline()
        # print("!!!", line)
        if line:
            yield line
            ptr += 1
        else:
            fp_list.pop(ptr).close()
        ptr = ptr % len(fp_list) if len(fp_list) > 0 else 0


def load_multiple_jsonl_files(data_dir, suffix="jsonl"):

    def load_jsonl_line(jline):
        return json.loads(jline)

    data_list = []
    with Pool() as p:
        for lexemb in p.imap(
                load_jsonl_line,
                tqdm(shards_iterator(shards_iterator(get_data_path_list(data_dir, suffix))), desc="read all lines"),
                1000):
            data_list.append(lexemb)
    return data_list

# =============
# ==STR Part ==
def is_lower(text):
    return all(ord("a") <= ord(c) <= ord("z") for c in text)


def is_capital(text):
    return all(ord("A") <= ord(c) <= ord("Z") for c in text)


def is_word(text):
    return all(ord("A") <= ord(c) <= ord("Z") or ord("a") <= ord(c) <= ord("z") for c in text)


def get_val_str_from_dict(val_dict):
    # sort
    sorted_list = list(sorted(val_dict.items(), key=lambda item: item[0]))
    str_return = ""
    for key, val in sorted_list:
        if len(str_return) > 0:
            str_return += ", "
        str_return += "%s: %.4f" % (key, val)
    return str_return


def parse_span_str(_span_str, min_val=0, max_val=10000):

    if isinstance(_span_str, (int, float)):
        return int(_span_str), max_val
    elif isinstance(_span_str, type(None)):
        return None
    elif not isinstance(_span_str, str):
        return min_val, max_val
    lst = _span_str.split(",")
    if len(lst) == 0:
        return min_val, max_val
    elif len(lst) == 1:
        if len(lst[0]) > 0:
            return int(lst[0]), max_val
        else:
            return min_val, max_val
    elif len(lst) == 2:
        _minv, _maxv  = min_val, max_val
        if len(lst[0]) > 0:
            _minv = int(lst[0])
        if len(lst[1]) > 0:
            _maxv = int(lst[1])
        return _minv, _maxv
    else:
        raise AttributeError("got invalid {} as {}".format(_span_str, _span_str.split(",")))


# statistics
def get_statistics_for_num_list(num_list):
    res = {}
    arr = np.array(num_list, dtype="float32")
    res["mean"] = np.mean(arr)
    res["median"] = np.median(arr)
    res["min"] = np.min(arr)
    res["max"] = np.max(arr)
    res["std"] = np.std(arr)
    res["len"] = len(num_list)
    return res


def remove_filename_suffix(path):
    return ".".join(path.split(".")[:-1])


def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()

if __name__ == "__main__":
    jsonl_file = "../data/thinkqe/aops/bm25-aug2_result_retrieval.trec.topics.jsonl"
    json_file = "../data/thinkqe/aops/bm25-aug2_result_retrieval.trec.topics.json"
    with open(jsonl_file, "r", encoding="utf-8") as f:
        obj = [json.loads(line) for line in f]
    save_json_dict_format(obj, json_file)
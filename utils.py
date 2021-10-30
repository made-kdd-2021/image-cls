import json
import pickle


def dump_object(path_to_file, object, by_ref: bool = False):
    with open(path_to_file, "wb") as file_object:
        pickle.dump(object, file_object, protocol=3)


def dump_json(path_to_file, json_dict: dict):
    with open(path_to_file, "w", encoding="utf-8") as file_object:
        json.dump(json_dict, file_object)


def load_dump(path_to_file):
    with open(path_to_file, "rb") as file_object:
        return pickle.load(file_object)


def load_json(path_to_file):
    with open(path_to_file, "r", encoding="utf-8") as file_object:
        return json.load(file_object)

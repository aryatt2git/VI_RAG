import os
import json
import pytest

class OllamaResponse:
    def __init__(self, filepath="pmid_json.txt"):
        self.filepath = filepath


    def ollama_import(self):
        file = open(self.filepath, "r")
        json_str = file.read()
        file.close()

        return json.loads(json_str)

class RefFiles:
    def __init__(self, ref_folder="References/"):
        self.ref_folder = ref_folder

    def ref_files(self):
        return os.listdir(self.ref_folder)


#def test_Ref_PMID():

def test_ref_folder_files():

    filepath = "/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/pmid_json.txt"
    ref_folder = "/Users/arjun/PycharmProjects/VI_RAG/VI_RAG/tools/References"

    ollama_response = OllamaResponse(filepath=filepath)
    ref_folder = RefFiles(ref_folder=ref_folder)

    for pmids, filepaths in ollama_response.ollama_import().items():

        for paths, status in filepaths.items():

            file = paths.split("/")[-1]

            if status != "copy":
                assert file in ref_folder.ref_files()

            if status == "copy":
                assert file not in ref_folder.ref_files()





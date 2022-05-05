import os
import pickle
from collections import Counter
from glob import glob

import clip
import gdown
import gradio as gr
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from SimSearch import FaissCosineNeighbors

# DOWNLOAD THE DATASET and Files
gdown.download(
    "https://static.taesiri.com/gamephysics/GTAV-Videos.zip",
    output="./GTAV-Videos.zip",
    quiet=False,
)
gdown.download(
    "https://static.taesiri.com/gamephysics/mini-GTA-V-Embeddings.zip",
    output="./GTA-V-Embeddings.zip",
    quiet=False,
)

# EXTRACT

torchvision.datasets.utils.extract_archive(
    from_path="GTAV-Videos.zip", to_path="Videos/", remove_finished=False
)
# EXTRACT

torchvision.datasets.utils.extract_archive(
    from_path="GTA-V-Embeddings.zip",
    to_path="Embeddings/VIT32/",
    remove_finished=False,
)
# Initialize CLIP model
clip.available_models()

# # Searcher
class GamePhysicsSearcher:
    def __init__(self, CLIP_MODEL, GAME_NAME, EMBEDDING_PATH="./Embeddings/VIT32/"):
        self.CLIP_MODEL = CLIP_MODEL
        self.GAME_NAME = GAME_NAME
        self.simsearcher = FaissCosineNeighbors()

        self.all_embeddings = glob(f"{EMBEDDING_PATH}{self.GAME_NAME}/*.npy")

        self.filenames = [os.path.basename(x) for x in self.all_embeddings]
        self.file_to_class_id = {x: i for i, x in enumerate(self.filenames)}
        self.class_id_to_file = {i: x for i, x in enumerate(self.filenames)}
        self.build_index()

    def read_features(self, file_path):
        with open(file_path, "rb") as f:
            video_features = pickle.load(f)
        return video_features

    def read_all_features(self):
        features = {}
        filenames_extended = []

        X_train = []
        y_train = []

        for i, vfile in enumerate(tqdm(self.all_embeddings)):
            vfeatures = self.read_features(vfile)
            features[vfile.split("/")[-1]] = vfeatures
            X_train.extend(vfeatures)
            y_train.extend([i] * vfeatures.shape[0])
            filenames_extended.extend(vfeatures.shape[0] * [vfile.split("/")[-1]])

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        return X_train, y_train

    def build_index(self):
        X_train, y_train = self.read_all_features()
        self.simsearcher.fit(X_train, y_train)

    def text_to_vector(self, query):
        text_tokens = clip.tokenize(query)
        with torch.no_grad():
            text_features = self.CLIP_MODEL.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    # Source: https://stackoverflow.com/a/480227
    def f7(self, seq):
        seen = set()
        seen_add = seen.add  # This is for performance improvement, don't remove
        return [x for x in seq if not (x in seen or seen_add(x))]

    def search_top_k(self, q, k=5, pool_size=1000, search_mod="Majority"):
        q = self.text_to_vector(q)
        nearest_data_points = self.simsearcher.get_nearest_labels(q, pool_size)

        if search_mod == "Majority":
            topKs = [x[0] for x in Counter(nearest_data_points[0]).most_common(k)]
        elif search_mod == "Top-K":
            topKs = list(self.f7(nearest_data_points[0]))[:k]

        video_filename = [
            f"./Videos/{self.GAME_NAME}/"
            + self.class_id_to_file[x].replace("npy", "mp4")
            for x in topKs
        ]

        return video_filename


################ SEARCH CORE ################
# CRAETE CLIP MODEL
vit_model, vit_preprocess = clip.load("ViT-B/32")
vit_model.eval()

saved_searchers = {}


def gradio_search(query, game_name, selected_model, aggregator, pool_size, k=6):
    # print(query, game_name, selected_model, aggregator, pool_size)
    if f"{game_name}_{selected_model}" in saved_searchers.keys():
        searcher = saved_searchers[f"{game_name}_{selected_model}"]
    else:
        if selected_model == "ViT-B/32":
            model = vit_model
            searcher = GamePhysicsSearcher(CLIP_MODEL=model, GAME_NAME=game_name)
        else:
            raise

        saved_searchers[f"{game_name}_{selected_model}"] = searcher

    results = []
    relevant_videos = searcher.search_top_k(
        query, k=k, pool_size=pool_size, search_mod=aggregator
    )

    params = ", ".join(
        map(str, [query, game_name, selected_model, aggregator, pool_size])
    )
    results.append(params)

    for v in relevant_videos:
        results.append(v)
        sid = v.split("/")[-1].split(".")[0]
        results.append(f"https://www.reddit.com/r/GamePhysics/comments/{sid}/")
    return results


def main():
    list_of_games = ["Grand Theft Auto V"]

    title = "CLIP + GamePhysics - Searching dataset of Gameplay bugs"
    description = "Enter your query and select the game you want to search. The results will be displayed in the console."
    article = """
  This demo shows how to use the CLIP model to search for gameplay bugs in a video game.
  """

    # GRADIO APP
    iface = gr.Interface(
        fn=gradio_search,
        inputs=[
            gr.inputs.Textbox(
                lines=1,
                placeholder="Search Query",
                default="A person flying in the air",
                label=None,
            ),
            gr.inputs.Radio(list_of_games, label="Game To Search"),
            gr.inputs.Radio(["ViT-B/32"], label="MODEL"),
            gr.inputs.Radio(["Majority", "Top-K"], label="Aggregator"),
            gr.inputs.Slider(300, 2000, label="Pool Size", default=1000),
        ],
        outputs=[
            gr.outputs.Textbox(type="auto", label="Search Params"),
            gr.outputs.Video(type="mp4", label="Result 1"),
            gr.outputs.Textbox(type="auto", label="Submission URL - Result 1"),
            gr.outputs.Video(type="mp4", label="Result 2"),
            gr.outputs.Textbox(type="auto", label="Submission URL - Result 2"),
            gr.outputs.Video(type="mp4", label="Result 3"),
            gr.outputs.Textbox(type="auto", label="Submission URL - Result 3"),
            gr.outputs.Video(type="mp4", label="Result 4"),
            gr.outputs.Textbox(type="auto", label="Submission URL - Result 4"),
            gr.outputs.Video(type="mp4", label="Result 5"),
            gr.outputs.Textbox(type="auto", label="Submission URL - Result 5"),
        ],
        examples=[
            ["A red car", list_of_games[0], "ViT-B/32", "Top-K", 1000],
            ["A person wearing pink", list_of_games[0], "ViT-B/32", "Top-K", 1000],
            ["A car flying in the air", list_of_games[0], "ViT-B/32", "Majority", 1000],
            [
                "A person flying in the air",
                list_of_games[0],
                "ViT-B/32",
                "Majority",
                1000,
            ],
            [
                "A car in vertical position",
                list_of_games[0],
                "ViT-B/32",
                "Majority",
                1000,
            ],
            ["A bike inside a car", list_of_games[0], "ViT-B/32", "Majority", 1000],
            ["A bike on a wall", list_of_games[0], "ViT-B/32", "Majority", 1000],
            ["A car stuck in a rock", list_of_games[0], "ViT-B/32", "Majority", 1000],
            ["A car stuck in a tree", list_of_games[0], "ViT-B/32", "Majority", 1000],
        ],
        title=title,
        description=description,
        article=article,
        enable_queue=True,
    )

    iface.launch()


if __name__ == "__main__":
    main()

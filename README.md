<div align="center">    

# CLIP meets GamePhysics


[![Website](http://img.shields.io/badge/Website-4b44ce.svg)](https://asgaardlab.github.io/CLIPxGamePhysics/)
[![arXiv](https://img.shields.io/badge/arXiv-2203.11096-b31b1b.svg)](https://arxiv.org/abs/2203.11096)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/taesiri/CLIPxGamePhysics)
[![Hugging Face Dataset (FULL)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-red)](https://huggingface.co/datasets/taesiri/GamePhysics)
</div>



This repository will contain code for the paper "CLIP meets GamePhysics: Towards bug identification in gameplay videos using zero-shot transfer learning"

by [Mohammad Reza Taesiri](https://taesiri.com), Finlay Macklon, [Cor-Paul Bezemer](https://asgaard.ece.ualberta.ca/)


<div align="center">    

# Abstract

</div>

*Gameplay videos contain rich information about how players interact with the game and how the game responds. Sharing gameplay videos on social media platforms, such as Reddit, has become a common practice for many players. Often, players will share gameplay videos that showcase video game bugs. Such gameplay videos are software artifacts that can be utilized for game testing, as they provide insight for bug analysis. Although large repositories of gameplay videos exist, parsing and mining them in an effective and structured fashion has still remained a big challenge. In this paper, we propose a search method that accepts any English text query as input to retrieve relevant videos from large repositories of gameplay videos. Our approach does not rely on any external information (such as video metadata); it works solely based on the content of the video. By leveraging the zero-shot transfer capabilities of the Contrastive Language-Image Pre-Training (CLIP) model, our approach does not require any data labeling or training. To evaluate our approach, we present the GamePhysics dataset consisting of 26,954 videos from 1,873 games, that were collected from the GamePhysics section on the Reddit website. Our approach shows promising results in our extensive analysis of simple queries, compound queries, and bug queries, indicating that our approach is useful for object and event detection in gameplay videos. An example application of our approach is as a gameplay video search engine to aid in reproducing video game bugs.*



## Citation information

```
@INPROCEEDINGS {9796271,
        author = {M. Taesiri and F. Macklon and C. Bezemer},
        booktitle = {2022 IEEE/ACM 19th International Conference on Mining Software Repositories (MSR)},
        title = {CLIP meets GamePhysics: Towards bug identification in gameplay videos using zero-shot transfer learning},
        year = {2022},
        volume = {},
        issn = {},
        pages = {270-281},
        abstract = {Gameplay videos contain rich information about how players interact with the game and how the game responds. Sharing gameplay videos on social media platforms, such as Reddit, has become a common practice for many players. Often, players will share game-play videos that showcase video game bugs. Such gameplay videos are software artifacts that can be utilized for game testing, as they provide insight for bug analysis. Although large repositories of gameplay videos exist, parsing and mining them in an effective and structured fashion has still remained a big challenge. In this paper, we propose a search method that accepts any English text query as input to retrieve relevant videos from large repositories of gameplay videos. Our approach does not rely on any external information (such as video metadata); it works solely based on the content of the video. By leveraging the zero-shot transfer capabilities of the Contrastive Language-Image Pre-Training (CLIP) model, our approach does not require any data labeling or training. To evaluate our approach, we present the GamePhysics dataset consisting of 26,954 videos from 1,873 games, that were collected from the GamePhysics section on the Reddit website. Our approach shows promising results in our extensive analysis of simple queries, compound queries, and bug queries, indicating that our approach is useful for object and event detection in gameplay videos. An example application of our approach is as a gameplay video search engine to aid in reproducing video game bugs. Please visit the following link for the code and the data: https://asgaardlab.github.io/CLIPxGamePhysics/},
        keywords = {training;visualization;social networking (online);computer bugs;transfer learning;games;software},
        doi = {10.1145/3524842.3528438},
        url = {https://doi.ieeecomputersociety.org/10.1145/3524842.3528438},
        publisher = {IEEE Computer Society},
        address = {Los Alamitos, CA, USA},
        month = {may}
        }
```

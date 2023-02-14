# ⚡️ BLIP Research Poster 🔬

[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai)
![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

Use this app to share your research paper results. This app lets you connect a blogpost, arxiv paper, and a jupyter
notebook and even have an interactive demo for people to play with the model. This app also allows industry
practitioners to reproduce your work.

> The model demo is implemented using the official [BLIP](https://github.com/salesforce/BLIP) implementation by salesforce team.

## Getting started

### Installation

#### With Lightning CLI

`lightning install app lightning/icml22-blip`

### Example

```python
# update app.py at the root of the repo
import lightning as L

poster_dir = "resources"
paper = "https://arxiv.org/pdf/2201.12086.pdf"
blog = "https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/"
github = "https://github.com/salesforce/BLIP"
tabs = ["Poster", "Blog", "Model Demo", "Notebook Viewer", "Paper"]

app = L.LightningApp(
    ResearchApp(
        poster_dir=poster_dir,
        paper=paper,
        blog=blog,
        notebook_path="BLIP/demo.ipynb",
        launch_gradio=True,
        tab_order=tabs,
        launch_jupyter_lab=False,  # don't launch for public app, can expose to security vulnerability
    )
)
```

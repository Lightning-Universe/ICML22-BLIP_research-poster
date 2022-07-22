import logging
import sys

import gradio as gr
from lightning.app.components.serve import ServeGradio
from rich.logging import RichHandler

sys.path.append("BLIP")
from gradio_app import Model

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger(__name__)


class ModelDemo(ServeGradio):
    """Serve model with Gradio UI.

    You need to define i. `build_model` and ii. `predict` method and Lightning `ServeGradio` component will
    automatically launch the Gradio interface.
    """

    inputs = [
        gr.inputs.Image(type="pil"),
        gr.inputs.Radio(
            choices=["Image Captioning", "Visual Question Answering"],
            type="value",
            default="Image Captioning",
            label="Task",
        ),
        gr.inputs.Textbox(lines=2, label="Question"),
    ]
    outputs = gr.outputs.Textbox(label="Output")
    enable_queue = True
    examples = [
        ["test.jpg", "Image Captioning", "None"],
        ["test1.jpg", "Image Captioning", "None"],
        ["test2.jpg", "Visual Question Answering", "Which bird is this?"]
        ]

    def __init__(self):
        super().__init__(parallel=True)

    def build_model(self):
        logger.info("loading model...")
        model = Model()
        logger.info("built model!")
        return model

    def predict(self, image, task: str, question: str) -> str:
        return self.model.predict(image, task, question)

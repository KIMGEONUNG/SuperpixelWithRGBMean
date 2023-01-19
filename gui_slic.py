#!/usr/bin/env python

from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import gradio as gr
from glob import glob


class GUISLIC():

    def __init__(self):
        files = sorted(glob("images/*"))
        with gr.Blocks() as self.demo:
            height = 400
            with gr.Box(), gr.Row():
                view = gr.Image(interactive=False).style(height=height)
                view_slic = gr.Image(label="slic",
                                     interactive=False).style(height=height)

            with gr.Box(), gr.Column():
                with gr.Tab("slic"), gr.Row():
                    b_num_seg = gr.Number(
                        label="num segment",
                        value=20,
                    )
                    b_sigma = gr.Slider(
                        label="sigma",
                        value=5.0,
                        maximum=10,
                    )
                btn = gr.Button("Estimate")

            gr.Examples(files, inputs=[view])
            btn.click(self.estimate,
                      inputs=[
                          view,
                          b_num_seg,
                          b_sigma,
                      ],
                      outputs=[view_slic])

    def launch(self, share=False):
        self.demo.launch(share=share)

    def estimate(
        self,
        x,
        b_num_seg,
        b_sigma,
    ):
        x = img_as_float(x)

        segments_slic = slic(
            x,
            n_segments=int(b_num_seg),
            sigma=b_sigma,
        )

        out_slic = mark_boundaries(x, segments_slic)

        return out_slic


if __name__ == "__main__":
    gui = GUISLIC()
    gui.launch()

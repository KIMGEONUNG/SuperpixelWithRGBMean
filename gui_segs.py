#!/usr/bin/env python

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
import gradio as gr
from glob import glob


class GUISegs():

    def __init__(self):
        with gr.Blocks() as self.demo:
            height = 400
            with gr.Box():
                view = gr.Image(interactive=False).style(height=height)
                with gr.Row():
                    view_fz = gr.Image(label="felzenszwalb",
                                       interactive=False).style(height=height)
                    view_slic = gr.Image(
                        label="slic", interactive=False).style(height=height)
                    view_quick = gr.Image(
                        label="quickshift",
                        interactive=False).style(height=height)
                    view_watershed = gr.Image(
                        label="watershed",
                        interactive=False).style(height=height)
            gr.Examples(sorted(glob("images/*")), inputs=view)
            with gr.Column():
                with gr.Tab("fz"), gr.Row():
                    a_scale = gr.Number(label="scale", value=100)
                    a_sigma = gr.Slider(label="sigma", value=5.0)
                    a_min_size = gr.Number(label="min size", value=50)
                with gr.Tab("slic"), gr.Row():
                    b_num_seg = gr.Number(label="num segment", value=100)
                    b_sigma = gr.Slider(label="sigma", value=5.0)
                with gr.Tab("quick"), gr.Row():
                    c_kernel_size = gr.Number(label="kernel size", value=3)
                    c_max_dist = gr.Number(label="max dist", value=6)
                    c_ratio = gr.Slider(label="ratio", value=0.5)
                with gr.Tab("watershed"), gr.Row():
                    d_markers = gr.Number(label="markers", value=250)
                    d_compact = gr.Slider(label="compactness", value=0.001)
                btn = gr.Button("Estimate")

            btn.click(self.estimate,
                      inputs=[
                          view,
                          a_scale,
                          a_sigma,
                          a_min_size,
                          b_num_seg,
                          b_sigma,
                          c_kernel_size,
                          c_max_dist,
                          c_ratio,
                          d_markers,
                          d_compact,
                      ],
                      outputs=[view_fz, view_slic, view_quick, view_watershed])

    def launch(self, share=False):
        self.demo.launch(share=share)

    def estimate(
        self,
        x,
        a_scale,
        a_sigma,
        a_min_size,
        b_num_seg,
        b_sigma,
        c_kernel_size,
        c_max_dist,
        c_ratio,
        d_markers,
        d_compact,
    ):
        x = img_as_float(x)

        # fz
        segments_fz = felzenszwalb(
            x,
            scale=int(a_scale),
            sigma=a_sigma,
            min_size=int(a_min_size),
        )

        # slic
        segments_slic = slic(
            x,
            n_segments=int(b_num_seg),
            sigma=b_sigma,
        )

        # quick
        segments_quick = quickshift(
            x,
            kernel_size=int(c_kernel_size),
            max_dist=int(c_max_dist),
            ratio=c_ratio,
        )

        # watershed
        gradient = sobel(rgb2gray(x))
        segments_watershed = watershed(
            gradient,
            markers=int(d_markers),
            compactness=d_compact,
        )

        out_fz = mark_boundaries(x, segments_fz)
        out_slic = mark_boundaries(x, segments_slic)
        out_quick = mark_boundaries(x, segments_quick)
        out_watershed = mark_boundaries(x, segments_watershed)

        return out_fz, out_slic, out_quick, out_watershed


if __name__ == "__main__":
    gui = GUISegs()
    gui.launch()

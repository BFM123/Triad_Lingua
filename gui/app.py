import gradio as gr
from chained_translate import translate_chichewa_to_hindi

def run_gui():
    def translate(text, show_intermediate):
        final, intermediate = translate_chichewa_to_hindi(text, return_intermediate=True)
        return intermediate if show_intermediate else "", final

    with gr.Blocks(title="Triad Lingua Translator: Chichewa → Hindi") as demo:
        gr.Markdown("## Triad Lingua Translator: Chichewa → Hindi _(via English)_")
        gr.Markdown("Translates Chichewa to Hindi by chaining through English. Optionally display the intermediate English translation.")
        
        with gr.Row():
            with gr.Column(scale=1):
                chichewa_input = gr.Textbox(label="Enter Chichewa text")
                show_intermediate = gr.Checkbox(label="Show Intermediate English Translation", value=False)
                translate_btn = gr.Button("Translate")
            with gr.Column(scale=1):
                intermediate_out = gr.Textbox(label="Intermediate English", interactive=False)
                final_out = gr.Textbox(label="Final Hindi Translation", interactive=False)

        translate_btn.click(fn=translate, inputs=[chichewa_input, show_intermediate], outputs=[intermediate_out, final_out])

    demo.launch()

if __name__ == "__main__":
    run_gui()

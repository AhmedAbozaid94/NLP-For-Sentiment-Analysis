from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import gradio as gr
import os 
from keras.models import load_model
from Functions import *
#-------------------------TEXT------------------------#
#                      TEXT MODEL
#-----------------------------------------------------#
label2id = {'sad': 5, 'fear': 2, 'angry': 0, 'neutral': 4, 'disgust': 1, 'happy': 3}
id2label = {v: k for k, v in label2id.items()}

model_path = "D:/A_Graduate Project/Full Application/bert-base-cased-finetuned-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6, id2label=id2label, label2id=label2id)
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def predict_text(text):
    if not text.strip():  
        return None  
    result = pipeline(text)
    result = {label['label']: label['score'] for label in result[0]}
    return result
#-------------------------AUDIO-------------------------#
#                      AUDIO MODEL
#-------------------------------------------------------#
MODEL_PATH = 'D:/A_Graduate Project/Full Application/Audio_Model/my_model.keras'
audio_model = load_model(MODEL_PATH)
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'neutral']

def predict_audio(audio_path):
    if not audio_path or os.path.getsize(audio_path) == 0:  
        return None  
    preprocessed_audio = preprocess_audio(audio_path)
    predictions = audio_model.predict(preprocessed_audio)
    scores = predictions[0]
    class_scores = {class_name: float(score) for class_name, score in zip(class_names, scores)}
    results = {class_name: score for class_name, score in class_scores.items()}
    return results
#-------------------------INTERFACE---------------------#
#                     Gradio Interface
#-------------------------------------------------------#
css = """
.text-output-height {
    height: 100%;
}
.output-label {
    display: flex;
    flex-direction: column;
    justify-content: center; /* This centers the content vertically */
}
"""

with gr.Blocks(css = css) as demo:
    gr.Markdown("""
    <div style="text-align: center; color: White !important; padding: 10px; font-size: 30px;">
        SENTIMENT ANALYZER
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Text Sentiment Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(lines=10, placeholder="Type your text here...", label="Textbox")
                    submit_text = gr.Button("Submit")
                    clear_text = gr.Button("Clear")
                    flag_text = gr.Button("Save Result")
                with gr.Column(scale=1):
                    text_output = gr.Label(label="Predicted Emotion", num_top_classes=6, elem_classes=["text-output-height", "output-label"])
            
            submit_text.click(fn=predict_text, inputs=text_input, outputs=text_output)
            clear_text.click(fn=clear_t, inputs=[], outputs=[text_input, text_output])
            flag_text.click(fn=save_and_commit_text, inputs=[text_input, text_output], outputs=[])

        with gr.Tab("Audio Sentiment Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(label="Record or Upload Audio", type="filepath")
                    submit_audio = gr.Button("Submit")
                    clear_audio = gr.Button("Clear")
                    flag_audio = gr.Button("Save Result")
                with gr.Column(scale=1):
                    audio_output = gr.Label(label="Predicted Emotion", num_top_classes=6, elem_classes=["text-output-height", "output-label"])
            
            submit_audio.click(predict_audio, inputs=audio_input, outputs=audio_output)
            clear_audio.click(fn=clear_a, inputs=[], outputs=[audio_output])
            flag_audio.click(fn=save_and_commit_audio, inputs=[audio_input, audio_output], outputs=[])

    gr.Markdown("""
    <div style="text-align: center; color: White !important; padding: 20px;">
        POWERED BY MUST AI DEPARTMENT TEAM 2024
    </div>
    """)

demo.launch(inbrowser=True)
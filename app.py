import gradio as gr
import joblib
import numpy as np

def greet(name):
    return "Hello " + name + "!"

knn_model = joblib.load('neighbors.pkl')
final_dataset = joblib.load('final_dataset')
titles = joblib.load('titles')



def prediction(index):
    return knn_model.kneighbors(final_dataset[index].reshape(1,-1), n_neighbors=4, return_distance = False)

print(titles.head())
print(np.where(titles == "avatar"))
with gr.Blocks() as demo:
    name = gr.Textbox(label = "Enter a movie name: ")


    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")

demo.launch()
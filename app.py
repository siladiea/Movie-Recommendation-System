import gradio as gr
import joblib
import numpy as np
import string


knn_model = joblib.load('neighbors.pkl')
final_dataset = joblib.load('final_dataset')
titles = joblib.load('titles')

titles = titles.tolist()

def prediction(index):
    return knn_model.kneighbors(final_dataset[index].reshape(1,-1), n_neighbors=4, return_distance = False)

def greet(name):
    if(isinstance(name,str)):
        name = name.lower()
    try:
        prediction(titles.index(name))
        indices = prediction(titles.index(name))
        print(indices)
        print('         ')
        return f"Movies that you may like: {string.capwords(titles[indices[0][1]])}, {string.capwords(titles[indices[0][2]])}, and {string.capwords(titles[indices[0][3]])}"
    except:
        return f"Sorry that movie doesn't exist in our database or it might be spelt incorrecty. Here are some top movies in our database: {', '.join(titles[:11])}"

with gr.Blocks() as demo:
    name = gr.Textbox(label = "Enter a movie name: ")
    output = gr.Textbox(label="Similar Movies:")
    greet_btn = gr.Button("Generate")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")

demo.launch(share=True)

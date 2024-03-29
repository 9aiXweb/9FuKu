from flask import Flask, render_template, request
import torch
from transformers import pipeline
import pickle
import os

app = Flask(__name__)

# GPUが利用可能であればCUDAデバイスを使用
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route('/', methods=['GET', 'POST'])
def main_stock():

    if request.method == 'POST':
        if request.form.get('input') is None:
            return render_template('index.html')
        
        inputs = str(request.form['input'])
        filename = 'data/model.sav'

        if os.path.exists(filename):
            pipe = pickle.load(open(filename, 'rb'))
        else:
            # torch.device(device) を指定して GPU を使用
            pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=device)

        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": inputs},
        ]
        
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        # モデルの保存
        
        pickle.dump(pipe, open(filename, 'wb'))
        


        html = render_template('index.html', output=outputs)

    else:    
        html = render_template('index.html')
    return html

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4444)

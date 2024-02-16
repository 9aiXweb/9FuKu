from flask import Flask, render_template, request
import torch
from transformers import pipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_stock():

    if request.method == 'POST':
        if(request.form.get('input')  == None):
            return render_template('index.html')
        outputs = str(request.form['input'])
        # inputs = str(request.form['input'])
        # pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16)#, device_map="auto")

        # # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a friendly chatbot who always responds in the style of a pirate",
        #     },
        #     {"role": "user", "content": inputs},
        # ]
        # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        html = render_template('index.html', output=outputs)

    else:    
        html = render_template('index.html')
    return html

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=4444)
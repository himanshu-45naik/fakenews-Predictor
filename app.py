from flask import Flask, render_template, request
from fakenews import manual_testing


app = Flask(__name__, static_folder='static')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        news = request.form['news']
        result = manual_testing(news)
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
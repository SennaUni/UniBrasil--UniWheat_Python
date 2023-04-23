from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process')
def process():
    months = request.args.get('months')
    subprocess.call(['python', 'timeSeries/Prophet.py', months])
    return 'Executado com sucesso'

if __name__ == '__main__':
    app.run(debug=True)
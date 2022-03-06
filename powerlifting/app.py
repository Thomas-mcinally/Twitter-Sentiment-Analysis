'''
Tutorial series to use: https://www.youtube.com/watch?v=mqhxxeeTbu0&list=PLzMcBGfZo4-n4vJJybUVV3Un_NFS5EOgX
'''


from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello world!</p>"



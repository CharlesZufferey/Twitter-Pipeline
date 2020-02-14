# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:24:14 2020

@author: Charles
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Hello from Flask</h1>'

app.run(host='0.0.0.0')
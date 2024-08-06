from flask import Flask, request, jsonify, send_from_directory
from delivery_optimizer import DeliveryOptimizer
import os

app = Flask(__name__, static_folder='static')

optimizer = DeliveryOptimizer()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    locations = data['locations']
    optimized_route, stats = optimizer.optimize_route(locations)
    return jsonify({'route': optimized_route, 'stats': stats})

if __name__ == '__main__':
    app.run(debug=True)
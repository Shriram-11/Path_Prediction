from flask import Flask, request, jsonify
from sklearn.cluster import DBSCAN
import numpy as np
from datetime import datetime

app = Flask(__name__)


class ParticleFilter:
    def __init__(self, num_particles, initial_state, transition_std):
        self.num_particles = num_particles
        self.particles = np.ones((num_particles, 2)) * initial_state
        self.weights = np.ones(num_particles) / num_particles
        self.transition_std = transition_std

    def predict(self):
        self.particles += np.random.randn(self.num_particles,
                                          2) * self.transition_std

    def update(self, measurement):
        self.weights *= self.gaussian_likelihood(measurement)
        self.weights += 1.e-300  # to avoid round-off to zero
        self.weights /= np.sum(self.weights)  # normalize

    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.  # avoid round-off error
        indexes = np.searchsorted(
            cumulative_sum, np.random.rand(self.num_particles))
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0 / self.num_particles)

    def gaussian_likelihood(self, measurement):
        dist = np.linalg.norm(self.particles - measurement, axis=1)
        return np.exp(-0.5 * (dist / self.transition_std)**2) / (self.transition_std * np.sqrt(2 * np.pi))

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)


def clean_and_process_json_data(json_data):
    cleaned_data = []
    try:
        data = json_data['data']
        for entry in data:
            try:
                if all(k in entry for k in ["timestamp", "latitude", "longitude"]):
                    timestamp = datetime.fromisoformat(
                        entry["timestamp"].replace("Z", "+00:00"))
                    latitude = float(entry["latitude"])
                    longitude = float(entry["longitude"])
                    cleaned_data.append([timestamp, latitude, longitude])
            except Exception as e:
                print("Error processing entry:", e)
    except Exception as e:
        print("Error processing JSON data:", e)

    return cleaned_data


def predict_next_location(coordinates):
    initial_state = coordinates[-1][1:]  # Initial state without timestamp
    pf = ParticleFilter(num_particles=1000,
                        initial_state=initial_state, transition_std=0.0001)

    # Update the filter with provided data
    for coord in coordinates:
        pf.predict()
        pf.update(coord[1:])  # Exclude timestamp from update
        pf.resample()

    # Predicted next location
    predicted_location = pf.estimate()
    return predicted_location


@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    cleaned_data = clean_and_process_json_data(json_data)

    if cleaned_data:
        predicted_location = predict_next_location(cleaned_data)
        return jsonify({"predicted_location": predicted_location.tolist()})
    else:
        return jsonify({"error": "Failed to process JSON data or no valid data points found."}), 400


if __name__ == '__main__':
    app.run(debug=True)

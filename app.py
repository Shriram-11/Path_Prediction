from flask import Flask, request, jsonify
from sklearn.cluster import DBSCAN
import numpy as np
import json
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
        data = json.loads(json_data)
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
    json_data = '''
    [
    {
        "timestamp": "2024-06-03T03:19:27.810Z",
        "latitude": "15.8837952",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-03T03:19:35.811Z",
        "latitude": "15.8838256",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-03T03:19:39.132Z",
        "latitude": "15.884784",
        "longitude": "75.704992"
    },
    {
        "timestamp": "2024-06-03T03:19:48.107Z",
        "latitude": "15.8837808",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-03T03:19:51.428Z",
        "latitude": "15.8837648",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-03T03:19:58.399Z",
        "latitude": "15.884576",
        "longitude": "75.7035904"
    },
    {
        "timestamp": "2024-06-03T03:20:01.708Z",
        "latitude": "15.8842752",
        "longitude": "75.7032576"
    },
    {
        "timestamp": "2024-06-03T03:20:09.688Z",
        "latitude": "15.8834896",
        "longitude": "75.702624"
    },
    {
        "timestamp": "2024-06-03T03:20:13.003Z",
        "latitude": "15.8836992",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-03T03:20:26.550Z",
        "latitude": "15.8849136",
        "longitude": "75.703552"
    },
    {
        "timestamp": "2024-06-03T03:20:27.776Z",
        "latitude": "15.8837936",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-03T03:20:36.887Z",
        "latitude": "15.8838304",
        "longitude": "75.7030976"
    },
    {
        "timestamp": "2024-06-03T03:20:40.175Z",
        "latitude": "15.8839152",
        "longitude": "75.7032384"
    },
    {
        "timestamp": "2024-06-03T03:20:48.726Z",
        "latitude": "15.8835488",
        "longitude": "75.7033088"
    },
    {
        "timestamp": "2024-06-03T03:20:52.064Z",
        "latitude": "15.8925776",
        "longitude": "75.7042816"
    },
    {
        "timestamp": "2024-06-03T03:21:00.031Z",
        "latitude": "15.8839136",
        "longitude": "75.7033536"
    },
    {
        "timestamp": "2024-06-03T03:21:03.364Z",
        "latitude": "15.8840912",
        "longitude": "75.7025856"
    },
    {
        "timestamp": "2024-06-03T03:21:10.339Z",
        "latitude": "15.8840544",
        "longitude": "75.703008"
    },
    {
        "timestamp": "2024-06-03T03:21:13.659Z",
        "latitude": "15.8840128",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-03T03:21:15.564Z",
        "latitude": "15.8841088",
        "longitude": "75.7026752"
    },
    {
        "timestamp": "2024-06-03T03:21:22.652Z",
        "latitude": "15.8840784",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-03T03:21:23.847Z",
        "latitude": "15.9046704",
        "longitude": "75.70608"
    },
    {
        "timestamp": "2024-06-03T03:21:26.955Z",
        "latitude": "15.9143632",
        "longitude": "75.7114112"
    },
    {
        "timestamp": "2024-06-03T03:21:28.179Z",
        "latitude": "15.9117008",
        "longitude": "75.7123648"
    },
    {
        "timestamp": "2024-06-03T03:21:29.377Z",
        "latitude": "15.8838432",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-03T03:21:37.472Z",
        "latitude": "15.8838096",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-03T03:21:38.689Z",
        "latitude": "15.8840752",
        "longitude": "75.7030784"
    },
    {
        "timestamp": "2024-06-03T03:21:40.670Z",
        "latitude": "15.8814128",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-03T03:21:44.004Z",
        "latitude": "15.8843248",
        "longitude": "75.702848"
    },
    {
        "timestamp": "2024-06-03T03:21:57.006Z",
        "latitude": "15.8829744",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-03T03:21:58.211Z",
        "latitude": "15.8824688",
        "longitude": "75.7027968"
    },
    {
        "timestamp": "2024-06-03T03:22:02.278Z",
        "latitude": "15.8818768",
        "longitude": "75.7024832"
    },
    {
        "timestamp": "2024-06-03T03:22:09.482Z",
        "latitude": "15.8844832",
        "longitude": "75.7032384"
    },
    {
        "timestamp": "2024-06-03T03:22:13.594Z",
        "latitude": "15.883992",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-03T03:22:20.787Z",
        "latitude": "15.8836096",
        "longitude": "75.702464"
    },
    {
        "timestamp": "2024-06-03T03:22:24.866Z",
        "latitude": "15.884192",
        "longitude": "75.7031488"
    },
    {
        "timestamp": "2024-06-03T03:22:32.098Z",
        "latitude": "15.8839792",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T03:22:36.181Z",
        "latitude": "15.8840864",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-03T03:22:43.394Z",
        "latitude": "15.884088",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T03:22:47.465Z",
        "latitude": "15.883872",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-03T03:22:54.688Z",
        "latitude": "15.884192",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-03T03:27:27.968Z",
        "latitude": "15.8838992",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-03T03:27:29.166Z",
        "latitude": "15.8838048",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-03T03:27:39.248Z",
        "latitude": "15.857944",
        "longitude": "75.689536"
    },
    {
        "timestamp": "2024-06-03T03:27:41.555Z",
        "latitude": "15.8837408",
        "longitude": "75.7027904"
    },
    {
        "timestamp": "2024-06-03T03:27:50.120Z",
        "latitude": "15.8721264",
        "longitude": "75.6839808"
    },
    {
        "timestamp": "2024-06-03T03:27:53.452Z",
        "latitude": "15.87312",
        "longitude": "75.682656"
    },
    {
        "timestamp": "2024-06-03T03:27:59.130Z",
        "latitude": "15.8839264",
        "longitude": "75.7028864"
    },
    {
        "timestamp": "2024-06-03T03:28:00.366Z",
        "latitude": "15.8631136",
        "longitude": "75.7087488"
    },
    {
        "timestamp": "2024-06-03T03:28:04.412Z",
        "latitude": "15.8838144",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-03T03:28:06.422Z",
        "latitude": "15.88444",
        "longitude": "75.7015104"
    },
    {
        "timestamp": "2024-06-03T03:28:12.751Z",
        "latitude": "15.8838752",
        "longitude": "75.703456"
    },
    {
        "timestamp": "2024-06-03T03:28:23.317Z",
        "latitude": "15.8837872",
        "longitude": "75.7029184"
    },
    {
        "timestamp": "2024-06-03T03:28:24.541Z",
        "latitude": "15.8838592",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T03:28:25.727Z",
        "latitude": "15.8837696",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-03T03:28:26.968Z",
        "latitude": "15.883872",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-03T03:28:41.041Z",
        "latitude": "15.8842704",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-03T03:28:42.262Z",
        "latitude": "15.8840848",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T03:28:46.327Z",
        "latitude": "15.8837232",
        "longitude": "75.7028288"
    },
    {
        "timestamp": "2024-06-03T03:28:52.557Z",
        "latitude": "15.8839536",
        "longitude": "75.7026112"
    },
    {
        "timestamp": "2024-06-03T03:28:57.643Z",
        "latitude": "15.883704",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T03:29:04.855Z",
        "latitude": "15.8838656",
        "longitude": "75.702688"
    },
    {
        "timestamp": "2024-06-03T03:29:08.931Z",
        "latitude": "15.883832",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-03T03:29:16.164Z",
        "latitude": "15.8837408",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-03T03:29:21.237Z",
        "latitude": "55.6478123",
        "longitude": "364.8716801"
    },
    {
        "timestamp": "2024-06-03T03:29:27.449Z",
        "latitude": "15.8836544",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T03:29:31.541Z",
        "latitude": "15.8835872",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-03T03:29:38.741Z",
        "latitude": "15.883688",
        "longitude": "75.7032256"
    },
    {
        "timestamp": "2024-06-03T03:29:42.840Z",
        "latitude": "15.8838256",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-03T03:29:50.057Z",
        "latitude": "15.883784",
        "longitude": "75.7029184"
    },
    {
        "timestamp": "2024-06-03T03:29:54.140Z",
        "latitude": "15.8837472",
        "longitude": "75.7030848"
    },
    {
        "timestamp": "2024-06-03T03:30:01.334Z",
        "latitude": "15.8836608",
        "longitude": "75.7030336"
    },
    {
        "timestamp": "2024-06-03T03:30:05.409Z",
        "latitude": "15.8700368",
        "longitude": "75.7093632"
    },
    {
        "timestamp": "2024-06-03T04:42:10.037Z",
        "latitude": "15.8840928",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-03T04:42:11.261Z",
        "latitude": "15.8840144",
        "longitude": "75.702912"
    },
    {
        "timestamp": "2024-06-03T04:42:21.306Z",
        "latitude": "15.8843408",
        "longitude": "75.7032768"
    },
    {
        "timestamp": "2024-06-03T04:42:23.637Z",
        "latitude": "15.8843408",
        "longitude": "75.7026496"
    },
    {
        "timestamp": "2024-06-03T04:42:32.202Z",
        "latitude": "15.8843408",
        "longitude": "75.702304"
    },
    {
        "timestamp": "2024-06-03T04:42:35.514Z",
        "latitude": "15.8837632",
        "longitude": "75.7026624"
    },
    {
        "timestamp": "2024-06-03T04:42:43.474Z",
        "latitude": "15.8839568",
        "longitude": "75.7031488"
    },
    {
        "timestamp": "2024-06-03T04:45:12.280Z",
        "latitude": "15.883864",
        "longitude": "75.7028288"
    },
    {
        "timestamp": "2024-06-03T04:45:13.508Z",
        "latitude": "15.8845664",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T04:45:14.738Z",
        "latitude": "15.884128",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-03T04:45:25.893Z",
        "latitude": "15.8841808",
        "longitude": "75.7024896"
    },
    {
        "timestamp": "2024-06-03T04:45:27.109Z",
        "latitude": "15.8842336",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-03T04:45:37.793Z",
        "latitude": "15.8846512",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-03T04:45:38.999Z",
        "latitude": "15.8846032",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-03T04:45:49.081Z",
        "latitude": "15.8835728",
        "longitude": "75.7027008"
    },
    {
        "timestamp": "2024-06-03T04:45:50.288Z",
        "latitude": "15.8839792",
        "longitude": "75.702464"
    },
    {
        "timestamp": "2024-06-03T04:46:00.385Z",
        "latitude": "15.8839584",
        "longitude": "75.7021568"
    },
    {
        "timestamp": "2024-06-03T04:46:01.597Z",
        "latitude": "15.8841152",
        "longitude": "75.702528"
    },
    {
        "timestamp": "2024-06-03T04:46:11.671Z",
        "latitude": "15.8841088",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T04:46:12.895Z",
        "latitude": "15.8840176",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-03T04:46:22.975Z",
        "latitude": "15.884208",
        "longitude": "75.7028736"
    },
    {
        "timestamp": "2024-06-03T04:46:24.207Z",
        "latitude": "15.8839472",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-03T04:46:34.309Z",
        "latitude": "15.8839376",
        "longitude": "75.7026816"
    },
    {
        "timestamp": "2024-06-03T04:46:35.492Z",
        "latitude": "15.883912",
        "longitude": "75.7027264"
    },
    {
        "timestamp": "2024-06-03T04:46:45.588Z",
        "latitude": "15.8840336",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-03T04:46:46.791Z",
        "latitude": "15.8841632",
        "longitude": "75.7032192"
    },
    {
        "timestamp": "2024-06-03T04:46:56.886Z",
        "latitude": "15.884176",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T04:46:58.104Z",
        "latitude": "15.8836992",
        "longitude": "75.7034304"
    },
    {
        "timestamp": "2024-06-03T04:47:08.188Z",
        "latitude": "15.8836336",
        "longitude": "75.7033216"
    },
    {
        "timestamp": "2024-06-03T04:47:09.382Z",
        "latitude": "15.8838464",
        "longitude": "75.702784"
    },
    {
        "timestamp": "2024-06-03T04:47:19.488Z",
        "latitude": "15.8842656",
        "longitude": "75.702624"
    },
    {
        "timestamp": "2024-06-03T04:47:30.793Z",
        "latitude": "15.8840704",
        "longitude": "75.7025664"
    },
    {
        "timestamp": "2024-06-03T04:47:32.030Z",
        "latitude": "15.8840944",
        "longitude": "75.7025088"
    },
    {
        "timestamp": "2024-06-03T04:47:42.075Z",
        "latitude": "15.8842432",
        "longitude": "75.702688"
    },
    {
        "timestamp": "2024-06-03T04:47:43.288Z",
        "latitude": "15.8836864",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-03T04:47:53.394Z",
        "latitude": "15.883784",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-03T04:47:54.584Z",
        "latitude": "15.8837168",
        "longitude": "75.7027968"
    },
    {
        "timestamp": "2024-06-03T04:48:04.699Z",
        "latitude": "15.8836464",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-03T04:48:05.887Z",
        "latitude": "15.8836944",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-03T04:48:13.163Z",
        "latitude": "15.8836176",
        "longitude": "75.703008"
    },
    {
        "timestamp": "2024-06-03T04:48:15.955Z",
        "latitude": "15.8836176",
        "longitude": "75.703008"
    },
    {
        "timestamp": "2024-06-03T04:48:33.133Z",
        "latitude": "15.884408",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-03T04:48:35.981Z",
        "latitude": "15.884408",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-03T04:48:34.341Z",
        "latitude": "15.8835328",
        "longitude": "75.7032256"
    },
    {
        "timestamp": "2024-06-03T04:48:37.201Z",
        "latitude": "15.8835328",
        "longitude": "75.7032256"
    },
    {
        "timestamp": "2024-06-03T04:48:35.628Z",
        "latitude": "15.884136",
        "longitude": "75.7028672"
    },
    {
        "timestamp": "2024-06-03T04:48:38.415Z",
        "latitude": "15.884136",
        "longitude": "75.7028672"
    },
    {
        "timestamp": "2024-06-03T04:48:39.215Z",
        "latitude": "15.8838864",
        "longitude": "75.70352"
    },
    {
        "timestamp": "2024-06-03T04:48:42.084Z",
        "latitude": "15.8838864",
        "longitude": "75.70352"
    },
    {
        "timestamp": "2024-06-03T04:48:40.468Z",
        "latitude": "108.5746859",
        "longitude": "326.5593345"
    },
    {
        "timestamp": "2024-06-03T04:48:43.325Z",
        "latitude": "108.5746859",
        "longitude": "326.5593345"
    },
    {
        "timestamp": "2024-06-03T04:48:41.705Z",
        "latitude": "15.880016",
        "longitude": "75.704736"
    },
    {
        "timestamp": "2024-06-03T04:48:44.543Z",
        "latitude": "15.880016",
        "longitude": "75.704736"
    },
    {
        "timestamp": "2024-06-03T04:49:57.143Z",
        "latitude": "15.880016",
        "longitude": "75.704736"
    },
    {
        "timestamp": "2024-06-03T04:50:00.002Z",
        "latitude": "15.880016",
        "longitude": "75.704736"
    },
    {
        "timestamp": "2024-06-03T04:49:58.850Z",
        "latitude": "15.8844896",
        "longitude": "75.6870144"
    },
    {
        "timestamp": "2024-06-03T04:50:01.708Z",
        "latitude": "15.8844896",
        "longitude": "75.6870144"
    },
    {
        "timestamp": "2024-06-03T04:50:00.063Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T04:50:02.916Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T04:50:08.959Z",
        "latitude": "15.8840288",
        "longitude": "75.7034176"
    },
    {
        "timestamp": "2024-06-03T04:50:11.824Z",
        "latitude": "15.8840288",
        "longitude": "75.7034176"
    },
    {
        "timestamp": "2024-06-03T04:50:10.170Z",
        "latitude": "15.8839296",
        "longitude": "75.7025984"
    },
    {
        "timestamp": "2024-06-03T04:50:13.026Z",
        "latitude": "15.8839296",
        "longitude": "75.7025984"
    },
    {
        "timestamp": "2024-06-03T04:50:19.450Z",
        "latitude": "15.883728",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-03T04:50:22.312Z",
        "latitude": "15.883728",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-03T04:50:20.699Z",
        "latitude": "15.8839216",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-03T04:50:23.550Z",
        "latitude": "15.8839216",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-03T04:50:21.990Z",
        "latitude": "15.8838256",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-03T04:50:24.849Z",
        "latitude": "15.8838256",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-03T04:50:23.188Z",
        "latitude": "15.8838848",
        "longitude": "75.7035776"
    },
    {
        "timestamp": "2024-06-03T04:50:26.050Z",
        "latitude": "15.8838848",
        "longitude": "75.7035776"
    },
    {
        "timestamp": "2024-06-03T04:50:32.513Z",
        "latitude": "15.8838048",
        "longitude": "75.7035392"
    },
    {
        "timestamp": "2024-06-03T04:50:35.386Z",
        "latitude": "15.8838048",
        "longitude": "75.7035392"
    },
    {
        "timestamp": "2024-06-03T04:50:33.748Z",
        "latitude": "15.8837968",
        "longitude": "75.7034176"
    },
    {
        "timestamp": "2024-06-03T04:50:36.575Z",
        "latitude": "15.8837968",
        "longitude": "75.7034176"
    },
    {
        "timestamp": "2024-06-03T04:50:35.022Z",
        "latitude": "15.883792",
        "longitude": "75.7035456"
    },
    {
        "timestamp": "2024-06-03T04:50:37.875Z",
        "latitude": "15.883792",
        "longitude": "75.7035456"
    },
    {
        "timestamp": "2024-06-03T04:50:36.240Z",
        "latitude": "15.8838528",
        "longitude": "75.7034304"
    },
    {
        "timestamp": "2024-06-03T04:50:39.079Z",
        "latitude": "15.8838528",
        "longitude": "75.7034304"
    },
    {
        "timestamp": "2024-06-03T04:50:39.445Z",
        "latitude": "15.8834832",
        "longitude": "75.7035584"
    },
    {
        "timestamp": "2024-06-03T04:50:42.306Z",
        "latitude": "15.8834832",
        "longitude": "75.7035584"
    },
    {
        "timestamp": "2024-06-03T04:50:40.655Z",
        "latitude": "15.8841072",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-03T04:50:43.515Z",
        "latitude": "15.8841072",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-03T04:50:56.540Z",
        "latitude": "15.883768",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-03T04:50:59.384Z",
        "latitude": "15.883768",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-03T04:50:57.743Z",
        "latitude": "15.88444",
        "longitude": "75.7015104"
    },
    {
        "timestamp": "2024-06-03T04:51:00.588Z",
        "latitude": "15.88444",
        "longitude": "75.7015104"
    },
    {
        "timestamp": "2024-06-03T05:11:07.511Z",
        "latitude": "15.8841488",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-03T05:11:10.308Z",
        "latitude": "15.8841488",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-03T05:11:08.710Z",
        "latitude": "15.884104",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-03T05:11:11.519Z",
        "latitude": "15.884104",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-03T05:11:09.930Z",
        "latitude": "15.8841264",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-03T05:11:12.736Z",
        "latitude": "15.8841264",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-03T05:11:11.205Z",
        "latitude": "15.883792",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-03T05:11:14.021Z",
        "latitude": "15.883792",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-03T05:11:12.500Z",
        "latitude": "15.883784",
        "longitude": "75.7031936"
    },
    {
        "timestamp": "2024-06-03T05:11:15.188Z",
        "latitude": "15.883784",
        "longitude": "75.7031936"
    },
    {
        "timestamp": "2024-06-03T05:11:21.665Z",
        "latitude": "15.884136",
        "longitude": "75.7028672"
    },
    {
        "timestamp": "2024-06-03T05:11:24.484Z",
        "latitude": "15.884136",
        "longitude": "75.7028672"
    },
    {
        "timestamp": "2024-06-03T05:11:22.900Z",
        "latitude": "15.8841824",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T05:11:25.710Z",
        "latitude": "15.8841824",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T05:12:29.992Z",
        "latitude": "15.883872",
        "longitude": "75.703584"
    },
    {
        "timestamp": "2024-06-03T05:12:32.805Z",
        "latitude": "15.883872",
        "longitude": "75.703584"
    },
    {
        "timestamp": "2024-06-03T05:12:31.250Z",
        "latitude": "15.8837344",
        "longitude": "75.7034432"
    },
    {
        "timestamp": "2024-06-03T05:12:34.038Z",
        "latitude": "15.8837344",
        "longitude": "75.7034432"
    },
    {
        "timestamp": "2024-06-03T05:14:44.300Z",
        "latitude": "15.8837152",
        "longitude": "75.7026112"
    },
    {
        "timestamp": "2024-06-03T05:14:47.099Z",
        "latitude": "15.8837152",
        "longitude": "75.7026112"
    },
    {
        "timestamp": "2024-06-03T05:16:24.186Z",
        "latitude": "15.8840704",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-03T05:16:26.947Z",
        "latitude": "15.8840704",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-03T05:16:25.423Z",
        "latitude": "15.8840976",
        "longitude": "75.7031936"
    },
    {
        "timestamp": "2024-06-03T05:16:28.198Z",
        "latitude": "15.8840976",
        "longitude": "75.7031936"
    },
    {
        "timestamp": "2024-06-03T05:16:26.623Z",
        "latitude": "15.8841216",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-03T05:16:29.400Z",
        "latitude": "15.8841216",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-03T05:19:55.027Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:19:57.775Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:21:41.770Z",
        "latitude": "15.8838304",
        "longitude": "75.703008"
    },
    {
        "timestamp": "2024-06-03T05:21:44.522Z",
        "latitude": "15.8838304",
        "longitude": "75.703008"
    },
    {
        "timestamp": "2024-06-03T05:23:13.323Z",
        "latitude": "15.8839056",
        "longitude": "75.7032704"
    },
    {
        "timestamp": "2024-06-03T05:23:16.055Z",
        "latitude": "15.8839056",
        "longitude": "75.7032704"
    },
    {
        "timestamp": "2024-06-03T05:26:02.114Z",
        "latitude": "15.8837136",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-03T05:26:04.870Z",
        "latitude": "15.8837136",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-03T05:26:03.329Z",
        "latitude": "15.8843296",
        "longitude": "75.7025344"
    },
    {
        "timestamp": "2024-06-03T05:26:06.095Z",
        "latitude": "15.8843296",
        "longitude": "75.7025344"
    },
    {
        "timestamp": "2024-06-03T05:26:04.594Z",
        "latitude": "15.8840928",
        "longitude": "75.7028096"
    },
    {
        "timestamp": "2024-06-03T05:26:07.326Z",
        "latitude": "15.8840928",
        "longitude": "75.7028096"
    },
    {
        "timestamp": "2024-06-03T05:26:23.159Z",
        "latitude": "15.8835696",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-03T05:26:24.391Z",
        "latitude": "15.884248",
        "longitude": "75.7026304"
    },
    {
        "timestamp": "2024-06-03T05:26:33.215Z",
        "latitude": "15.8834512",
        "longitude": "75.702624"
    },
    {
        "timestamp": "2024-06-03T05:26:46.925Z",
        "latitude": "15.8844608",
        "longitude": "75.7037632"
    },
    {
        "timestamp": "2024-06-03T05:27:36.927Z",
        "latitude": "15.8840416",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T05:31:08.515Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:31:11.223Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:31:51.779Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:31:54.525Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:32:08.931Z",
        "latitude": "15.8841488",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-03T05:33:33.185Z",
        "latitude": "15.8839456",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-03T05:37:04.022Z",
        "latitude": "15.8840576",
        "longitude": "75.7032448"
    },
    {
        "timestamp": "2024-06-03T05:37:48.230Z",
        "latitude": "15.883752",
        "longitude": "75.7021632"
    },
    {
        "timestamp": "2024-06-03T05:37:49.452Z",
        "latitude": "15.8839264",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-03T05:37:50.686Z",
        "latitude": "15.8836176",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-03T05:38:00.854Z",
        "latitude": "15.8837408",
        "longitude": "75.7026304"
    },
    {
        "timestamp": "2024-06-03T05:38:02.630Z",
        "latitude": "15.8842768",
        "longitude": "75.7030336"
    },
    {
        "timestamp": "2024-06-03T05:38:12.748Z",
        "latitude": "15.8840064",
        "longitude": "75.70288"
    },
    {
        "timestamp": "2024-06-03T05:38:13.955Z",
        "latitude": "15.8842816",
        "longitude": "75.7029184"
    },
    {
        "timestamp": "2024-06-03T05:38:24.053Z",
        "latitude": "15.8837984",
        "longitude": "75.7026624"
    },
    {
        "timestamp": "2024-06-03T05:38:25.277Z",
        "latitude": "15.8836992",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-03T05:38:36.369Z",
        "latitude": "15.8838704",
        "longitude": "75.7013568"
    },
    {
        "timestamp": "2024-06-03T05:38:37.559Z",
        "latitude": "15.8834832",
        "longitude": "75.699936"
    },
    {
        "timestamp": "2024-06-03T05:38:46.652Z",
        "latitude": "15.8843536",
        "longitude": "75.7034048"
    },
    {
        "timestamp": "2024-06-03T05:39:43.955Z",
        "latitude": "15.8840224",
        "longitude": "75.7032448"
    },
    {
        "timestamp": "2024-06-03T05:39:45.200Z",
        "latitude": "15.8841136",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-03T05:39:54.246Z",
        "latitude": "15.8839248",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-03T05:39:57.576Z",
        "latitude": "15.8842112",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T05:40:05.164Z",
        "latitude": "15.8842352",
        "longitude": "75.702912"
    },
    {
        "timestamp": "2024-06-03T05:40:09.467Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:40:16.442Z",
        "latitude": "15.884104",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-03T05:40:19.774Z",
        "latitude": "15.8838592",
        "longitude": "75.7027968"
    },
    {
        "timestamp": "2024-06-03T05:40:28.740Z",
        "latitude": "15.8843056",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-03T05:40:32.054Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:40:40.033Z",
        "latitude": "15.8834448",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-03T05:40:38.653Z",
        "latitude": "15.8374432",
        "longitude": "74.5061184"
    },
    {
        "timestamp": "2024-06-03T05:40:41.390Z",
        "latitude": "15.8374432",
        "longitude": "74.5061184"
    },
    {
        "timestamp": "2024-06-03T05:40:43.349Z",
        "latitude": "15.884024",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-03T05:40:56.334Z",
        "latitude": "15.8840672",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-03T05:40:57.537Z",
        "latitude": "15.8838976",
        "longitude": "75.7028032"
    },
    {
        "timestamp": "2024-06-03T05:41:01.631Z",
        "latitude": "15.8840832",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-03T05:41:09.845Z",
        "latitude": "15.88428",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T05:41:13.945Z",
        "latitude": "15.8839984",
        "longitude": "75.7035904"
    },
    {
        "timestamp": "2024-06-03T05:41:21.173Z",
        "latitude": "15.8840912",
        "longitude": "75.7025984"
    },
    {
        "timestamp": "2024-06-03T05:41:25.242Z",
        "latitude": "15.8833696",
        "longitude": "75.7042816"
    },
    {
        "timestamp": "2024-06-03T05:41:32.452Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:41:35.541Z",
        "latitude": "15.88432",
        "longitude": "75.7026688"
    },
    {
        "timestamp": "2024-06-03T05:41:42.740Z",
        "latitude": "15.8838208",
        "longitude": "75.7024832"
    },
    {
        "timestamp": "2024-06-03T05:41:47.829Z",
        "latitude": "15.884272",
        "longitude": "75.701792"
    },
    {
        "timestamp": "2024-06-03T05:41:55.043Z",
        "latitude": "15.8839536",
        "longitude": "75.703424"
    },
    {
        "timestamp": "2024-06-03T05:41:59.122Z",
        "latitude": "15.8840672",
        "longitude": "75.702464"
    },
    {
        "timestamp": "2024-06-03T05:42:06.330Z",
        "latitude": "15.884176",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-03T05:42:09.435Z",
        "latitude": "15.8840032",
        "longitude": "75.7021248"
    },
    {
        "timestamp": "2024-06-03T05:42:16.715Z",
        "latitude": "15.8841088",
        "longitude": "75.7026752"
    },
    {
        "timestamp": "2024-06-03T05:42:20.719Z",
        "latitude": "15.8838528",
        "longitude": "75.7024832"
    },
    {
        "timestamp": "2024-06-03T05:42:36.096Z",
        "latitude": "15.8839488",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-03T05:44:06.091Z",
        "latitude": "15.8839568",
        "longitude": "75.7026048"
    },
    {
        "timestamp": "2024-06-03T05:47:38.431Z",
        "latitude": "15.837496",
        "longitude": "74.5063104"
    },
    {
        "timestamp": "2024-06-03T05:47:41.137Z",
        "latitude": "15.837496",
        "longitude": "74.5063104"
    },
    {
        "timestamp": "2024-06-03T05:47:45.404Z",
        "latitude": "15.8838464",
        "longitude": "75.7031168"
    },
    {
        "timestamp": "2024-06-03T05:47:49.183Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-03T05:47:51.895Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-03T05:49:55.579Z",
        "latitude": "15.8837408",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-03T05:49:56.800Z",
        "latitude": "15.883848",
        "longitude": "75.7029184"
    },
    {
        "timestamp": "2024-06-03T05:50:06.865Z",
        "latitude": "15.8856176",
        "longitude": "75.7039936"
    },
    {
        "timestamp": "2024-06-03T05:50:09.187Z",
        "latitude": "15.8862016",
        "longitude": "75.7023872"
    },
    {
        "timestamp": "2024-06-03T05:50:18.769Z",
        "latitude": "15.8844656",
        "longitude": "75.7033088"
    },
    {
        "timestamp": "2024-06-03T05:50:21.084Z",
        "latitude": "15.8834352",
        "longitude": "75.7025664"
    },
    {
        "timestamp": "2024-06-03T05:50:30.061Z",
        "latitude": "15.8837632",
        "longitude": "75.702752"
    },
    {
        "timestamp": "2024-06-03T05:50:32.379Z",
        "latitude": "15.8837152",
        "longitude": "75.7026112"
    },
    {
        "timestamp": "2024-06-03T05:50:40.361Z",
        "latitude": "15.8834512",
        "longitude": "75.702624"
    },
    {
        "timestamp": "2024-06-03T05:50:42.722Z",
        "latitude": "15.8843536",
        "longitude": "75.7034048"
    },
    {
        "timestamp": "2024-06-03T05:52:54.493Z",
        "latitude": "15.8837616",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T05:54:17.938Z",
        "latitude": "15.8839632",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T05:55:06.037Z",
        "latitude": "15.883768",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-03T05:58:04.549Z",
        "latitude": "15.883696",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-03T05:58:21.044Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T05:58:23.769Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T06:01:15.087Z",
        "latitude": "15.8837904",
        "longitude": "75.7027072"
    },
    {
        "timestamp": "2024-06-03T06:03:12.955Z",
        "latitude": "15.884032",
        "longitude": "75.7030976"
    },
    {
        "timestamp": "2024-06-03T06:05:25.570Z",
        "latitude": "15.8838304",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-03T06:06:15.930Z",
        "latitude": "15.884128",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-03T06:06:17.179Z",
        "latitude": "15.8840768",
        "longitude": "75.702464"
    },
    {
        "timestamp": "2024-06-03T06:06:27.241Z",
        "latitude": "15.883984",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-03T06:06:29.562Z",
        "latitude": "15.8836624",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-03T06:06:38.146Z",
        "latitude": "15.8838496",
        "longitude": "75.7028608"
    },
    {
        "timestamp": "2024-06-03T06:06:41.455Z",
        "latitude": "15.8841904",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-03T06:06:49.419Z",
        "latitude": "15.8836176",
        "longitude": "75.703008"
    },
    {
        "timestamp": "2024-06-03T06:06:51.748Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T06:07:01.717Z",
        "latitude": "15.8840496",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-03T06:07:04.050Z",
        "latitude": "15.8840416",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-03T06:07:13.030Z",
        "latitude": "15.8838528",
        "longitude": "75.7024832"
    },
    {
        "timestamp": "2024-06-03T06:07:15.334Z",
        "latitude": "15.8839632",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T06:08:23.610Z",
        "latitude": "15.8840544",
        "longitude": "75.7025216"
    },
    {
        "timestamp": "2024-06-03T06:08:40.902Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T06:08:43.612Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T06:11:33.143Z",
        "latitude": "15.8837664",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-03T06:13:31.111Z",
        "latitude": "15.8837696",
        "longitude": "75.7031424"
    },
    {
        "timestamp": "2024-06-03T06:15:36.206Z",
        "latitude": "15.8376",
        "longitude": "74.505888"
    },
    {
        "timestamp": "2024-06-03T06:15:38.892Z",
        "latitude": "15.8376",
        "longitude": "74.505888"
    },
    {
        "timestamp": "2024-06-03T06:15:44.290Z",
        "latitude": "15.8840384",
        "longitude": "75.7028288"
    },
    {
        "timestamp": "2024-06-03T06:18:41.761Z",
        "latitude": "15.8842656",
        "longitude": "75.7031424"
    },
    {
        "timestamp": "2024-06-03T06:20:38.152Z",
        "latitude": "15.883832",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-03T06:21:51.171Z",
        "latitude": "15.8840864",
        "longitude": "75.7027264"
    },
    {
        "timestamp": "2024-06-03T06:23:49.234Z",
        "latitude": "15.8841776",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-03T06:26:04.207Z",
        "latitude": "15.8839472",
        "longitude": "75.702912"
    },
    {
        "timestamp": "2024-06-03T06:26:14.213Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T06:26:16.867Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T06:28:58.893Z",
        "latitude": "15.8832512",
        "longitude": "75.7008832"
    },
    {
        "timestamp": "2024-06-03T06:32:09.206Z",
        "latitude": "15.8835184",
        "longitude": "75.7024448"
    },
    {
        "timestamp": "2024-06-03T06:32:35.927Z",
        "latitude": "15.8843472",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-03T06:34:06.396Z",
        "latitude": "15.8836336",
        "longitude": "75.7031552"
    },
    {
        "timestamp": "2024-06-03T06:36:23.338Z",
        "latitude": "15.8838352",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-03T06:36:49.510Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T06:39:26.918Z",
        "latitude": "15.88412",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-03T06:42:28.286Z",
        "latitude": "15.884112",
        "longitude": "75.7028864"
    },
    {
        "timestamp": "2024-06-03T06:44:15.945Z",
        "latitude": "15.8844416",
        "longitude": "75.702848"
    },
    {
        "timestamp": "2024-06-03T06:44:24.577Z",
        "latitude": "15.8842016",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-03T06:46:42.573Z",
        "latitude": "15.8839296",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-03T06:47:16.806Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T06:57:37.851Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T07:08:07.572Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-03T07:17:17.151Z",
        "latitude": "15.8374272",
        "longitude": "74.5064448"
    },
    {
        "timestamp": "2024-06-03T07:23:41.799Z",
        "latitude": "15.884712",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-03T07:24:22.207Z",
        "latitude": "15.837472",
        "longitude": "74.505472"
    },
    {
        "timestamp": "2024-06-03T07:26:39.561Z",
        "latitude": "15.8839008",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-03T07:28:29.683Z",
        "latitude": "15.8842224",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T07:30:59.140Z",
        "latitude": "15.8839056",
        "longitude": "75.7030272"
    },
    {
        "timestamp": "2024-06-03T07:32:03.765Z",
        "latitude": "15.8841728",
        "longitude": "75.7026112"
    },
    {
        "timestamp": "2024-06-03T07:34:12.147Z",
        "latitude": "15.8841632",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-03T07:38:07.992Z",
        "latitude": "15.8838256",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-03T07:39:59.922Z",
        "latitude": "15.8840288",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-03T07:42:14.517Z",
        "latitude": "15.8839504",
        "longitude": "75.702688"
    },
    {
        "timestamp": "2024-06-03T07:43:32.607Z",
        "latitude": "15.8839776",
        "longitude": "75.7035584"
    },
    {
        "timestamp": "2024-06-03T07:45:26.415Z",
        "latitude": "15.8839504",
        "longitude": "75.7035456"
    },
    {
        "timestamp": "2024-06-03T07:49:45.302Z",
        "latitude": "15.8838752",
        "longitude": "75.7029184"
    },
    {
        "timestamp": "2024-06-03T07:51:41.624Z",
        "latitude": "15.8838032",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-03T07:53:56.472Z",
        "latitude": "15.883992",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-03T07:55:21.433Z",
        "latitude": "15.884232",
        "longitude": "75.702528"
    },
    {
        "timestamp": "2024-06-03T07:56:53.359Z",
        "latitude": "15.8841328",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-03T08:01:13.989Z",
        "latitude": "15.88384",
        "longitude": "75.7027968"
    },
    {
        "timestamp": "2024-06-03T08:03:23.579Z",
        "latitude": "15.8837296",
        "longitude": "75.7031424"
    },
    {
        "timestamp": "2024-06-03T08:05:45.760Z",
        "latitude": "15.8840064",
        "longitude": "75.7028288"
    },
    {
        "timestamp": "2024-06-03T08:07:43.743Z",
        "latitude": "15.8839056",
        "longitude": "75.7027264"
    },
    {
        "timestamp": "2024-06-03T08:08:31.695Z",
        "latitude": "15.8839888",
        "longitude": "75.7034368"
    },
    {
        "timestamp": "2024-06-03T08:08:39.389Z",
        "latitude": "15.883576",
        "longitude": "75.7016448"
    },
    {
        "timestamp": "2024-06-03T08:12:47.384Z",
        "latitude": "15.883704",
        "longitude": "75.7028608"
    },
    {
        "timestamp": "2024-06-03T08:15:10.302Z",
        "latitude": "15.8838736",
        "longitude": "75.702912"
    },
    {
        "timestamp": "2024-06-03T08:17:23.229Z",
        "latitude": "15.883984",
        "longitude": "75.7034432"
    },
    {
        "timestamp": "2024-06-03T08:17:29.029Z",
        "latitude": "15.8839056",
        "longitude": "75.7030272"
    },
    {
        "timestamp": "2024-06-03T08:19:44.266Z",
        "latitude": "15.8836896",
        "longitude": "75.7029184"
    },
    {
        "timestamp": "2024-06-03T08:20:05.323Z",
        "latitude": "15.8840224",
        "longitude": "75.703424"
    },
    {
        "timestamp": "2024-06-03T17:09:32.740Z",
        "latitude": "15.8835984",
        "longitude": "75.7026048"
    },
    {
        "timestamp": "2024-06-03T17:09:47.070Z",
        "latitude": "15.8840112",
        "longitude": "75.7025728"
    },
    {
        "timestamp": "2024-06-03T17:10:24.289Z",
        "latitude": "15.883816",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-04T04:12:27.952Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T04:21:02.379Z",
        "latitude": "15.8377392",
        "longitude": "74.5047552"
    },
    {
        "timestamp": "2024-06-04T04:29:10.658Z",
        "latitude": "15.837368",
        "longitude": "74.50672"
    },
    {
        "timestamp": "2024-06-04T04:36:30.268Z",
        "latitude": "15.8375088",
        "longitude": "74.50576"
    },
    {
        "timestamp": "2024-06-04T04:44:10.300Z",
        "latitude": "15.8374048",
        "longitude": "74.5059072"
    },
    {
        "timestamp": "2024-06-04T04:48:46.173Z",
        "latitude": "15.837448",
        "longitude": "74.5056128"
    },
    {
        "timestamp": "2024-06-04T04:48:48.973Z",
        "latitude": "15.837448",
        "longitude": "74.5056128"
    },
    {
        "timestamp": "2024-06-04T04:51:52.733Z",
        "latitude": "15.8378368",
        "longitude": "74.5127808"
    },
    {
        "timestamp": "2024-06-04T04:54:12.343Z",
        "latitude": "15.8374832",
        "longitude": "74.503616"
    },
    {
        "timestamp": "2024-06-04T04:58:47.866Z",
        "latitude": "15.8375568",
        "longitude": "74.5067776"
    },
    {
        "timestamp": "2024-06-04T05:01:18.584Z",
        "latitude": "15.8375536",
        "longitude": "74.505376"
    },
    {
        "timestamp": "2024-06-04T05:07:59.697Z",
        "latitude": "15.8376896",
        "longitude": "74.5058624"
    },
    {
        "timestamp": "2024-06-04T05:12:06.496Z",
        "latitude": "15.8373136",
        "longitude": "74.5059968"
    },
    {
        "timestamp": "2024-06-04T05:15:10.956Z",
        "latitude": "15.8374096",
        "longitude": "74.5065408"
    },
    {
        "timestamp": "2024-06-04T05:17:18.877Z",
        "latitude": "15.8375824",
        "longitude": "74.5061184"
    },
    {
        "timestamp": "2024-06-04T05:19:21.417Z",
        "latitude": "15.8373632",
        "longitude": "74.5063232"
    },
    {
        "timestamp": "2024-06-04T05:22:02.958Z",
        "latitude": "15.836968",
        "longitude": "74.50656"
    },
    {
        "timestamp": "2024-06-04T05:26:44.249Z",
        "latitude": "15.8370368",
        "longitude": "74.5065408"
    },
    {
        "timestamp": "2024-06-04T05:29:03.904Z",
        "latitude": "15.8368064",
        "longitude": "74.5083392"
    },
    {
        "timestamp": "2024-06-04T05:36:02.033Z",
        "latitude": "15.8377136",
        "longitude": "74.5061184"
    },
    {
        "timestamp": "2024-06-04T11:18:07.489Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:18:08.746Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:18:09.987Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:18:25.616Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:18:26.872Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:19:11.673Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:19:12.967Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:19:14.221Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:19:41.542Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:19:42.817Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:20:40.209Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:20:41.479Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:20:42.737Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:21:41.593Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:21:42.865Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:21:44.115Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:21:58.758Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:21:59.994Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:22:14.148Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:22:15.398Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:22:28.037Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:22:30.289Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:22:43.953Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:22:45.188Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:22:58.827Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:23:00.075Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:23:13.733Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:23:14.975Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:23:28.619Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:23:29.867Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:23:43.530Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:23:44.770Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:23:58.410Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:23:59.666Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:24:13.304Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:24:14.562Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T11:24:28.207Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:24:30.445Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T11:27:06.202Z",
        "latitude": "",
        "longitude": ""
    },
    {
        "timestamp": "2024-06-04T14:42:47.333Z",
        "latitude": "15.8835424",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-04T14:42:48.534Z",
        "latitude": "15.8840624",
        "longitude": "75.7027456"
    },
    {
        "timestamp": "2024-06-04T14:42:49.769Z",
        "latitude": "15.8833392",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-04T14:43:00.913Z",
        "latitude": "15.8838656",
        "longitude": "75.7034304"
    },
    {
        "timestamp": "2024-06-04T14:43:02.148Z",
        "latitude": "15.8833792",
        "longitude": "75.7028736"
    },
    {
        "timestamp": "2024-06-04T14:43:12.813Z",
        "latitude": "15.8827088",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T14:43:14.033Z",
        "latitude": "15.8831344",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-04T14:43:24.108Z",
        "latitude": "15.8831904",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T14:43:25.304Z",
        "latitude": "15.8838",
        "longitude": "75.702784"
    },
    {
        "timestamp": "2024-06-04T14:43:35.404Z",
        "latitude": "15.88356",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-04T14:43:36.604Z",
        "latitude": "15.8837984",
        "longitude": "75.7028736"
    },
    {
        "timestamp": "2024-06-04T14:43:46.699Z",
        "latitude": "15.8836112",
        "longitude": "75.702688"
    },
    {
        "timestamp": "2024-06-04T14:43:47.926Z",
        "latitude": "15.8835712",
        "longitude": "75.70256"
    },
    {
        "timestamp": "2024-06-04T14:43:58.009Z",
        "latitude": "15.8835216",
        "longitude": "75.7024448"
    },
    {
        "timestamp": "2024-06-04T14:43:59.203Z",
        "latitude": "15.8842368",
        "longitude": "75.7025472"
    },
    {
        "timestamp": "2024-06-04T14:44:10.285Z",
        "latitude": "15.8835776",
        "longitude": "75.702624"
    },
    {
        "timestamp": "2024-06-04T14:44:11.511Z",
        "latitude": "15.8835776",
        "longitude": "75.7027648"
    },
    {
        "timestamp": "2024-06-04T14:44:21.597Z",
        "latitude": "15.8838144",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T14:44:22.807Z",
        "latitude": "15.883968",
        "longitude": "75.7033792"
    },
    {
        "timestamp": "2024-06-04T14:44:31.888Z",
        "latitude": "15.8841312",
        "longitude": "75.7036608"
    },
    {
        "timestamp": "2024-06-04T14:44:34.100Z",
        "latitude": "15.8832416",
        "longitude": "75.7039552"
    },
    {
        "timestamp": "2024-06-04T14:44:43.194Z",
        "latitude": "15.8841488",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-04T14:44:46.404Z",
        "latitude": "15.8837056",
        "longitude": "75.703232"
    },
    {
        "timestamp": "2024-06-04T14:44:54.491Z",
        "latitude": "15.8843232",
        "longitude": "75.7030784"
    },
    {
        "timestamp": "2024-06-04T14:44:56.691Z",
        "latitude": "15.8839936",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T14:45:05.785Z",
        "latitude": "15.8839552",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T14:45:08.018Z",
        "latitude": "15.8839888",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-04T14:45:17.089Z",
        "latitude": "15.883984",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T14:45:19.296Z",
        "latitude": "15.8842416",
        "longitude": "75.7033152"
    },
    {
        "timestamp": "2024-06-04T14:45:28.389Z",
        "latitude": "108.0241835",
        "longitude": "218.5297921"
    },
    {
        "timestamp": "2024-06-04T14:45:30.618Z",
        "latitude": "15.8836944",
        "longitude": "75.7027648"
    },
    {
        "timestamp": "2024-06-04T14:45:39.696Z",
        "latitude": "15.8832432",
        "longitude": "75.7035968"
    },
    {
        "timestamp": "2024-06-04T14:45:41.890Z",
        "latitude": "15.88396",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-04T14:45:50.976Z",
        "latitude": "15.883744",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-04T14:45:53.215Z",
        "latitude": "15.882584",
        "longitude": "75.7035584"
    },
    {
        "timestamp": "2024-06-04T14:46:03.282Z",
        "latitude": "15.8839456",
        "longitude": "75.7030848"
    },
    {
        "timestamp": "2024-06-04T14:46:05.513Z",
        "latitude": "15.8839056",
        "longitude": "75.7027904"
    },
    {
        "timestamp": "2024-06-04T14:46:13.581Z",
        "latitude": "15.88388",
        "longitude": "75.70288"
    },
    {
        "timestamp": "2024-06-04T14:46:15.792Z",
        "latitude": "15.8842752",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T14:46:24.895Z",
        "latitude": "15.8837632",
        "longitude": "75.7027264"
    },
    {
        "timestamp": "2024-06-04T14:46:27.113Z",
        "latitude": "15.8837056",
        "longitude": "75.702784"
    },
    {
        "timestamp": "2024-06-04T14:46:36.180Z",
        "latitude": "15.883272",
        "longitude": "75.7025536"
    },
    {
        "timestamp": "2024-06-04T14:46:39.412Z",
        "latitude": "15.88332",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T14:46:47.494Z",
        "latitude": "15.8838112",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-04T14:46:49.714Z",
        "latitude": "15.8839616",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T14:46:58.784Z",
        "latitude": "15.883672",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-04T14:47:01.004Z",
        "latitude": "15.8840064",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T14:47:11.141Z",
        "latitude": "15.8834608",
        "longitude": "75.7030336"
    },
    {
        "timestamp": "2024-06-04T14:47:12.284Z",
        "latitude": "15.883968",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-04T14:47:22.367Z",
        "latitude": "15.8835984",
        "longitude": "75.7027008"
    },
    {
        "timestamp": "2024-06-04T14:47:23.591Z",
        "latitude": "15.8838048",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-04T14:47:32.690Z",
        "latitude": "15.8840624",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-04T14:47:34.899Z",
        "latitude": "15.8836512",
        "longitude": "75.7026624"
    },
    {
        "timestamp": "2024-06-04T14:47:43.962Z",
        "latitude": "15.8838992",
        "longitude": "75.7026752"
    },
    {
        "timestamp": "2024-06-04T14:47:46.188Z",
        "latitude": "15.88384",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T14:47:56.260Z",
        "latitude": "15.8837344",
        "longitude": "75.7026432"
    },
    {
        "timestamp": "2024-06-04T14:47:58.503Z",
        "latitude": "15.8839728",
        "longitude": "75.7027264"
    },
    {
        "timestamp": "2024-06-04T14:48:06.586Z",
        "latitude": "15.8838384",
        "longitude": "75.702688"
    },
    {
        "timestamp": "2024-06-04T14:48:08.781Z",
        "latitude": "15.8836912",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-04T14:48:17.866Z",
        "latitude": "15.8838192",
        "longitude": "75.7026688"
    },
    {
        "timestamp": "2024-06-04T14:48:20.086Z",
        "latitude": "15.883792",
        "longitude": "75.702688"
    },
    {
        "timestamp": "2024-06-04T14:48:29.230Z",
        "latitude": "15.883776",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-04T14:48:38.482Z",
        "latitude": "15.8841312",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-04T14:48:39.720Z",
        "latitude": "15.8833312",
        "longitude": "75.7013376"
    },
    {
        "timestamp": "2024-06-04T14:48:40.935Z",
        "latitude": "15.8843008",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T14:48:52.126Z",
        "latitude": "15.884712",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-04T14:48:53.292Z",
        "latitude": "15.8841632",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T14:49:03.009Z",
        "latitude": "15.8840352",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T14:49:07.161Z",
        "latitude": "15.8842752",
        "longitude": "75.703296"
    },
    {
        "timestamp": "2024-06-04T14:49:08.385Z",
        "latitude": "15.8841136",
        "longitude": "75.7030272"
    },
    {
        "timestamp": "2024-06-04T14:49:09.608Z",
        "latitude": "15.8839792",
        "longitude": "75.7031552"
    },
    {
        "timestamp": "2024-06-04T14:49:21.774Z",
        "latitude": "15.8838432",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-04T14:49:22.979Z",
        "latitude": "15.8836608",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-04T14:49:28.682Z",
        "latitude": "15.8836112",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-04T14:49:34.290Z",
        "latitude": "15.8838528",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T14:49:40.953Z",
        "latitude": "15.8839296",
        "longitude": "75.7028352"
    },
    {
        "timestamp": "2024-06-04T14:49:45.588Z",
        "latitude": "15.8836672",
        "longitude": "75.703008"
    },
    {
        "timestamp": "2024-06-04T14:49:51.261Z",
        "latitude": "15.883616",
        "longitude": "75.7034432"
    },
    {
        "timestamp": "2024-06-04T14:50:03.551Z",
        "latitude": "15.883704",
        "longitude": "75.7033024"
    },
    {
        "timestamp": "2024-06-04T14:50:08.186Z",
        "latitude": "15.8837792",
        "longitude": "75.7035072"
    },
    {
        "timestamp": "2024-06-04T14:50:13.862Z",
        "latitude": "15.8840224",
        "longitude": "75.7035072"
    },
    {
        "timestamp": "2024-06-04T14:50:19.465Z",
        "latitude": "15.8837072",
        "longitude": "75.7034688"
    },
    {
        "timestamp": "2024-06-04T14:50:25.170Z",
        "latitude": "15.8839504",
        "longitude": "75.7033856"
    },
    {
        "timestamp": "2024-06-04T14:50:30.786Z",
        "latitude": "15.8840128",
        "longitude": "75.7033216"
    },
    {
        "timestamp": "2024-06-04T14:50:36.456Z",
        "latitude": "15.8841072",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-04T14:50:42.074Z",
        "latitude": "15.8840784",
        "longitude": "75.7032832"
    },
    {
        "timestamp": "2024-06-04T14:50:47.759Z",
        "latitude": "15.8842272",
        "longitude": "75.7031424"
    },
    {
        "timestamp": "2024-06-04T14:50:53.384Z",
        "latitude": "15.8838288",
        "longitude": "75.7031552"
    },
    {
        "timestamp": "2024-06-04T14:50:59.058Z",
        "latitude": "15.8839056",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-04T14:51:04.690Z",
        "latitude": "15.8841664",
        "longitude": "75.7028608"
    },
    {
        "timestamp": "2024-06-04T14:51:10.346Z",
        "latitude": "15.8839936",
        "longitude": "75.7031232"
    },
    {
        "timestamp": "2024-06-04T14:51:15.987Z",
        "latitude": "15.8838048",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-04T14:51:27.660Z",
        "latitude": "15.8838336",
        "longitude": "75.7031552"
    },
    {
        "timestamp": "2024-06-04T14:51:32.959Z",
        "latitude": "15.883672",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-04T14:51:38.965Z",
        "latitude": "15.8838208",
        "longitude": "75.7031232"
    },
    {
        "timestamp": "2024-06-04T14:51:44.256Z",
        "latitude": "15.8836992",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T14:51:51.265Z",
        "latitude": "15.88368",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T14:52:02.575Z",
        "latitude": "15.88416",
        "longitude": "75.7034176"
    },
    {
        "timestamp": "2024-06-04T14:52:06.843Z",
        "latitude": "15.8836944",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-04T14:52:12.849Z",
        "latitude": "15.883864",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T14:52:18.154Z",
        "latitude": "15.88344",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T14:52:24.167Z",
        "latitude": "15.8837392",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-04T14:52:35.466Z",
        "latitude": "15.8837856",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T14:52:40.735Z",
        "latitude": "15.8836656",
        "longitude": "75.7029184"
    },
    {
        "timestamp": "2024-06-04T14:52:46.734Z",
        "latitude": "15.8836272",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-04T14:52:52.046Z",
        "latitude": "15.8836208",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T14:52:58.060Z",
        "latitude": "15.8836032",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T14:53:03.344Z",
        "latitude": "15.8836624",
        "longitude": "75.7028864"
    },
    {
        "timestamp": "2024-06-04T14:53:09.342Z",
        "latitude": "15.8836464",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-04T14:53:14.651Z",
        "latitude": "15.8843808",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-04T14:53:20.667Z",
        "latitude": "15.8840576",
        "longitude": "75.7028032"
    },
    {
        "timestamp": "2024-06-04T14:53:25.944Z",
        "latitude": "15.8842272",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-04T14:53:31.928Z",
        "latitude": "15.883984",
        "longitude": "75.70288"
    },
    {
        "timestamp": "2024-06-04T14:53:37.235Z",
        "latitude": "15.8841824",
        "longitude": "75.7032448"
    },
    {
        "timestamp": "2024-06-04T14:53:44.254Z",
        "latitude": "15.8839424",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T14:53:49.562Z",
        "latitude": "15.88428",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-04T14:53:54.578Z",
        "latitude": "15.8839728",
        "longitude": "75.7028352"
    },
    {
        "timestamp": "2024-06-04T14:53:59.835Z",
        "latitude": "15.8838608",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T14:54:05.833Z",
        "latitude": "15.8843184",
        "longitude": "75.7033024"
    },
    {
        "timestamp": "2024-06-04T14:54:11.146Z",
        "latitude": "15.88412",
        "longitude": "75.7028992"
    },
    {
        "timestamp": "2024-06-04T14:54:17.139Z",
        "latitude": "15.8840112",
        "longitude": "75.7028672"
    },
    {
        "timestamp": "2024-06-04T14:54:22.428Z",
        "latitude": "15.8841968",
        "longitude": "75.7028736"
    },
    {
        "timestamp": "2024-06-04T14:54:28.433Z",
        "latitude": "15.8837312",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T14:54:33.752Z",
        "latitude": "15.8837408",
        "longitude": "75.7028672"
    },
    {
        "timestamp": "2024-06-04T14:54:39.740Z",
        "latitude": "15.8840176",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T14:54:45.028Z",
        "latitude": "15.8838112",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T14:54:51.040Z",
        "latitude": "15.8836",
        "longitude": "75.7030272"
    },
    {
        "timestamp": "2024-06-04T14:54:56.357Z",
        "latitude": "15.8839136",
        "longitude": "75.702848"
    },
    {
        "timestamp": "2024-06-04T14:55:02.333Z",
        "latitude": "15.883856",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-04T14:55:07.642Z",
        "latitude": "15.8855232",
        "longitude": "75.701216"
    },
    {
        "timestamp": "2024-06-04T14:55:13.650Z",
        "latitude": "15.8837696",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T14:55:18.911Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T14:55:24.946Z",
        "latitude": "2.318824",
        "longitude": "107.5072896"
    },
    {
        "timestamp": "2024-06-04T14:55:30.213Z",
        "latitude": "15.8836496",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-04T14:55:37.215Z",
        "latitude": "15.8839056",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-04T14:55:42.513Z",
        "latitude": "15.8838896",
        "longitude": "75.7030272"
    },
    {
        "timestamp": "2024-06-04T14:55:47.548Z",
        "latitude": "15.8840224",
        "longitude": "75.7028096"
    },
    {
        "timestamp": "2024-06-04T14:55:52.823Z",
        "latitude": "15.8835152",
        "longitude": "75.7033792"
    },
    {
        "timestamp": "2024-06-04T14:55:58.816Z",
        "latitude": "15.884136",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-04T14:56:04.118Z",
        "latitude": "15.88392",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-04T14:56:11.113Z",
        "latitude": "15.8840944",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-04T14:56:16.423Z",
        "latitude": "15.8835712",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T14:56:21.450Z",
        "latitude": "15.883536",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T14:56:26.715Z",
        "latitude": "15.8837792",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-04T14:56:32.713Z",
        "latitude": "15.883728",
        "longitude": "75.7032384"
    },
    {
        "timestamp": "2024-06-04T14:56:38.033Z",
        "latitude": "15.8838816",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-04T14:56:44.019Z",
        "latitude": "15.8838528",
        "longitude": "75.7031936"
    },
    {
        "timestamp": "2024-06-04T14:56:49.308Z",
        "latitude": "15.8836896",
        "longitude": "75.7028672"
    },
    {
        "timestamp": "2024-06-04T14:56:56.314Z",
        "latitude": "15.8835808",
        "longitude": "75.7031424"
    },
    {
        "timestamp": "2024-06-04T14:57:00.625Z",
        "latitude": "15.8838352",
        "longitude": "75.7031232"
    },
    {
        "timestamp": "2024-06-04T14:57:06.628Z",
        "latitude": "15.8837712",
        "longitude": "75.703008"
    },
    {
        "timestamp": "2024-06-04T14:57:11.920Z",
        "latitude": "15.883808",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T14:57:17.927Z",
        "latitude": "15.8836912",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T14:57:23.192Z",
        "latitude": "15.883704",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-04T14:57:30.229Z",
        "latitude": "15.883696",
        "longitude": "75.7042112"
    },
    {
        "timestamp": "2024-06-04T14:57:35.520Z",
        "latitude": "15.8836272",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-04T14:57:40.558Z",
        "latitude": "15.8836608",
        "longitude": "75.703744"
    },
    {
        "timestamp": "2024-06-04T14:57:45.808Z",
        "latitude": "15.8836256",
        "longitude": "75.7032768"
    },
    {
        "timestamp": "2024-06-04T14:57:51.830Z",
        "latitude": "15.8837568",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-04T14:57:57.118Z",
        "latitude": "15.8839152",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T14:58:04.103Z",
        "latitude": "15.8838336",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-04T14:58:09.390Z",
        "latitude": "15.8839504",
        "longitude": "75.7027456"
    },
    {
        "timestamp": "2024-06-04T14:58:14.425Z",
        "latitude": "15.88396",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-04T14:58:19.707Z",
        "latitude": "15.8838192",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T14:58:25.708Z",
        "latitude": "15.8839232",
        "longitude": "75.7026496"
    },
    {
        "timestamp": "2024-06-04T14:58:30.984Z",
        "latitude": "15.8841424",
        "longitude": "75.702848"
    },
    {
        "timestamp": "2024-06-04T14:58:36.997Z",
        "latitude": "15.88396",
        "longitude": "75.7043712"
    },
    {
        "timestamp": "2024-06-04T14:58:42.338Z",
        "latitude": "15.8840016",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-04T14:58:48.280Z",
        "latitude": "15.8839232",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-04T14:58:54.586Z",
        "latitude": "15.8839776",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T14:58:59.590Z",
        "latitude": "15.8839392",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-04T14:59:04.901Z",
        "latitude": "15.8846256",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T14:59:10.897Z",
        "latitude": "15.8840304",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T14:59:16.180Z",
        "latitude": "15.8840272",
        "longitude": "75.7028736"
    },
    {
        "timestamp": "2024-06-04T14:59:23.195Z",
        "latitude": "15.8849648",
        "longitude": "75.7029312"
    },
    {
        "timestamp": "2024-06-04T14:59:28.492Z",
        "latitude": "15.8839728",
        "longitude": "75.702752"
    },
    {
        "timestamp": "2024-06-04T14:59:34.486Z",
        "latitude": "15.8847888",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-04T14:59:44.003Z",
        "latitude": "15.883688",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-04T14:59:45.222Z",
        "latitude": "15.8837712",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-04T14:59:46.441Z",
        "latitude": "15.8836944",
        "longitude": "75.7027456"
    },
    {
        "timestamp": "2024-06-04T14:59:50.628Z",
        "latitude": "15.8838656",
        "longitude": "75.7027072"
    },
    {
        "timestamp": "2024-06-04T14:59:51.858Z",
        "latitude": "15.8839776",
        "longitude": "75.7027584"
    },
    {
        "timestamp": "2024-06-04T14:59:57.344Z",
        "latitude": "15.8836384",
        "longitude": "75.7027648"
    },
    {
        "timestamp": "2024-06-04T14:59:58.561Z",
        "latitude": "15.8841968",
        "longitude": "75.7033088"
    },
    {
        "timestamp": "2024-06-04T15:00:12.380Z",
        "latitude": "15.8840384",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-04T15:00:14.458Z",
        "latitude": "15.8841648",
        "longitude": "75.7034368"
    },
    {
        "timestamp": "2024-06-04T15:00:24.710Z",
        "latitude": "15.8841552",
        "longitude": "75.7033472"
    },
    {
        "timestamp": "2024-06-04T15:00:26.825Z",
        "latitude": "15.8840112",
        "longitude": "75.703296"
    },
    {
        "timestamp": "2024-06-04T15:00:36.548Z",
        "latitude": "15.8839184",
        "longitude": "75.7032768"
    },
    {
        "timestamp": "2024-06-04T15:00:38.724Z",
        "latitude": "15.8841952",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-04T15:00:47.854Z",
        "latitude": "15.8843152",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T15:00:50.010Z",
        "latitude": "15.8840976",
        "longitude": "75.7033472"
    },
    {
        "timestamp": "2024-06-04T15:00:59.169Z",
        "latitude": "15.8843264",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-04T15:01:01.321Z",
        "latitude": "15.883808",
        "longitude": "75.7032768"
    },
    {
        "timestamp": "2024-06-04T15:01:10.492Z",
        "latitude": "15.8842416",
        "longitude": "75.7033216"
    },
    {
        "timestamp": "2024-06-04T15:01:12.621Z",
        "latitude": "15.88428",
        "longitude": "75.7033536"
    },
    {
        "timestamp": "2024-06-04T15:01:21.757Z",
        "latitude": "15.8841248",
        "longitude": "75.7032704"
    },
    {
        "timestamp": "2024-06-04T15:01:23.916Z",
        "latitude": "15.8847104",
        "longitude": "75.7032576"
    },
    {
        "timestamp": "2024-06-04T15:01:33.057Z",
        "latitude": "15.8841232",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-04T15:01:35.219Z",
        "latitude": "15.8841152",
        "longitude": "75.7030848"
    },
    {
        "timestamp": "2024-06-04T15:01:44.348Z",
        "latitude": "15.8841184",
        "longitude": "75.7030784"
    },
    {
        "timestamp": "2024-06-04T15:01:46.523Z",
        "latitude": "15.8844384",
        "longitude": "75.7028096"
    },
    {
        "timestamp": "2024-06-04T15:01:55.666Z",
        "latitude": "15.8841872",
        "longitude": "75.7028288"
    },
    {
        "timestamp": "2024-06-04T15:01:57.817Z",
        "latitude": "15.8839008",
        "longitude": "75.7028992"
    },
    {
        "timestamp": "2024-06-04T15:02:06.958Z",
        "latitude": "15.88456",
        "longitude": "75.7028736"
    },
    {
        "timestamp": "2024-06-04T15:02:09.126Z",
        "latitude": "15.8847296",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T15:02:18.276Z",
        "latitude": "15.8841984",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T15:02:21.414Z",
        "latitude": "15.8842608",
        "longitude": "75.7027648"
    },
    {
        "timestamp": "2024-06-04T15:02:29.557Z",
        "latitude": "15.8844352",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-04T15:02:31.689Z",
        "latitude": "15.8836496",
        "longitude": "75.7021504"
    },
    {
        "timestamp": "2024-06-04T15:02:40.852Z",
        "latitude": "15.8841584",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T15:02:43.021Z",
        "latitude": "15.8836672",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-04T15:02:52.164Z",
        "latitude": "15.8838944",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T15:02:54.287Z",
        "latitude": "15.8837936",
        "longitude": "75.702752"
    },
    {
        "timestamp": "2024-06-04T15:03:03.463Z",
        "latitude": "15.8840016",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-04T15:03:05.601Z",
        "latitude": "15.8835808",
        "longitude": "75.703456"
    },
    {
        "timestamp": "2024-06-04T15:03:14.745Z",
        "latitude": "15.8842256",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T15:03:16.898Z",
        "latitude": "15.8838288",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T15:03:26.035Z",
        "latitude": "15.8839728",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T15:03:28.173Z",
        "latitude": "15.8838256",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T15:03:38.347Z",
        "latitude": "15.8839008",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-04T15:03:39.558Z",
        "latitude": "15.8838416",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T15:03:48.627Z",
        "latitude": "15.8839152",
        "longitude": "75.7030848"
    },
    {
        "timestamp": "2024-06-04T15:03:50.853Z",
        "latitude": "15.8843024",
        "longitude": "75.70256"
    },
    {
        "timestamp": "2024-06-04T15:03:59.943Z",
        "latitude": "15.8842448",
        "longitude": "75.7031488"
    },
    {
        "timestamp": "2024-06-04T15:04:02.179Z",
        "latitude": "15.884",
        "longitude": "75.7028864"
    },
    {
        "timestamp": "2024-06-04T15:04:11.238Z",
        "latitude": "15.8840688",
        "longitude": "75.702784"
    },
    {
        "timestamp": "2024-06-04T15:04:14.444Z",
        "latitude": "15.884032",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-04T15:04:22.537Z",
        "latitude": "15.8838976",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-04T15:04:24.782Z",
        "latitude": "15.883832",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-04T15:04:33.839Z",
        "latitude": "15.8839424",
        "longitude": "75.7031488"
    },
    {
        "timestamp": "2024-06-04T15:04:36.049Z",
        "latitude": "15.8839728",
        "longitude": "75.7030784"
    },
    {
        "timestamp": "2024-06-04T15:04:45.135Z",
        "latitude": "15.884024",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-04T15:04:47.365Z",
        "latitude": "15.8840128",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-04T15:04:56.436Z",
        "latitude": "15.8843296",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-04T15:04:58.654Z",
        "latitude": "15.8839488",
        "longitude": "75.7031488"
    },
    {
        "timestamp": "2024-06-04T15:05:07.725Z",
        "latitude": "15.8840432",
        "longitude": "75.7032448"
    },
    {
        "timestamp": "2024-06-04T15:05:09.935Z",
        "latitude": "15.8840224",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-04T15:05:19.059Z",
        "latitude": "15.8839648",
        "longitude": "75.7032256"
    },
    {
        "timestamp": "2024-06-04T15:05:21.241Z",
        "latitude": "15.8846736",
        "longitude": "75.7041408"
    },
    {
        "timestamp": "2024-06-04T15:05:31.373Z",
        "latitude": "15.8839344",
        "longitude": "75.7036032"
    },
    {
        "timestamp": "2024-06-04T15:05:32.588Z",
        "latitude": "15.8839552",
        "longitude": "75.7035904"
    },
    {
        "timestamp": "2024-06-04T15:05:41.716Z",
        "latitude": "15.8839984",
        "longitude": "75.7036416"
    },
    {
        "timestamp": "2024-06-04T15:05:43.851Z",
        "latitude": "15.8840016",
        "longitude": "75.7034368"
    },
    {
        "timestamp": "2024-06-04T15:05:52.929Z",
        "latitude": "15.884312",
        "longitude": "75.7033088"
    },
    {
        "timestamp": "2024-06-04T15:05:55.135Z",
        "latitude": "15.884",
        "longitude": "75.703456"
    },
    {
        "timestamp": "2024-06-04T15:06:05.235Z",
        "latitude": "15.8840848",
        "longitude": "75.7034368"
    },
    {
        "timestamp": "2024-06-04T15:06:07.444Z",
        "latitude": "15.884424",
        "longitude": "75.7035072"
    },
    {
        "timestamp": "2024-06-04T15:06:15.539Z",
        "latitude": "15.8861328",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-04T15:06:17.727Z",
        "latitude": "15.88812",
        "longitude": "75.7027136"
    },
    {
        "timestamp": "2024-06-04T15:06:26.838Z",
        "latitude": "15.8836672",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T15:06:29.044Z",
        "latitude": "15.8834512",
        "longitude": "75.7036224"
    },
    {
        "timestamp": "2024-06-04T15:06:38.138Z",
        "latitude": "15.8837568",
        "longitude": "75.702784"
    },
    {
        "timestamp": "2024-06-04T15:06:41.327Z",
        "latitude": "15.8837408",
        "longitude": "75.70256"
    },
    {
        "timestamp": "2024-06-04T15:06:49.414Z",
        "latitude": "15.8839392",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T15:06:51.645Z",
        "latitude": "15.8840832",
        "longitude": "75.70336"
    },
    {
        "timestamp": "2024-06-04T15:07:00.719Z",
        "latitude": "15.8839456",
        "longitude": "75.703424"
    },
    {
        "timestamp": "2024-06-04T15:07:02.942Z",
        "latitude": "15.8843344",
        "longitude": "75.7028608"
    },
    {
        "timestamp": "2024-06-04T15:07:12.012Z",
        "latitude": "15.8838208",
        "longitude": "75.7022144"
    },
    {
        "timestamp": "2024-06-04T15:07:14.231Z",
        "latitude": "15.8840864",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T15:07:24.330Z",
        "latitude": "15.88392",
        "longitude": "75.7030272"
    },
    {
        "timestamp": "2024-06-04T15:07:25.538Z",
        "latitude": "15.8842064",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-04T15:07:34.611Z",
        "latitude": "15.8839728",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T15:07:36.814Z",
        "latitude": "15.8840576",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T15:07:45.927Z",
        "latitude": "15.8839504",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-04T15:07:48.141Z",
        "latitude": "15.8842368",
        "longitude": "75.7030848"
    },
    {
        "timestamp": "2024-06-04T15:07:58.208Z",
        "latitude": "15.8844112",
        "longitude": "75.7032192"
    },
    {
        "timestamp": "2024-06-04T15:08:00.416Z",
        "latitude": "15.8841264",
        "longitude": "75.7031104"
    },
    {
        "timestamp": "2024-06-04T15:08:08.515Z",
        "latitude": "15.88376",
        "longitude": "75.7033152"
    },
    {
        "timestamp": "2024-06-04T15:08:10.726Z",
        "latitude": "15.8846064",
        "longitude": "75.703552"
    },
    {
        "timestamp": "2024-06-04T15:08:19.819Z",
        "latitude": "15.8839056",
        "longitude": "75.7030848"
    },
    {
        "timestamp": "2024-06-04T15:08:22.005Z",
        "latitude": "15.883864",
        "longitude": "75.7034304"
    },
    {
        "timestamp": "2024-06-04T15:08:31.112Z",
        "latitude": "15.8836384",
        "longitude": "75.7026048"
    },
    {
        "timestamp": "2024-06-04T15:08:34.347Z",
        "latitude": "15.8839792",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T15:08:42.391Z",
        "latitude": "15.8833168",
        "longitude": "75.7027904"
    },
    {
        "timestamp": "2024-06-04T15:08:44.603Z",
        "latitude": "15.8839248",
        "longitude": "75.7031232"
    },
    {
        "timestamp": "2024-06-04T15:08:53.716Z",
        "latitude": "15.8841024",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T15:08:55.929Z",
        "latitude": "15.8840752",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-04T15:09:05.026Z",
        "latitude": "15.8838352",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T15:09:07.222Z",
        "latitude": "15.8840064",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-04T15:09:17.289Z",
        "latitude": "15.8839456",
        "longitude": "75.7030976"
    },
    {
        "timestamp": "2024-06-04T15:09:18.516Z",
        "latitude": "15.8843056",
        "longitude": "75.7033472"
    },
    {
        "timestamp": "2024-06-04T15:09:27.606Z",
        "latitude": "15.8838416",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-04T15:09:29.826Z",
        "latitude": "15.8838192",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-04T15:09:38.893Z",
        "latitude": "15.8833712",
        "longitude": "75.7025664"
    },
    {
        "timestamp": "2024-06-04T15:09:41.128Z",
        "latitude": "15.8833696",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T15:09:51.224Z",
        "latitude": "15.8840048",
        "longitude": "75.7028864"
    },
    {
        "timestamp": "2024-06-04T15:09:53.432Z",
        "latitude": "15.8838528",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-04T15:10:01.497Z",
        "latitude": "15.8840544",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T15:10:03.702Z",
        "latitude": "15.8843472",
        "longitude": "75.7021696"
    },
    {
        "timestamp": "2024-06-04T15:10:12.791Z",
        "latitude": "15.8857104",
        "longitude": "75.7034816"
    },
    {
        "timestamp": "2024-06-04T15:10:15.029Z",
        "latitude": "15.883792",
        "longitude": "75.7031552"
    },
    {
        "timestamp": "2024-06-04T15:10:25.096Z",
        "latitude": "15.8836272",
        "longitude": "75.7035648"
    },
    {
        "timestamp": "2024-06-04T15:10:27.307Z",
        "latitude": "15.8828464",
        "longitude": "75.7042752"
    },
    {
        "timestamp": "2024-06-04T15:10:35.388Z",
        "latitude": "15.8838976",
        "longitude": "75.7033984"
    },
    {
        "timestamp": "2024-06-04T15:10:37.609Z",
        "latitude": "15.8838928",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T15:10:46.697Z",
        "latitude": "15.8841072",
        "longitude": "75.7031168"
    },
    {
        "timestamp": "2024-06-04T15:10:47.974Z",
        "latitude": "15.8844992",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-04T15:10:49.227Z",
        "latitude": "15.8841904",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-04T15:11:04.271Z",
        "latitude": "15.884064",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-04T15:11:06.620Z",
        "latitude": "15.8838848",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-04T15:11:20.122Z",
        "latitude": "15.8840304",
        "longitude": "75.7027904"
    },
    {
        "timestamp": "2024-06-04T15:11:21.338Z",
        "latitude": "15.8839904",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-04T15:11:25.416Z",
        "latitude": "15.8837168",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-04T15:11:33.755Z",
        "latitude": "15.8837936",
        "longitude": "75.7028032"
    },
    {
        "timestamp": "2024-06-04T15:11:37.316Z",
        "latitude": "15.8835808",
        "longitude": "75.7028608"
    },
    {
        "timestamp": "2024-06-04T15:11:45.653Z",
        "latitude": "15.8837168",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T15:11:48.601Z",
        "latitude": "15.8840672",
        "longitude": "75.7026752"
    },
    {
        "timestamp": "2024-06-04T15:11:56.938Z",
        "latitude": "15.8838816",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-04T15:11:59.896Z",
        "latitude": "15.8844304",
        "longitude": "75.7025472"
    },
    {
        "timestamp": "2024-06-04T15:12:08.240Z",
        "latitude": "15.8838144",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T15:12:11.199Z",
        "latitude": "15.883704",
        "longitude": "75.7027648"
    },
    {
        "timestamp": "2024-06-04T15:12:19.547Z",
        "latitude": "15.883344",
        "longitude": "75.7023616"
    },
    {
        "timestamp": "2024-06-04T15:12:22.512Z",
        "latitude": "15.8838688",
        "longitude": "75.7026368"
    },
    {
        "timestamp": "2024-06-04T15:12:30.851Z",
        "latitude": "15.8839648",
        "longitude": "75.702784"
    },
    {
        "timestamp": "2024-06-04T15:12:33.818Z",
        "latitude": "15.8839264",
        "longitude": "75.7027648"
    },
    {
        "timestamp": "2024-06-04T15:12:42.157Z",
        "latitude": "15.8838544",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-04T15:12:45.148Z",
        "latitude": "15.8837552",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-04T15:12:53.419Z",
        "latitude": "15.8836672",
        "longitude": "75.702912"
    },
    {
        "timestamp": "2024-06-04T15:12:56.403Z",
        "latitude": "15.8835872",
        "longitude": "75.7025472"
    },
    {
        "timestamp": "2024-06-04T15:13:04.745Z",
        "latitude": "15.8838432",
        "longitude": "75.7028352"
    },
    {
        "timestamp": "2024-06-04T15:13:07.717Z",
        "latitude": "15.8838304",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T15:13:16.028Z",
        "latitude": "15.8837744",
        "longitude": "75.7025984"
    },
    {
        "timestamp": "2024-06-04T15:13:19.003Z",
        "latitude": "15.8838128",
        "longitude": "75.70288"
    },
    {
        "timestamp": "2024-06-04T15:13:27.321Z",
        "latitude": "15.8838976",
        "longitude": "75.7028032"
    },
    {
        "timestamp": "2024-06-04T15:13:30.300Z",
        "latitude": "15.8838592",
        "longitude": "75.702624"
    },
    {
        "timestamp": "2024-06-04T15:13:38.624Z",
        "latitude": "15.8837792",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-04T15:13:41.595Z",
        "latitude": "15.8836272",
        "longitude": "75.7028736"
    },
    {
        "timestamp": "2024-06-04T15:13:49.927Z",
        "latitude": "15.883696",
        "longitude": "75.7028096"
    },
    {
        "timestamp": "2024-06-04T15:13:52.883Z",
        "latitude": "15.8830352",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-04T15:14:01.219Z",
        "latitude": "15.8839376",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T15:14:04.185Z",
        "latitude": "15.8838288",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T15:14:12.516Z",
        "latitude": "15.8837392",
        "longitude": "75.7030848"
    },
    {
        "timestamp": "2024-06-04T15:14:15.485Z",
        "latitude": "15.8839952",
        "longitude": "75.703424"
    },
    {
        "timestamp": "2024-06-04T15:14:23.816Z",
        "latitude": "15.8842128",
        "longitude": "75.703552"
    },
    {
        "timestamp": "2024-06-04T15:14:26.777Z",
        "latitude": "15.8843008",
        "longitude": "75.7032064"
    },
    {
        "timestamp": "2024-06-04T15:14:35.117Z",
        "latitude": "15.8838336",
        "longitude": "75.7027968"
    },
    {
        "timestamp": "2024-06-04T15:14:38.084Z",
        "latitude": "15.8835632",
        "longitude": "75.7023488"
    },
    {
        "timestamp": "2024-06-04T15:14:46.404Z",
        "latitude": "15.8840304",
        "longitude": "75.702912"
    },
    {
        "timestamp": "2024-06-04T15:14:49.384Z",
        "latitude": "15.8839632",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T15:14:57.790Z",
        "latitude": "15.88408",
        "longitude": "75.7028096"
    },
    {
        "timestamp": "2024-06-04T15:15:00.675Z",
        "latitude": "15.8839968",
        "longitude": "75.7026752"
    },
    {
        "timestamp": "2024-06-04T15:15:09.011Z",
        "latitude": "15.883808",
        "longitude": "75.7027968"
    },
    {
        "timestamp": "2024-06-04T15:15:11.980Z",
        "latitude": "15.8841216",
        "longitude": "75.7027264"
    },
    {
        "timestamp": "2024-06-04T15:15:20.335Z",
        "latitude": "15.8839184",
        "longitude": "75.7036416"
    },
    {
        "timestamp": "2024-06-04T15:15:23.270Z",
        "latitude": "15.8839472",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-04T15:15:32.597Z",
        "latitude": "15.8843232",
        "longitude": "75.7030976"
    },
    {
        "timestamp": "2024-06-04T15:15:34.576Z",
        "latitude": "15.8842224",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T15:15:42.923Z",
        "latitude": "429.4967295",
        "longitude": "429.4967295"
    },
    {
        "timestamp": "2024-06-04T15:15:46.889Z",
        "latitude": "8.5214464",
        "longitude": "6.9010498"
    },
    {
        "timestamp": "2024-06-04T15:15:54.213Z",
        "latitude": "15.8837808",
        "longitude": "75.7028608"
    },
    {
        "timestamp": "2024-06-04T15:15:58.157Z",
        "latitude": "15.884144",
        "longitude": "75.7031232"
    },
    {
        "timestamp": "2024-06-04T15:16:06.495Z",
        "latitude": "15.8840752",
        "longitude": "75.7022016"
    },
    {
        "timestamp": "2024-06-04T15:16:08.454Z",
        "latitude": "15.8838528",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-04T15:16:16.795Z",
        "latitude": "15.8840096",
        "longitude": "75.7027968"
    },
    {
        "timestamp": "2024-06-04T15:16:19.785Z",
        "latitude": "15.8837568",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T15:16:28.120Z",
        "latitude": "15.8838448",
        "longitude": "75.7028288"
    },
    {
        "timestamp": "2024-06-04T15:16:31.056Z",
        "latitude": "15.884024",
        "longitude": "75.702848"
    },
    {
        "timestamp": "2024-06-04T15:16:39.398Z",
        "latitude": "15.8839216",
        "longitude": "75.7028032"
    },
    {
        "timestamp": "2024-06-04T15:16:42.370Z",
        "latitude": "15.8841088",
        "longitude": "75.7028352"
    },
    {
        "timestamp": "2024-06-04T15:16:50.694Z",
        "latitude": "15.8835056",
        "longitude": "75.7028032"
    },
    {
        "timestamp": "2024-06-04T15:16:53.655Z",
        "latitude": "15.8829584",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T15:17:01.986Z",
        "latitude": "15.8838064",
        "longitude": "75.702272"
    },
    {
        "timestamp": "2024-06-04T15:17:04.956Z",
        "latitude": "15.88324",
        "longitude": "75.7032448"
    },
    {
        "timestamp": "2024-06-04T15:17:13.301Z",
        "latitude": "15.883104",
        "longitude": "75.7031232"
    },
    {
        "timestamp": "2024-06-04T15:17:16.258Z",
        "latitude": "15.8836544",
        "longitude": "75.7029184"
    },
    {
        "timestamp": "2024-06-04T15:17:25.577Z",
        "latitude": "15.8830432",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-04T15:17:27.532Z",
        "latitude": "15.8836256",
        "longitude": "75.7031232"
    },
    {
        "timestamp": "2024-06-04T15:17:35.876Z",
        "latitude": "15.8837184",
        "longitude": "75.7030976"
    },
    {
        "timestamp": "2024-06-04T15:17:38.871Z",
        "latitude": "15.8839392",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-04T15:17:47.175Z",
        "latitude": "15.8837168",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T15:17:51.158Z",
        "latitude": "15.8847328",
        "longitude": "75.7040512"
    },
    {
        "timestamp": "2024-06-04T15:17:59.502Z",
        "latitude": "15.883984",
        "longitude": "75.7030784"
    },
    {
        "timestamp": "2024-06-04T15:18:01.459Z",
        "latitude": "15.88452",
        "longitude": "75.7034688"
    },
    {
        "timestamp": "2024-06-04T15:18:09.792Z",
        "latitude": "15.8837968",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-04T15:18:12.754Z",
        "latitude": "15.8839616",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T15:18:21.075Z",
        "latitude": "15.8838592",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T15:18:24.040Z",
        "latitude": "15.8838592",
        "longitude": "75.7027648"
    },
    {
        "timestamp": "2024-06-04T15:18:32.380Z",
        "latitude": "15.8838528",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T15:18:35.344Z",
        "latitude": "15.8839552",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-04T15:18:43.716Z",
        "latitude": "15.883576",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-04T15:18:46.643Z",
        "latitude": "15.8834192",
        "longitude": "75.7027456"
    },
    {
        "timestamp": "2024-06-04T15:18:54.982Z",
        "latitude": "15.8839168",
        "longitude": "75.7044736"
    },
    {
        "timestamp": "2024-06-04T15:18:57.922Z",
        "latitude": "15.8840128",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-04T15:19:06.263Z",
        "latitude": "15.8839568",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-04T15:19:10.236Z",
        "latitude": "15.8837008",
        "longitude": "75.7028672"
    },
    {
        "timestamp": "2024-06-04T15:19:18.573Z",
        "latitude": "15.8838592",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-04T15:19:20.547Z",
        "latitude": "15.8837056",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-04T15:19:28.872Z",
        "latitude": "15.8838368",
        "longitude": "75.7026624"
    },
    {
        "timestamp": "2024-06-04T15:19:31.839Z",
        "latitude": "15.8837568",
        "longitude": "75.7025984"
    },
    {
        "timestamp": "2024-06-04T15:19:40.180Z",
        "latitude": "15.8832224",
        "longitude": "75.7031168"
    },
    {
        "timestamp": "2024-06-04T15:19:44.143Z",
        "latitude": "15.88388",
        "longitude": "75.7024832"
    },
    {
        "timestamp": "2024-06-04T15:19:52.564Z",
        "latitude": "15.88396",
        "longitude": "75.702464"
    },
    {
        "timestamp": "2024-06-04T15:19:55.437Z",
        "latitude": "15.8839792",
        "longitude": "75.7025856"
    },
    {
        "timestamp": "2024-06-04T15:20:02.778Z",
        "latitude": "15.8838848",
        "longitude": "75.7024768"
    },
    {
        "timestamp": "2024-06-04T15:20:05.743Z",
        "latitude": "15.883776",
        "longitude": "75.70272"
    },
    {
        "timestamp": "2024-06-04T15:20:14.080Z",
        "latitude": "15.8841968",
        "longitude": "75.7033088"
    },
    {
        "timestamp": "2024-06-04T15:22:09.972Z",
        "latitude": "15.883928",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T15:22:25.440Z",
        "latitude": "15.884584",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T15:30:33.159Z",
        "latitude": "15.8841536",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T15:33:26.847Z",
        "latitude": "15.8840576",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T15:33:28.105Z",
        "latitude": "15.8840288",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-04T15:33:38.139Z",
        "latitude": "15.88404",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T15:33:49.801Z",
        "latitude": "15.8841568",
        "longitude": "75.70336"
    },
    {
        "timestamp": "2024-06-04T15:33:51.014Z",
        "latitude": "15.8864432",
        "longitude": "75.7033408"
    },
    {
        "timestamp": "2024-06-04T15:33:56.092Z",
        "latitude": "15.8873552",
        "longitude": "75.703424"
    },
    {
        "timestamp": "2024-06-04T15:34:03.414Z",
        "latitude": "15.8834528",
        "longitude": "75.7031488"
    },
    {
        "timestamp": "2024-06-04T15:34:06.979Z",
        "latitude": "15.8838304",
        "longitude": "75.7031808"
    },
    {
        "timestamp": "2024-06-04T15:34:15.319Z",
        "latitude": "15.8839584",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T15:34:18.267Z",
        "latitude": "15.8839008",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T15:34:26.615Z",
        "latitude": "15.8838464",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T15:34:30.574Z",
        "latitude": "15.8839264",
        "longitude": "75.7032832"
    },
    {
        "timestamp": "2024-06-04T15:34:36.885Z",
        "latitude": "15.88408",
        "longitude": "75.7029568"
    },
    {
        "timestamp": "2024-06-04T15:34:41.909Z",
        "latitude": "15.8835168",
        "longitude": "75.7034944"
    },
    {
        "timestamp": "2024-06-04T15:34:49.189Z",
        "latitude": "15.8845728",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-04T15:34:53.169Z",
        "latitude": "15.883984",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-04T15:35:00.493Z",
        "latitude": "15.8839824",
        "longitude": "75.7032704"
    },
    {
        "timestamp": "2024-06-04T15:35:10.792Z",
        "latitude": "15.883816",
        "longitude": "75.7033472"
    },
    {
        "timestamp": "2024-06-04T15:35:15.817Z",
        "latitude": "15.8837568",
        "longitude": "75.7033856"
    },
    {
        "timestamp": "2024-06-04T15:35:23.091Z",
        "latitude": "15.8838096",
        "longitude": "75.7033536"
    },
    {
        "timestamp": "2024-06-04T15:35:27.089Z",
        "latitude": "15.8833984",
        "longitude": "75.7034304"
    },
    {
        "timestamp": "2024-06-04T15:35:34.368Z",
        "latitude": "15.8840224",
        "longitude": "75.7033472"
    },
    {
        "timestamp": "2024-06-04T15:35:37.411Z",
        "latitude": "15.883872",
        "longitude": "75.7033152"
    },
    {
        "timestamp": "2024-06-04T15:35:48.673Z",
        "latitude": "15.8837952",
        "longitude": "75.7032832"
    },
    {
        "timestamp": "2024-06-04T15:35:57.011Z",
        "latitude": "15.8840576",
        "longitude": "75.703456"
    },
    {
        "timestamp": "2024-06-04T15:36:00.988Z",
        "latitude": "15.8843648",
        "longitude": "75.703392"
    },
    {
        "timestamp": "2024-06-04T15:36:12.291Z",
        "latitude": "15.883792",
        "longitude": "75.7032448"
    },
    {
        "timestamp": "2024-06-04T15:36:19.570Z",
        "latitude": "15.8836336",
        "longitude": "75.7032576"
    },
    {
        "timestamp": "2024-06-04T15:36:30.878Z",
        "latitude": "15.8840192",
        "longitude": "75.7033664"
    },
    {
        "timestamp": "2024-06-04T15:36:34.891Z",
        "latitude": "15.8837744",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T15:36:42.192Z",
        "latitude": "15.8838144",
        "longitude": "75.7027264"
    },
    {
        "timestamp": "2024-06-04T15:36:46.161Z",
        "latitude": "15.8837712",
        "longitude": "75.7034176"
    },
    {
        "timestamp": "2024-06-04T15:36:53.505Z",
        "latitude": "15.8838688",
        "longitude": "75.7037504"
    },
    {
        "timestamp": "2024-06-04T15:36:56.473Z",
        "latitude": "15.8841408",
        "longitude": "75.7036608"
    },
    {
        "timestamp": "2024-06-04T15:37:08.773Z",
        "latitude": "15.8836496",
        "longitude": "75.7030784"
    },
    {
        "timestamp": "2024-06-04T15:37:16.068Z",
        "latitude": "15.8839936",
        "longitude": "75.702912"
    },
    {
        "timestamp": "2024-06-04T15:37:21.064Z",
        "latitude": "15.8839088",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T15:37:27.376Z",
        "latitude": "15.8838128",
        "longitude": "75.7030784"
    },
    {
        "timestamp": "2024-06-04T15:37:31.371Z",
        "latitude": "15.8834928",
        "longitude": "75.7032832"
    },
    {
        "timestamp": "2024-06-04T15:37:37.667Z",
        "latitude": "15.8829952",
        "longitude": "75.7036544"
    },
    {
        "timestamp": "2024-06-04T15:37:43.673Z",
        "latitude": "15.8837456",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T15:37:49.951Z",
        "latitude": "15.883648",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T15:40:52.044Z",
        "latitude": "15.8842944",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-04T15:44:47.168Z",
        "latitude": "15.884336",
        "longitude": "75.7037248"
    },
    {
        "timestamp": "2024-06-04T15:44:48.396Z",
        "latitude": "15.884328",
        "longitude": "75.7027264"
    },
    {
        "timestamp": "2024-06-04T15:44:57.462Z",
        "latitude": "15.8840512",
        "longitude": "75.7032"
    },
    {
        "timestamp": "2024-06-04T15:44:59.783Z",
        "latitude": "15.8844656",
        "longitude": "75.702848"
    },
    {
        "timestamp": "2024-06-04T15:45:09.368Z",
        "latitude": "15.8839152",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-04T15:45:11.665Z",
        "latitude": "15.8842096",
        "longitude": "75.702688"
    },
    {
        "timestamp": "2024-06-04T15:45:20.652Z",
        "latitude": "15.8842816",
        "longitude": "75.7024256"
    },
    {
        "timestamp": "2024-06-04T15:45:22.976Z",
        "latitude": "15.8839616",
        "longitude": "75.7028864"
    },
    {
        "timestamp": "2024-06-04T15:45:31.940Z",
        "latitude": "15.8841088",
        "longitude": "75.7032192"
    },
    {
        "timestamp": "2024-06-04T15:45:34.255Z",
        "latitude": "15.884008",
        "longitude": "75.7033152"
    },
    {
        "timestamp": "2024-06-04T15:45:44.258Z",
        "latitude": "15.8841168",
        "longitude": "75.7020352"
    },
    {
        "timestamp": "2024-06-04T15:45:46.570Z",
        "latitude": "15.88412",
        "longitude": "75.7035904"
    },
    {
        "timestamp": "2024-06-04T15:45:54.538Z",
        "latitude": "15.883928",
        "longitude": "75.7032192"
    },
    {
        "timestamp": "2024-06-04T15:45:56.862Z",
        "latitude": "15.8839936",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T15:46:05.839Z",
        "latitude": "15.8837648",
        "longitude": "75.7032768"
    },
    {
        "timestamp": "2024-06-04T15:46:08.173Z",
        "latitude": "15.8838592",
        "longitude": "75.7031808"
    },
    {
        "timestamp": "2024-06-04T15:46:18.205Z",
        "latitude": "15.8838976",
        "longitude": "75.703264"
    },
    {
        "timestamp": "2024-06-04T15:46:20.459Z",
        "latitude": "15.8838832",
        "longitude": "75.7031488"
    },
    {
        "timestamp": "2024-06-04T15:46:28.432Z",
        "latitude": "15.883944",
        "longitude": "75.7032192"
    },
    {
        "timestamp": "2024-06-04T15:46:30.763Z",
        "latitude": "15.8838896",
        "longitude": "75.7033472"
    },
    {
        "timestamp": "2024-06-04T15:46:39.748Z",
        "latitude": "15.883056",
        "longitude": "75.7055424"
    },
    {
        "timestamp": "2024-06-04T15:46:42.061Z",
        "latitude": "15.8837792",
        "longitude": "75.7037888"
    },
    {
        "timestamp": "2024-06-04T15:46:52.044Z",
        "latitude": "15.884608",
        "longitude": "75.7037888"
    },
    {
        "timestamp": "2024-06-04T15:46:54.370Z",
        "latitude": "15.8846304",
        "longitude": "75.7034944"
    },
    {
        "timestamp": "2024-06-04T15:47:02.346Z",
        "latitude": "15.8842016",
        "longitude": "75.7031808"
    },
    {
        "timestamp": "2024-06-04T15:47:04.677Z",
        "latitude": "15.8844096",
        "longitude": "75.7032768"
    },
    {
        "timestamp": "2024-06-04T15:47:13.654Z",
        "latitude": "15.8840512",
        "longitude": "75.7030848"
    },
    {
        "timestamp": "2024-06-04T15:47:15.985Z",
        "latitude": "15.8839872",
        "longitude": "75.7034048"
    },
    {
        "timestamp": "2024-06-04T15:47:24.944Z",
        "latitude": "15.8840496",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T15:47:27.269Z",
        "latitude": "15.8839696",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-04T15:47:37.238Z",
        "latitude": "15.8836608",
        "longitude": "75.7032256"
    },
    {
        "timestamp": "2024-06-04T15:47:39.568Z",
        "latitude": "15.883864",
        "longitude": "75.7030976"
    },
    {
        "timestamp": "2024-06-04T15:47:47.536Z",
        "latitude": "15.8842816",
        "longitude": "75.7031424"
    },
    {
        "timestamp": "2024-06-04T15:47:49.862Z",
        "latitude": "15.8839696",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T15:47:58.844Z",
        "latitude": "15.8833392",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T15:48:01.162Z",
        "latitude": "15.8840032",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-04T15:48:11.139Z",
        "latitude": "15.8838976",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T15:48:13.466Z",
        "latitude": "15.883968",
        "longitude": "75.7028864"
    },
    {
        "timestamp": "2024-06-04T15:48:21.436Z",
        "latitude": "15.8839552",
        "longitude": "75.7028032"
    },
    {
        "timestamp": "2024-06-04T15:48:23.760Z",
        "latitude": "15.8841168",
        "longitude": "75.70288"
    },
    {
        "timestamp": "2024-06-04T15:48:32.727Z",
        "latitude": "15.8838752",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-04T15:48:35.055Z",
        "latitude": "15.8841328",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T15:48:45.030Z",
        "latitude": "15.8837584",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-04T15:48:47.363Z",
        "latitude": "15.8839984",
        "longitude": "75.7030336"
    },
    {
        "timestamp": "2024-06-04T15:48:55.347Z",
        "latitude": "15.8838368",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-04T15:49:16.030Z",
        "latitude": "15.883776",
        "longitude": "75.7028352"
    },
    {
        "timestamp": "2024-06-04T15:49:17.273Z",
        "latitude": "15.8837536",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-04T15:49:27.361Z",
        "latitude": "15.8837264",
        "longitude": "75.703168"
    },
    {
        "timestamp": "2024-06-04T15:49:28.667Z",
        "latitude": "15.883576",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-04T15:49:39.226Z",
        "latitude": "15.8837168",
        "longitude": "75.7030592"
    },
    {
        "timestamp": "2024-06-04T15:49:41.547Z",
        "latitude": "15.8835696",
        "longitude": "75.7031232"
    },
    {
        "timestamp": "2024-06-04T15:49:49.545Z",
        "latitude": "15.8837248",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T15:49:51.867Z",
        "latitude": "15.883632",
        "longitude": "75.7031808"
    },
    {
        "timestamp": "2024-06-04T15:50:01.833Z",
        "latitude": "15.8835568",
        "longitude": "75.7031424"
    },
    {
        "timestamp": "2024-06-04T15:50:04.127Z",
        "latitude": "15.8836944",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T15:50:13.122Z",
        "latitude": "15.8836304",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T15:50:15.441Z",
        "latitude": "15.8834448",
        "longitude": "75.7028736"
    },
    {
        "timestamp": "2024-06-04T15:50:23.429Z",
        "latitude": "15.8838064",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-04T15:50:25.737Z",
        "latitude": "15.8842288",
        "longitude": "75.7026752"
    },
    {
        "timestamp": "2024-06-04T15:50:35.718Z",
        "latitude": "15.8830592",
        "longitude": "75.7030272"
    },
    {
        "timestamp": "2024-06-04T15:50:38.044Z",
        "latitude": "15.8837184",
        "longitude": "75.7034048"
    },
    {
        "timestamp": "2024-06-04T15:50:47.033Z",
        "latitude": "15.8847888",
        "longitude": "75.7031616"
    },
    {
        "timestamp": "2024-06-04T15:50:49.338Z",
        "latitude": "15.8838848",
        "longitude": "75.7031296"
    },
    {
        "timestamp": "2024-06-04T15:50:57.336Z",
        "latitude": "15.8843856",
        "longitude": "75.7032"
    },
    {
        "timestamp": "2024-06-04T15:51:10.045Z",
        "latitude": "15.8846048",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T16:00:05.431Z",
        "latitude": "15.8840576",
        "longitude": "75.7029056"
    },
    {
        "timestamp": "2024-06-04T16:00:06.648Z",
        "latitude": "226.7098795",
        "longitude": "380.7248385"
    },
    {
        "timestamp": "2024-06-04T16:00:16.710Z",
        "latitude": "15.8840352",
        "longitude": "75.703104"
    },
    {
        "timestamp": "2024-06-04T16:00:19.037Z",
        "latitude": "15.8841072",
        "longitude": "75.7031168"
    },
    {
        "timestamp": "2024-06-04T16:00:28.599Z",
        "latitude": "15.883928",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T16:00:30.928Z",
        "latitude": "15.88404",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T16:00:39.910Z",
        "latitude": "15.8838368",
        "longitude": "75.7030208"
    },
    {
        "timestamp": "2024-06-04T16:01:33.332Z",
        "latitude": "15.884672",
        "longitude": "75.7030784"
    },
    {
        "timestamp": "2024-06-04T16:02:08.445Z",
        "latitude": "15.8837344",
        "longitude": "75.7031424"
    },
    {
        "timestamp": "2024-06-04T16:02:09.657Z",
        "latitude": "15.883648",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T16:02:19.730Z",
        "latitude": "15.8843856",
        "longitude": "75.7032"
    },
    {
        "timestamp": "2024-06-04T16:11:12.295Z",
        "latitude": "15.8847248",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T16:11:51.327Z",
        "latitude": "15.8842608",
        "longitude": "75.7029888"
    },
    {
        "timestamp": "2024-06-04T16:12:37.898Z",
        "latitude": "15.8837552",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-04T16:21:30.824Z",
        "latitude": "15.8838528",
        "longitude": "75.7033984"
    },
    {
        "timestamp": "2024-06-04T16:22:15.103Z",
        "latitude": "15.8843136",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T16:22:57.087Z",
        "latitude": "15.8835712",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-04T16:31:49.210Z",
        "latitude": "15.8854464",
        "longitude": "75.7029504"
    },
    {
        "timestamp": "2024-06-04T16:32:34.132Z",
        "latitude": "15.8841696",
        "longitude": "75.7027968"
    },
    {
        "timestamp": "2024-06-04T16:33:16.278Z",
        "latitude": "15.8838112",
        "longitude": "75.7033472"
    },
    {
        "timestamp": "2024-06-04T16:42:21.070Z",
        "latitude": "15.883904",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T16:43:07.059Z",
        "latitude": "15.8839936",
        "longitude": "75.702752"
    },
    {
        "timestamp": "2024-06-04T16:43:35.446Z",
        "latitude": "15.8836608",
        "longitude": "75.7028288"
    },
    {
        "timestamp": "2024-06-04T16:52:48.167Z",
        "latitude": "15.8847552",
        "longitude": "75.7036864"
    },
    {
        "timestamp": "2024-06-04T16:53:25.101Z",
        "latitude": "15.8839872",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-04T16:53:59.085Z",
        "latitude": "15.8838144",
        "longitude": "75.701632"
    },
    {
        "timestamp": "2024-06-04T17:03:12.018Z",
        "latitude": "15.8796432",
        "longitude": "75.7037376"
    },
    {
        "timestamp": "2024-06-04T17:03:44.127Z",
        "latitude": "15.8836944",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T17:04:23.069Z",
        "latitude": "15.8839424",
        "longitude": "75.7026816"
    },
    {
        "timestamp": "2024-06-04T17:13:30.105Z",
        "latitude": "15.8836656",
        "longitude": "75.703776"
    },
    {
        "timestamp": "2024-06-04T17:14:07.972Z",
        "latitude": "15.8839184",
        "longitude": "75.7026624"
    },
    {
        "timestamp": "2024-06-04T17:14:47.094Z",
        "latitude": "15.8836496",
        "longitude": "75.7030976"
    },
    {
        "timestamp": "2024-06-04T17:23:54.036Z",
        "latitude": "15.8835008",
        "longitude": "75.7034048"
    },
    {
        "timestamp": "2024-06-04T17:24:40.998Z",
        "latitude": "15.8835888",
        "longitude": "75.702688"
    },
    {
        "timestamp": "2024-06-04T17:25:31.265Z",
        "latitude": "15.8834016",
        "longitude": "75.702912"
    },
    {
        "timestamp": "2024-06-04T17:34:17.144Z",
        "latitude": "15.8832144",
        "longitude": "75.703456"
    },
    {
        "timestamp": "2024-06-04T17:34:58.018Z",
        "latitude": "15.8831664",
        "longitude": "75.7019584"
    },
    {
        "timestamp": "2024-06-04T17:36:24.818Z",
        "latitude": "15.8843248",
        "longitude": "75.7025984"
    },
    {
        "timestamp": "2024-06-04T17:44:59.050Z",
        "latitude": "15.883752",
        "longitude": "75.7036288"
    },
    {
        "timestamp": "2024-06-04T17:45:15.863Z",
        "latitude": "15.883856",
        "longitude": "75.7024768"
    },
    {
        "timestamp": "2024-06-04T17:46:48.038Z",
        "latitude": "15.8837216",
        "longitude": "75.7029696"
    },
    {
        "timestamp": "2024-06-04T17:55:17.533Z",
        "latitude": "15.8831744",
        "longitude": "75.7032448"
    },
    {
        "timestamp": "2024-06-04T17:55:34.897Z",
        "latitude": "15.8836864",
        "longitude": "75.702336"
    },
    {
        "timestamp": "2024-06-04T17:57:07.073Z",
        "latitude": "15.883744",
        "longitude": "75.7029952"
    },
    {
        "timestamp": "2024-06-04T18:05:44.960Z",
        "latitude": "15.8838736",
        "longitude": "75.7031808"
    },
    {
        "timestamp": "2024-06-04T18:05:58.250Z",
        "latitude": "15.8835024",
        "longitude": "75.7024"
    },
    {
        "timestamp": "2024-06-04T18:07:26.141Z",
        "latitude": "15.8838128",
        "longitude": "75.70304"
    },
    {
        "timestamp": "2024-06-04T18:16:04.005Z",
        "latitude": "15.8835632",
        "longitude": "75.7035072"
    },
    {
        "timestamp": "2024-06-04T18:16:22.133Z",
        "latitude": "15.8836944",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T18:17:53.225Z",
        "latitude": "15.8838544",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T18:26:40.039Z",
        "latitude": "15.8834496",
        "longitude": "75.7024256"
    },
    {
        "timestamp": "2024-06-04T18:27:21.116Z",
        "latitude": "15.8842336",
        "longitude": "75.7034816"
    },
    {
        "timestamp": "2024-06-04T18:27:22.361Z",
        "latitude": "15.8835632",
        "longitude": "75.7035072"
    },
    {
        "timestamp": "2024-06-04T18:28:11.389Z",
        "latitude": "15.8838608",
        "longitude": "75.7029632"
    },
    {
        "timestamp": "2024-06-04T18:37:03.425Z",
        "latitude": "15.883768",
        "longitude": "75.7026496"
    },
    {
        "timestamp": "2024-06-04T18:37:40.497Z",
        "latitude": "15.8843376",
        "longitude": "75.703584"
    },
    {
        "timestamp": "2024-06-04T18:38:34.991Z",
        "latitude": "15.8836224",
        "longitude": "75.7030656"
    },
    {
        "timestamp": "2024-06-04T18:47:35.778Z",
        "latitude": "15.8836224",
        "longitude": "75.7025344"
    },
    {
        "timestamp": "2024-06-04T18:48:04.026Z",
        "latitude": "15.8833488",
        "longitude": "75.7019904"
    },
    {
        "timestamp": "2024-06-04T18:49:10.027Z",
        "latitude": "15.8838256",
        "longitude": "75.7029248"
    },
    {
        "timestamp": "2024-06-04T18:58:08.080Z",
        "latitude": "15.8836832",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-04T18:58:21.084Z",
        "latitude": "15.8831792",
        "longitude": "75.7048256"
    },
    {
        "timestamp": "2024-06-04T18:59:27.948Z",
        "latitude": "15.8837632",
        "longitude": "75.70288"
    },
    {
        "timestamp": "2024-06-04T19:08:54.053Z",
        "latitude": "15.8836816",
        "longitude": "75.7037184"
    },
    {
        "timestamp": "2024-06-04T19:09:38.972Z",
        "latitude": "15.8837488",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-04T19:09:45.847Z",
        "latitude": "15.8837504",
        "longitude": "75.703072"
    },
    {
        "timestamp": "2024-06-04T19:19:17.842Z",
        "latitude": "15.8836112",
        "longitude": "75.7027712"
    },
    {
        "timestamp": "2024-06-04T19:20:04.769Z",
        "latitude": "15.8836624",
        "longitude": "75.7030464"
    },
    {
        "timestamp": "2024-06-04T19:21:00.971Z",
        "latitude": "15.8837408",
        "longitude": "75.7028864"
    },
    {
        "timestamp": "2024-06-04T19:21:02.222Z",
        "latitude": "15.8836832",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-04T19:21:12.281Z",
        "latitude": "15.8837488",
        "longitude": "75.7027328"
    },
    {
        "timestamp": "2024-06-04T19:29:41.589Z",
        "latitude": "15.8841728",
        "longitude": "75.702944"
    },
    {
        "timestamp": "2024-06-04T19:31:04.008Z",
        "latitude": "15.8837552",
        "longitude": "75.7030016"
    },
    {
        "timestamp": "2024-06-04T19:31:35.976Z",
        "latitude": "15.883864",
        "longitude": "75.7028224"
    },
    {
        "timestamp": "2024-06-04T19:40:13.976Z",
        "latitude": "15.8837888",
        "longitude": "75.7031872"
    },
    {
        "timestamp": "2024-06-04T19:41:23.751Z",
        "latitude": "15.8838336",
        "longitude": "75.7029824"
    },
    {
        "timestamp": "2024-06-04T19:42:17.079Z",
        "latitude": "15.8839792",
        "longitude": "75.7028416"
    },
    {
        "timestamp": "2024-06-04T19:50:32.989Z",
        "latitude": "15.884624",
        "longitude": "75.7019072"
    },
    {
        "timestamp": "2024-06-04T19:51:42.598Z",
        "latitude": "15.8837136",
        "longitude": "75.70304"
    }
]

'''
    cleaned_data = clean_and_process_json_data(json_data)
    print(cleaned_data)

    if cleaned_data:
        predicted_location = predict_next_location(cleaned_data)
        return jsonify({"predicted_location": predicted_location.tolist()})
    else:
        return jsonify({"error": "Failed to process JSON data or no valid data points found."}), 400


if __name__ == '__main__':
    app.run(debug=True)

# mqtt_client.py
# streams/mqtt_client.py

import paho.mqtt.client as mqtt
import json
import os

LIVE_FEED_FILE = "data/live_sensor_feed.json"

class MQTTSensorClient:
    def __init__(self, broker='localhost', port=1883, topic='reservoir/sensors'):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def start(self):
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT Broker: {self.broker} with result code {rc}")
        client.subscribe(self.topic)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.save_to_file(payload)
        except Exception as e:
            print(f"Failed to process MQTT message: {e}")

    def save_to_file(self, data):
        os.makedirs(os.path.dirname(LIVE_FEED_FILE), exist_ok=True)
        with open(LIVE_FEED_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def read_live_feed():
        if not os.path.exists(LIVE_FEED_FILE):
            return {}
        with open(LIVE_FEED_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)


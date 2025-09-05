from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

# List of Mininet interfaces to apply delay on
interfaces = [
    's1-eth2', 's1-eth3', 's1-eth4', 's1-eth5',
    's2-eth2', 's2-eth3', 's2-eth4',
    's3-eth1', 's3-eth2', 's3-eth3',
    's4-eth1', 's4-eth2', 's4-eth3',
    's5-eth1', 's5-eth2',
    's6-eth1', 's6-eth2'
]

@app.route('/network/delay', methods=['POST'])
def update_delay():
    try:
        data = request.get_json()
        delay_ms = int(data.get("delay_ms", 100))

        for intf in interfaces:
            print(f"→ Updating {intf} to {delay_ms}ms")
            # Safely reset and apply new delay
            subprocess.run(f"tc qdisc del dev {intf} root", shell=True)
            subprocess.run(f"tc qdisc add dev {intf} root netem delay {delay_ms}ms", shell=True)

        return jsonify({"status": "success", "delay_ms": delay_ms})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)

# Launch the MLflow server so you can see the UI at http://localhost:5000
import subprocess

# Starts MLflow's tracking server with default artifact storage
subprocess.run([
    "mlflow",
    "ui",
    "--host", "localhost",
    "--port", "5000"
])

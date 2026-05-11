from roboflow import Roboflow
rf = Roboflow(api_key="QxtGjX286t0nGqCfyl84")
project = rf.workspace("prithika-tfgu5").project("Pest Detection")
dataset = project.version(1).download("yolov8")
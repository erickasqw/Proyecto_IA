import web_server


def test_detection_has_stricter_stability_limits():
    assert web_server.DEFAULT_FACE_DETECTION["scaleFactor"] >= 1.1
    assert web_server.DEFAULT_FACE_DETECTION["minNeighbors"] >= 5
    assert web_server.DEFAULT_FACE_DETECTION["minSize"][0] >= 50
    assert web_server.MIN_EMOTION_CONFIDENCE >= 0.4


def test_low_confidence_faces_are_rejected():
    assert web_server.should_accept_prediction("Happy", 0.39) is False
    assert web_server.should_accept_prediction("Happy", 0.75) is True

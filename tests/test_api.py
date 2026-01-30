import sys
import os
from fastapi.testclient import TestClient

# --- PATH FIX: Force Python to find 'src' ---
# This ensures imports work whether you use pytest, python command, or VS Code Run button
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# --------------------------------------------

from api import app

client = TestClient(app)

def test_health_check():
    """
    Verifies the /health endpoint returns 200 OK and status: ok.
    (PDF Core Req 11)
    """
    print("Testing /health endpoint...", end=" ")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("OK")

def test_predict_no_file():
    """
    Verifies the /predict endpoint returns 400 when no file is uploaded.
    (PDF Core Req 10, Scenario 1)
    """
    print("Testing /predict (no file)...", end=" ")
    # Sending a request without the 'file' parameter
    response = client.post("/predict")
    
    # We expect 400 Bad Request
    assert response.status_code == 400
    assert response.json() == {"detail": "No file uploaded"}
    print("OK")

def test_predict_invalid_file_type():
    """
    Verifies the /predict endpoint returns 400 for non-image files.
    (PDF Core Req 10, Scenario 2)
    """
    print("Testing /predict (invalid file)...", end=" ")
    # Create a dummy text file to simulate user error
    files = {'file': ('test.txt', b'this is not an image', 'text/plain')}
    response = client.post("/predict", files=files)
    
    # We expect 400 Bad Request
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
    print("OK")

# --- EXECUTION BLOCK: Runs tests when you click Play ---
if __name__ == "__main__":
    print("\n--- Starting Manual Test Run ---\n")
    try:
        test_health_check()
        test_predict_no_file()
        test_predict_invalid_file_type()
        print("\n✅ ALL TESTS PASSED SUCCESSFULLY")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
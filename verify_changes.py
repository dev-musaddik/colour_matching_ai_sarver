import os
import sys
import requests

def test_imports():
    print("Testing imports...")
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully.")
    except ImportError as e:
        print(f"❌ Failed to import MediaPipe: {e}")
        sys.exit(1)

    try:
        import torch
        print("⚠️ Warning: Torch is still importable. Did you uninstall it?")
    except ImportError:
        print("✅ Torch is correctly removed (not importable).")

def test_model_download():
    print("\nTesting model download logic...")
    try:
        from training_service import get_embedder, MP_MODEL_PATH
        
        if os.path.exists(MP_MODEL_PATH):
            print(f"ℹ️ Model file {MP_MODEL_PATH} already exists.")
            os.remove(MP_MODEL_PATH) # Force re-download to test
            print("   Deleted to test download...")
            
        embedder = get_embedder()
        
        if os.path.exists(MP_MODEL_PATH):
            print(f"✅ Model downloaded successfully to {MP_MODEL_PATH}")
            size = os.path.getsize(MP_MODEL_PATH)
            print(f"   Size: {size / 1024 / 1024:.2f} MB")
        else:
            print("❌ Model failed to download.")
            
    except Exception as e:
        print(f"❌ Error during model test: {e}")

if __name__ == "__main__":
    test_imports()
    test_model_download()

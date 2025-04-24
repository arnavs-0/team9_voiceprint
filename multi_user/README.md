# Multi-User Voice Authentication System

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd team9_voiceprint/multi_user
   ```

2. Install dependencies:
   ```
   pip install torch speechbrain sounddevice scipy flask pydub
   ```

3. Download SpeechBrain pre-trained models (happens automatically on first run)

## Usage

### Web Interface

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open browser and navigate to `http://127.0.0.1:5000`

### Command Line Interface

1. Run the model directly:
   ```
   python model.py
   ```

2. Follow the prompts to:
   - Enroll new users
   - Verify existing users

## File Structure

- `model.py`: Core voice recognition functionality
- `app.py`: Flask web application
- `templates/`: HTML templates for web interface
- `*.wav`: (When existing) Voice enrollment samples
- `speaker_embeddings.pt`: (When existing) Cached embeddings database
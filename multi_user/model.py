import os
import torch
from speechbrain import inference
import sounddevice as sd
import scipy.io.wavfile as wav
import time

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

verification = inference.SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts=device
)

SAMPLE_RATE = 16000
DURATION = 3
THRESHOLD = 0.35
SPEAKER_EMBEDDINGS_FILE = "speaker_embeddings.pt"
MAX_DB_SIZE = 4000 

if os.path.exists(SPEAKER_EMBEDDINGS_FILE):
    speaker_db = torch.load(SPEAKER_EMBEDDINGS_FILE)
else:
    speaker_db = {}

def monitor_db_size():
    if os.path.exists(SPEAKER_EMBEDDINGS_FILE):
        size = os.path.getsize(SPEAKER_EMBEDDINGS_FILE)
        print(f"Embeddings file size: {size} bytes")

def maybe_evict_oldest_speaker():
    if not os.path.exists(SPEAKER_EMBEDDINGS_FILE):
        return
    size = os.path.getsize(SPEAKER_EMBEDDINGS_FILE)
    if size > MAX_DB_SIZE:
        oldest_user = None
        oldest_time = float('inf')
        for user_id, data in speaker_db.items():
            if data["timestamp"] < oldest_time:
                oldest_time = data["timestamp"]
                oldest_user = user_id
        if oldest_user:
            del speaker_db[oldest_user]
            torch.save(speaker_db, SPEAKER_EMBEDDINGS_FILE)
            print(f"Evicted oldest user: {oldest_user}, DB size now {os.path.getsize(SPEAKER_EMBEDDINGS_FILE)} bytes")

def get_embedding(file_path):
    waveform = verification.load_audio(file_path)
    return verification.encode_batch(waveform).squeeze(1)

def record_audio(filename, duration=DURATION, with_watermark=False):
    print(f"Recording for {duration} seconds. State a command.")

    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    
    wav.write(filename, SAMPLE_RATE, recording)

def check_uncached_files(command_embedding):
    enrolled_files = [f for f in os.listdir('.') if f.startswith("enrolled_user_") and f.endswith(".wav")]
    for f in enrolled_files:
        user_id = f.replace(".wav", "")
        if user_id not in speaker_db:
            embed = get_embedding(f)
            score = verification.similarity(embed, command_embedding).item()
            if score > THRESHOLD:

                file_name_parts = f.replace('.wav', '').split('_')
                if len(file_name_parts) >= 3:
                    
                    if file_name_parts[-1].isdigit():
                        display_name = ' '.join(file_name_parts[2:-1])
                    else:
                        display_name = ' '.join(file_name_parts[2:])
                else:
                    display_name = user_id
                
                speaker_db[user_id] = {
                    "embedding": embed,
                    "timestamp": time.time(),
                    "display_name": display_name
                }
                torch.save(speaker_db, SPEAKER_EMBEDDINGS_FILE)
                maybe_evict_oldest_speaker()
                monitor_db_size()
                print(f"Re-inserted {user_id} into speaker_db with display name: {display_name}")
                return True, user_id, display_name, score
    return False, None, None, 0.0

def verify_speaker():
    command_file = "command.wav"
    record_audio(command_file)
    
    sample_rate, audio_data = wav.read(command_file)
    
    command_embedding = get_embedding(command_file)
    for user_id, data in speaker_db.items():
        score = verification.similarity(data["embedding"], command_embedding).item()
        print(f"User_id {user_id} Verification score: {score}")
        if score > THRESHOLD:
            print(f"Authentication successful for user_id {user_id} with score: {score}")
            data["timestamp"] = time.time()
            torch.save(speaker_db, SPEAKER_EMBEDDINGS_FILE)
            return True
    print("Not found in speaker_db. Checking uncached files...")
    if check_uncached_files(command_embedding):
        return True
    print("Authentication failed. No matching user found.")
    return False

def enroll_speaker(num):
    filename = f"enrolled_user_{num}.wav"
    print("Enrolling user ...")

    record_audio(filename, with_watermark=False)

    
    embedding = get_embedding(filename)
    speaker_db[f"user_{num}"] = {
        "embedding": embedding,
        "timestamp": time.time()
    }

    torch.save(speaker_db, SPEAKER_EMBEDDINGS_FILE)
    monitor_db_size()
    maybe_evict_oldest_speaker()
    print("Enrollment complete.")


def get_next_user_index():
    max_index = -1
    for key in speaker_db:
        if key.startswith("user_"):
            idx = int(key.split('_')[1])
            if idx > max_index:
                max_index = idx
    return max_index + 1


def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def verify_speaker_with_cached_embeddings(command_file):
    command_embedding = get_embedding(command_file)
    
    for user_id, data in speaker_db.items():
        score = verification.similarity(data["embedding"], command_embedding).item()
        if score > THRESHOLD:
            return True, user_id, score
    
    return False, None, 0

def verify_speaker_without_cache(command_file):
    """Verify speaker by recomputing embeddings from WAV files each time"""
    sample_rate, audio_data = wav.read(command_file)
    command_embedding = get_embedding(command_file)
    
    enrolled_files = [f for f in os.listdir('.') if f.startswith("enrolled_user_") and f.endswith(".wav")]
    for f in enrolled_files:
        user_id = f.replace(".wav", "").replace("enrolled_user_", "user_")
        embedding = get_embedding(f)
        score = verification.similarity(embedding, command_embedding).item()
        if score > THRESHOLD:
            return True, user_id, score
    
    return False, None, 0

def benchmark_verification_methods(iterations=5):
    print(f"Running Benchmark {iterations} times...")
    
    # Ensure we have a command file to test with
    command_file = "benchmark_command.wav"
    if not os.path.exists(command_file):
        print("Recording a sample command for benchmarking...")
        record_audio(command_file)
    
    # Test with cached embeddings
    cached_times = []
    cached_results = []
    for i in range(iterations):
        result, execution_time = time_function(verify_speaker_with_cached_embeddings, command_file)
        cached_times.append(execution_time)
        cached_results.append(result)
        print(f"Cached method - Iteration {i+1}: {execution_time:.4f} seconds")
    
    # Test without cached embeddings
    uncached_times = []
    uncached_results = []
    for i in range(iterations):
        result, execution_time = time_function(verify_speaker_without_cache, command_file)
        uncached_times.append(execution_time)
        uncached_results.append(result)
        print(f"Uncached method - Iteration {i+1}: {execution_time:.4f} seconds")
    
    # Calculate and display results
    avg_cached = sum(cached_times) / len(cached_times)
    avg_uncached = sum(uncached_times) / len(uncached_times)
    speedup = avg_uncached / avg_cached if avg_cached > 0 else 0
    
    print(f"Average time with cached embeddings: {avg_cached:.4f} seconds")
    print(f"Average time without cached embeddings: {avg_uncached:.4f} seconds")
    print(f"Speedup factor: {speedup:.2f}x")

def main():
    num_users = get_next_user_index()
    if num_users == 0:
        print("No users enrolled. Enroll a user first.")
        enroll_speaker(num_users)
        num_users += 1
    else:
        print(f"Currently enrolled users: {num_users}")
        
    active = input("Would you like to continue? y/n: ")

    while (active != "n"):
        enroll_new = input('Would you like to enroll a new user? y/n')

        if enroll_new == 'y':
            num_users = get_next_user_index()
            enroll_speaker(num_users)
            continue
        elif enroll_new == 'n':
            pass
        matched = verify_speaker()
        
        if matched:
            print("Command Accepted")
            if enroll_new == 'y':
                enroll_speaker(num_users)
                num_users+=1
            else:
                pass
                # use whisper here
        else:
            print("Command Denied")
            break
        
        active = input("Would you like to continue? Answer with yes or no.")

if __name__ == "__main__":
    main()

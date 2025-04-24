# Wake Word and Voiceprint Optimization in Voice Controllable Devices 

## End-to-End Model
`full_system`: Used to record commands, augment data, prepare data for training, train a model, and run the model. 

## Multi-User Embeddings
`multi_user`: Used to store registered user embeddings in a cache to optimize storage on an embedded device.  

## Replay Defense
`watermark`: Used for noise injection to create watermarks and prevent replay attacks on the authentication system. 

### Individual Contributions:

- Sydney Belt: Developed full end-to-end pipeline: wake-word & command authentication 
- Meha Goyal: Benchmarking Speech Brain model
- Arnav Shah: Developed multi-user pipeline with caching, replay attack defense, and dataset generation

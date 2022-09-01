# zzx_debug



## test sample


Sample Abdom
```
docker run --gpus all -v "/home/alex/mood_data/adbom/toy/:/mnt/data" -v "/home/alex/output:/mnt/pred" --read-only mood_example /workspace/run_sample_abdom.sh /mnt/data /mnt/pred
```

Sample Brain
```
docker run --gpus all -v "/home/alex/mood_data/brain/toy/:/mnt/data" -v "/home/alex/output:/mnt/pred" --read-only mood_example /workspace/run_sample_brain.sh /mnt/data /mnt/pred
```


## test pixel

Pixel Abdom
```
docker run --gpus all -v "/home/alex/mood_data/adbom/toy/:/mnt/data" -v "/home/alex/output:/mnt/pred" --read-only mood_example /workspace/run_pixel_abdom.sh /mnt/data /mnt/pred
```

Pixel Brain
```
docker run --gpus all -v "/home/alex/mood_data/brain/toy/:/mnt/data" -v "/home/alex/output:/mnt/pred" --read-only mood_example /workspace/run_pixel_brain.sh /mnt/data /mnt/pred
```

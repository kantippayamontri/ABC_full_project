
# ABC Gauge Detection

This project make to doing an ABC gauge detection focusing on object detection task 

## Dataset template

we use yolo like dataset which bounding box contain this information

| data             | description                                                                |
| ----------------- | ------------------------------------------------------------------ |
| x |  center x in range [0,1] |
| y |  center y in range[0,1] |
| w |  width of the image from center range [0,1] |
| h |  height of the image from center range [0,1] |


## Create config .yml file
before using preprocess and augmentation we need to create a config file in .yml format like this 

```yaml
AUGMENT:
  AUGMENT_LIST: 
   - RESIZE:
       TARGET_WIDTH: 640
       TARGET_HEIGHT: 640
       REPLACE: true
   - CHANNEL_SHUFFLE:
      P: 0.5
      REPLACE: false
   - MULTIPLICATIVE_NOISE:
      MULTIPLIER:
      - 0.4
      - 1.0
      ELEMENT_WISE: true
      P: 0.5
      REPLACE: false
   - BLUR:
      BLUR_LIMIT:
      - 3
      - 3 
      P: 0.5
      REPLACE: false
   - ROTATE:
      LIMIT:
      - -20
      - 20 
      P: 0.5
   - COLOR_JITTER:
      BRIGHTNESS: 0.5
      CONTRAST: 0.5
      SATURATION: 0.5
      HUE: 0.5
      P: 0.75
      REPLACE: false
   - LONGEST_MAX_SIZE:
      MAX_SIZE: 640
      P: 1.0
      REPLACE: false
   - PAD_IF_NEEDED:
      MIN_WIDTH: 640
      MIN_HEIGHT: 640
      P: 1.0
      REPLACE: false
  FOLDER:
  - train
  - valid
  # - test
  NUMBER_AUGMENT: 3
  PREFIX: aug
DATASET:
  DATASET_PATH:
  - ./dataset
  DATASET_TYPE: digital
  FINAL_DATASET_PATH: dataset_new
  PERCENT_TEST: 5
  PERCENT_TRAIN: 90
  PERCENT_VAL: 5
PREPROCESS:
  FOLDER:
  - train
  - valid
  - test
  PREFIX: pre
  PREPROCESS_LIST:
  - CROP:
      ADD_PIXEL: 50
      CLASS_CROP_LIST: 
      - 0
      - 1
      CLASS_IGNORE: []
      NEED_RESIZE: true
      TARGET_HEIGHT: 640
      TARGET_WIDTH: 640
      REPLACE: false
  - RESIZE:
      TARGET_HEIGHT: 640
      TARGET_WIDTH: 640
      REPLACE: true
  - GRAY:
      P: 1.0
      REPLACE: true
```

**Note** working process is Combind -> Preprocess -> Augment

## Preprocess

using command 

```bash
  python preprocess.py {path_to_config_yml_file}
```

```bash
  python preprocess.py /Users/kantip/Desktop/work/ABC_training/config/preprocess_config/preprocess.yml
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `path_to_config_yml_file` | `string` | **Required**. path to your yml config file |

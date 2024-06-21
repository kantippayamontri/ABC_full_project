# ABC Gauge Detection

This project was create do an ABC gauge detection focusing on object detection task focusing on data preprocessing, training  and evaluating

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
  - /media/kan/kan_ex/dataset_convert/user3_231109
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

**Note** working process order is Combind -> Preprocess -> Augment
1. Preprocess -> do each process (for crop operation)
2. Augmentation -> do all operation in the same time.

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

### Proprocess attributes for augmentation and preprocess
the source code provide in `./preprocess/transforms.py`
***NOTE:*** all the operation have parameter call REPLACE: BOOL if set to true -> the result image from each operation will overwrite original image. Please use carefully in preprocess approach.

- `CROP`
    - description: 
    this function use to crop the image in the original image to make new image -> use for digital imagaes that use to make crop gauge, frame class
    - parameters: 
        - `ADD_PIXEL`: INT 
        add pixel in bounding box that use to crop the orginal images -> make crop image not too tight.
        - CLASS_CROP_LSIT: LIST[INT]
        List of index of the class in yaml file that use to crop from the original images.
        - `CLASS_IGNORE`: LIST[INT]
        List of index of the class in yaml file that ignore not appear in the final dataset.
        - `NEED_RESIZE`: BOOL
        crop image may by very small if we want size of the images equal to the input size of the ML model set to "true".
        if we set to "true" -> we need to set  TARGET_HEIGHT [INT] and TARGET_WIDTH [INT].
        - `TARGET_HEIGHT`: INT
        resize the crop images height to TARGET_HEIGHT.
        - `TARGET_WIDTH`: INT
        resize the crop images width to TARGET_WIDTH.
    - code:
        ```
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
        ```

- `RESIZE`
    - description:
    this funciton use to resize the image from (h,w) to (TARGET_HEIGHT, TARGET_WIDTH)
    - parameters:
        - `TARGET_HEIGHT`: INT
        height of the target image 
        - `TARGET_WIDTH`: INT
        width of the target image
    - code:
    ```
      - RESIZE:
          TARGET_HEIGHT: 640
          TARGET_WIDTH: 640
          REPLACE: true
    ```

- `GRAY`
    - description:
    this function make gray scale image
    - parameters:
        - `P`: INT
        Probability of applying the transform if set P=1 -> all output grayscale 100% , if set P=0 -> all output grayscale 0%
   - code:
    ```
      - GRAY:
          P: 1.0
          REPLACE: true
    ```

- `CHANNEL_SHUFFLE`
    - description
    Randomly rearrange channels of the image.
    - parameters
        - `P`
        Probability of applying the transform
    - code
    ```
    - CHANNEL_SHUFFLE:
          P: 0.5
          REPLACE: false
    ```

- `MULTIPLICATIVE_NOISE`
    - description
    Multiply image by a random number or array of numbers.
    - parameters
        - `MULTIPLIER`: List[FLOAT]
        list of float size 2 [start, end] that the value between start and end will randomly choose and multiply the images. 
        ***Note:*** the value the default value from ALBUMENTAION (https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.transforms.MultiplicativeNoise) is [0.9, 1.1] is (`you need to set`)
        ***Fix later: for now you need to input 2 parameters***
        - `ELEMENT_WISE`: BOOL
        If `False`, multiply all pixels in the image by a single random value sampled once.If `True`, multiply image pixels by values that are pixelwise randomly sampled.
        - `P`: FLOAT
         Probability of applying the transform
    - code
    ```
    - MULTIPLICATIVE_NOISE:
          MULTIPLIER:
          - 0.4
          - 1.0
          ELEMENT_WISE: true
          P: 0.5
          REPLACE: false
    ```
    
- `BLUR`
    - description
    Blur the input image using a random-sized kernel.
    - parameters
        - `BLUR_LIMIT`: LIST[start, end]
        maximum kernel size for blurring the input image. Should be in range [3, inf). Default: (3, 7).
        ***Fix later: for this project you need to input list of size 2***
        - `P`: FLOAT
        Probability of applying the transform
    - code
    ```
    - BLUR:
          BLUR_LIMIT:
          - 3
          - 3 
          P: 0.5
          REPLACE: false
    ```

- `ROTATE`
    - description
    Rotate the input by an angle selected randomly from the uniform distribution.
    - parameters
        - `LIMIT`: LIST[min_degree, max_degree]
        range from which a random angle is picked. If limit is a single int an angle is picked from (-limit, limit). Default: (-90, 90)
        ***FIX ME: from the albumentation website have more parameters (check this: https://albumentations.ai/docs/api_reference/augmentations/geometric/rotate/?h=rotate#albumentations.augmentations.geometric.rotate.Rotate)***
        - `P`: FLOAT
        Probability of applying the transform
    - code
    ```
    - ROTATE:
          LIMIT:
          - -20
          - 20 
          P: 0.5
    ```

- `COLOR_JITTER`
    - description
    Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision, this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8 overflow, but we use value saturation.
    - parameters
        - `BRIGHTNESS`: FLOAT (***FIX to float or tuple of float (min, max)***)
        How much to jitter brightness. If float: brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] If Tuple[float, float]] will be sampled from that range. Both values should be non negative numbers.
        - `CONTRAST`: FLOAT (***FIX to float or tuple of float (min, max)***)
        How much to jitter contrast. If float: contrast_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] If Tuple[float, float]] will be sampled from that range. Both values should be non negative numbers.
        - `SATURATION`: FLOAT (***FIX to float or tuple of float (min, max)***)
        How much to jitter saturation. If float: saturation_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] If Tuple[float, float]] will be sampled from that range. Both values should be non negative numbers.
        - `HUE`: FLOAT (***FIX to float or tuple of float (min, max)***)
        How much to jitter hue. If float: saturation_factor is chosen uniformly from [-hue, hue]. Should have 0 <= hue <= 0.5. If Tuple[float, float]] will be sampled from that range. Both values should be in range [-0.5, 0.5].
        - `P`: FLOAT
        Probability of applying the transform
    - code
    ```
    - COLOR_JITTER:
          BRIGHTNESS: 0.5
          CONTRAST: 0.5
          SATURATION: 0.5
          HUE: 0.5
          P: 0.75
          REPLACE: false
    ```

- `LONGEST_MAX_SIZE`
    - description
    Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.
    ปรับขนาดรูปภาพใหม่เพื่อให้ด้านสูงสุดเท่ากับ max_size โดยคงอัตราส่วนของรูปภาพเริ่มต้นไว้
    - parameters
        - `MAX_SIZE`: INT
            maximum size of the image after the transformation. When using a list, max size will be randomly selected from the values in the list.
        - ***FIX this to have this parameters*** `interpolation` : OpenCV flag
            interpolation method. Default: cv2.INTER_LINEAR.
        - `P`:FLOAT
          Probability of applying the transform
    - code
    ```
       - LONGEST_MAX_SIZE:
          MAX_SIZE: 640
          P: 1.0
          REPLACE: false
    ```
    
- `PAD_IF_NEEDED`
    - description
    Pads the sides of an image if the image dimensions are less than the specified minimum dimensions.
    - parameter
        - `MIN_WIDTH`: INT
        Minimum desired width of the image. Ensures image width is at least this value.
        - `MIN_HEIGHT`: INT
        Minimum desired height of the image. Ensures image height is at least this value.
        - ***FIX to add parameters*** `pad_height_divisor`, `pad_width_divisor`
    - code
    ```
    - PAD_IF_NEEDED:
          MIN_WIDTH: 640
          MIN_HEIGHT: 640
          P: 1.0
          REPLACE: false
    ```



## Training
***FIXME: fix to add more model yolov8 9 10***

## Create train .yml file like this
```yaml
DATASET:
  DATASET_PATH: dataset_new
  DATASET_TYPE: digital
MODEL:
  MODEL_PATH: null
  MODEL_TYPE: MEDIUM
TRAIN_PARAMETERS:
  BATCH_SIZE: 4
  CACHE: false
  DEVICE: 0
  EPOCHS: 20
  FINAL_LEARNING_RATE: 0.01
  IMG_SIZE: 640
  LEARNING_RATE: 0.001
  PATIENCE: 15
  RESUME: true
  WORKERS: 4
```

## Train.yml attribute
- `DATASET`
    - description
        set to make YOLO model know the dataset path and dataset type
    - parameters
        - `DATASET_PATH`: PATH
            path to folder that have dataset type folder name inside For ex, if we set `DATASET_PATH` = dataset_new , the dataset_new folder must have folder digital, dial, number, level, inside
        - `DATASET_TYPE`: PATH
            type of dataset that contain the data ex. digital, number, dial, level

- `MODEL`
    - description
        to tell model that use to train and size of model that use to evaluate
    - parameters
        - `MODEL_PATH`: PATH
        set to takes the model for transfer learning
        - `MODEL_TYPE`: PATH
        set to type of model that want to train -> the path is `{MODEL_PATH}/{MODEL_TYPE}`

- `TRAIN_PARAMETERS`
    - description
    set parameters for training
    - parameters
        - `BATCH_SIZE`: INT
        - `CACHE`: INT
        - `DEVICE`: INT, LIST[INT]
        - `EPOCHS`: INT
        - `FINAL_LEARNING_RATE`: FLOAT
        - `IMG_SIZE`: INT
        - `LEARNING_RATE`: FLOAT
        - `PATIENCE`: INT
        - `RESUME`: BOOL
        - `WORKERS`: INT

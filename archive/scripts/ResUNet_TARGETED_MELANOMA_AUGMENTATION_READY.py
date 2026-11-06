# ============================================================
# AI-ASSISTED DIAGNOSIS OF CUTANEOUS MELANOMA
# FINAL RESUNET + EFFICIENTNETB0 PIPELINE
# Matched to Oneli's Attention U-Net experimental setup
# ============================================================

import os, random, time, zipfile, shutil
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from IPython.display import display

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, classification_report,
)

# ============================================================
# 1. REPRODUCIBILITY + CONFIGURATION
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

TOTAL_CASES = 300
MODELLING_CASES = 200
FINAL_TEST_CASES = 100
MODEL_TRAIN_CASES = 160
MODEL_VAL_CASES = 40

SEG_IMAGE_SIZE = (256, 256)
CLS_IMAGE_SIZE = (224, 224)
SEG_BATCH_SIZE = 4
CLS_BATCH_SIZE = 16
SEG_EPOCHS = 40
CLS_HEAD_EPOCHS = 15
CLS_FINE_TUNE_EPOCHS = 10
SEG_LEARNING_RATE = 1e-3
CLS_HEAD_LR = 1e-3
CLS_FINE_TUNE_LR = 1e-5
SEG_THRESHOLD = 0.50
CLS_THRESHOLD = 0.50
DO_FINE_TUNING = True
EPS = 1e-7
AUTOTUNE = tf.data.AUTOTUNE

print('TensorFlow:', tf.__version__)
print('GPU devices:', tf.config.list_physical_devices('GPU'))

# ============================================================
# 2. STORAGE
# Large data in /content; only outputs/models in Drive
# ============================================================
RUNTIME_ROOT = Path('/content/melanoma_resunet_300_targeted_aug')
DOWNLOAD_DIR = RUNTIME_ROOT / 'downloads'
DATA_DIR = RUNTIME_ROOT / 'selected_300'
ROI_DIR = RUNTIME_ROOT / 'roi_images'
AUGMENTED_ROI_DIR = RUNTIME_ROOT / 'targeted_augmented_roi_images'
for folder in [RUNTIME_ROOT, DOWNLOAD_DIR, DATA_DIR, ROI_DIR, AUGMENTED_ROI_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

try:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_DIR = Path('/content/drive/MyDrive/Melanoma_AI_ResUNet_Targeted_Malignant_Aug_Results')
except Exception:
    PROJECT_DIR = RUNTIME_ROOT / 'results'

MODELS_DIR = PROJECT_DIR / 'models'
RESULTS_DIR = PROJECT_DIR / 'results'
FIGURES_DIR = PROJECT_DIR / 'figures'
SPLITS_DIR = PROJECT_DIR / 'splits'
for folder in [PROJECT_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, SPLITS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print('Runtime root:', RUNTIME_ROOT)
print('Persistent outputs:', PROJECT_DIR)

# ============================================================
# 3. DOWNLOAD OFFICIAL ISIC 2016 TASK 3B TRAINING DATA
# Same source used in Oneli's pipeline
# ============================================================
DATA_ZIP_URL = (
    'https://isic-archive.s3.amazonaws.com/challenges/2016/'
    'ISBI2016_ISIC_Part3B_Training_Data.zip'
)
LABEL_CSV_URL = (
    'https://isic-archive.s3.amazonaws.com/challenges/2016/'
    'ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'
)
DATA_ZIP = DOWNLOAD_DIR / 'ISBI2016_ISIC_Part3B_Training_Data.zip'
LABEL_CSV = DOWNLOAD_DIR / 'ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'


def download_file(url, destination, chunk_mb=8):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 100:
        print('Already downloaded:', destination.name)
        return destination
    print('Downloading:', destination.name)
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0)) or None
        with open(destination, 'wb') as output_file, tqdm(
            total=total, unit='B', unit_scale=True, desc=destination.name
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    output_file.write(chunk)
                    progress.update(len(chunk))
    return destination


download_file(LABEL_CSV_URL, LABEL_CSV)
download_file(DATA_ZIP_URL, DATA_ZIP)
assert zipfile.is_zipfile(DATA_ZIP), 'Downloaded Task 3B ZIP is invalid.'
print('Official ISIC 2016 Task 3B files ready.')

# ============================================================
# 4. LOAD 900 LABELS, SELECT STRATIFIED 300, SPLIT 160/40/100
# ============================================================
labels_df = pd.read_csv(LABEL_CSV, header=None, names=['image_id', 'diagnosis'])
labels_df['diagnosis'] = labels_df['diagnosis'].astype(str).str.strip().str.lower()
labels_df['label'] = labels_df['diagnosis'].map({'benign': 0, 'malignant': 1})
assert len(labels_df) == 900, f'Expected 900 labels, found {len(labels_df)}.'
assert labels_df['label'].notna().all(), 'Unexpected diagnosis values.'
assert labels_df['image_id'].is_unique, 'Duplicate image IDs found.'

print('\nFull ISIC 2016 Task 3B label distribution:')
display(labels_df['diagnosis'].value_counts().to_frame('count'))

selected_cases, _ = train_test_split(
    labels_df,
    train_size=TOTAL_CASES,
    stratify=labels_df['label'],
    random_state=SEED,
)
selected_cases = selected_cases.reset_index(drop=True)
assert len(selected_cases) == TOTAL_CASES

print('\nSelected 300-case distribution:')
display(selected_cases['diagnosis'].value_counts().to_frame('count'))

model_df, test_df = train_test_split(
    selected_cases,
    train_size=MODELLING_CASES,
    test_size=FINAL_TEST_CASES,
    stratify=selected_cases['label'],
    random_state=SEED,
)
train_df, val_df = train_test_split(
    model_df,
    train_size=MODEL_TRAIN_CASES,
    test_size=MODEL_VAL_CASES,
    stratify=model_df['label'],
    random_state=SEED,
)

for frame_name in ['train_df', 'val_df', 'model_df', 'test_df']:
    globals()[frame_name] = globals()[frame_name].reset_index(drop=True)

assert len(train_df) == 160 and len(val_df) == 40 and len(test_df) == 100
assert set(train_df.image_id).isdisjoint(set(val_df.image_id))
assert set(train_df.image_id).isdisjoint(set(test_df.image_id))
assert set(val_df.image_id).isdisjoint(set(test_df.image_id))


def show_split(name, frame):
    counts = frame['diagnosis'].value_counts()
    percentages = frame['diagnosis'].value_counts(normalize=True).mul(100).round(1)
    print(f'\n{name}: n={len(frame)}')
    display(pd.DataFrame({'count': counts, 'percent': percentages}))


show_split('TRAIN', train_df)
show_split('VALIDATION', val_df)
show_split('FINAL TEST', test_df)
print('\nZERO IMAGE-ID OVERLAP CONFIRMED.')

train_df[['image_id', 'diagnosis', 'label']].to_csv(SPLITS_DIR / 'train_160.csv', index=False)
val_df[['image_id', 'diagnosis', 'label']].to_csv(SPLITS_DIR / 'validation_40.csv', index=False)
test_df[['image_id', 'diagnosis', 'label']].to_csv(SPLITS_DIR / 'final_test_100.csv', index=False)

# ============================================================
# 5. EXTRACT ONLY THE SELECTED 300 IMAGE-MASK PAIRS
# ============================================================
selected_ids = set(selected_cases['image_id'])
with zipfile.ZipFile(DATA_ZIP, 'r') as archive:
    members = [m for m in archive.namelist() if not m.endswith('/')]
    by_basename = {Path(m).name: m for m in members}
    records, missing_images, missing_masks = [], [], []

    for image_id in tqdm(sorted(selected_ids), desc='Extracting selected 300 cases'):
        image_base = f'{image_id}.jpg'
        mask_base = f'{image_id}_Segmentation.png'
        image_member = by_basename.get(image_base)
        mask_member = by_basename.get(mask_base)
        if image_member is None:
            missing_images.append(image_id)
            continue
        if mask_member is None:
            missing_masks.append(image_id)
            continue

        image_output = DATA_DIR / image_base
        mask_output = DATA_DIR / mask_base
        if not image_output.exists():
            with archive.open(image_member) as src, open(image_output, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        if not mask_output.exists():
            with archive.open(mask_member) as src, open(mask_output, 'wb') as dst:
                shutil.copyfileobj(src, dst)

        records.append({
            'image_id': image_id,
            'image_path': str(image_output),
            'mask_path': str(mask_output),
        })

assert not missing_images, f'Missing images: {missing_images[:10]}'
assert not missing_masks, f'Missing masks: {missing_masks[:10]}'
assert len(records) == TOTAL_CASES, f'Expected 300 extracted cases, found {len(records)}.'
path_df = pd.DataFrame(records)


def add_paths(frame):
    result = frame.merge(path_df, on='image_id', how='left', validate='one_to_one')
    assert result[['image_path', 'mask_path']].notna().all().all()
    return result


train_df = add_paths(train_df)
val_df = add_paths(val_df)
model_df = add_paths(model_df)
test_df = add_paths(test_df)
selected_cases = add_paths(selected_cases)

if DATA_ZIP.exists():
    DATA_ZIP.unlink()
    print('Deleted temporary large ZIP.')

# Dataset distribution
plt.figure(figsize=(6, 4))
counts = selected_cases['diagnosis'].value_counts().reindex(['benign', 'malignant'])
plt.bar(['Benign', 'Melanoma'], counts.values)
for i, v in enumerate(counts.values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.ylabel('Number of images')
plt.title('ISIC 2016 Task 3B - Selected 300 Cases')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dataset_distribution.png', dpi=180)
plt.show()

# ============================================================
# 6. IMAGE HELPERS + STANDARDISED OVERLAY
# ============================================================
def load_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB').copy()


def load_mask(path):
    with Image.open(path) as image:
        return image.convert('L').copy()


def segmentation_overlay(image_pil, mask_binary, alpha=0.30):
    img_arr = np.asarray(image_pil.convert('RGB')).astype(np.float32)
    mask_pil = Image.fromarray((np.asarray(mask_binary).astype(np.uint8) * 255))
    mask = np.asarray(mask_pil.resize(image_pil.size, Image.Resampling.NEAREST)) > 127
    overlay = img_arr.copy()
    overlay[mask, 0] = np.clip(overlay[mask, 0] * (1 - alpha) + 255 * alpha, 0, 255)
    return Image.fromarray(overlay.astype(np.uint8))

# Quick visual sanity check
sample_rows = pd.concat([
    train_df[train_df['label'] == 0].head(2),
    train_df[train_df['label'] == 1].head(2),
]).reset_index(drop=True)
fig, axes = plt.subplots(len(sample_rows), 3, figsize=(12, 4 * len(sample_rows)))
for r, row in sample_rows.iterrows():
    image = load_rgb(row['image_path'])
    mask = (np.asarray(load_mask(row['mask_path'])) > 127).astype(np.uint8)
    axes[r, 0].imshow(image)
    axes[r, 0].set_title(f"{row['image_id']} | {row['diagnosis']}")
    axes[r, 1].imshow(mask, cmap='gray')
    axes[r, 1].set_title('Expert Ground-Truth Mask')
    axes[r, 2].imshow(segmentation_overlay(image, mask))
    axes[r, 2].set_title('Ground-Truth Overlay')
    for ax in axes[r]:
        ax.axis('off')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dataset_sanity.png', dpi=180)
plt.show()

# ============================================================
# 7. tf.data SEGMENTATION PIPELINE
# Same preprocessing and augmentation style as Oneli
# ============================================================
def decode_segmentation_pair(image_path, mask_path):
    image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)
    image = tf.image.resize(image, SEG_IMAGE_SIZE, method='bilinear')
    image = tf.cast(image, tf.float32) / 255.0
    image.set_shape((*SEG_IMAGE_SIZE, 3))

    mask = tf.io.decode_png(tf.io.read_file(mask_path), channels=1)
    mask = tf.image.resize(mask, SEG_IMAGE_SIZE, method='nearest')
    mask = tf.cast(mask > 127, tf.float32)
    mask.set_shape((*SEG_IMAGE_SIZE, 1))
    return image, mask


def paired_segmentation_augmentation(image, mask):
    seed1 = tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32)
    seed2 = tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32)
    image = tf.image.stateless_random_flip_left_right(image, seed1)
    mask = tf.image.stateless_random_flip_left_right(mask, seed1)
    image = tf.image.stateless_random_flip_up_down(image, seed2)
    mask = tf.image.stateless_random_flip_up_down(mask, seed2)
    image = tf.image.random_brightness(image, max_delta=0.06)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask


def make_seg_dataset(frame, training=False):
    ds = tf.data.Dataset.from_tensor_slices((
        frame['image_path'].astype(str).values,
        frame['mask_path'].astype(str).values,
    ))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_segmentation_pair, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(paired_segmentation_augmentation, num_parallel_calls=AUTOTUNE)
    return ds.batch(SEG_BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)


train_seg_ds = make_seg_dataset(train_df, training=True)
val_seg_ds = make_seg_dataset(val_df, training=False)
test_seg_ds = make_seg_dataset(test_df, training=False)

batch_images, batch_masks = next(iter(train_seg_ds))
print('Image batch shape:', batch_images.shape)
print('Mask batch shape:', batch_masks.shape)
print('Mask values:', np.unique(batch_masks.numpy()))

# ============================================================
# 8. SEGMENTATION METRICS + LOSS
# ============================================================
@keras.utils.register_keras_serializable(package='MelanomaAI')
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + EPS) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + EPS
    )


@keras.utils.register_keras_serializable(package='MelanomaAI')
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


@keras.utils.register_keras_serializable(package='MelanomaAI')
def iou_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > SEG_THRESHOLD, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + EPS) / (union + EPS)


@keras.utils.register_keras_serializable(package='MelanomaAI')
def combined_segmentation_loss(y_true, y_pred):
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


CUSTOM_OBJECTS = {
    'dice_coefficient': dice_coefficient,
    'dice_loss': dice_loss,
    'iou_coefficient': iou_coefficient,
    'combined_segmentation_loss': combined_segmentation_loss,
}

# Sanity test
_test_true = tf.constant([[[[1.0]], [[0.0]]]])
_test_pred = tf.constant([[[[1.0]], [[0.0]]]])
print('Perfect Dice:', float(dice_coefficient(_test_true, _test_pred)))
print('Perfect IoU:', float(iou_coefficient(_test_true, _test_pred)))

# ============================================================
# 9. RESUNET ARCHITECTURE
# Main architectural change from Oneli's Attention U-Net
# ============================================================
def residual_block(x, filters, name):
    shortcut = x

    y = layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv1')(x)
    y = layers.BatchNormalization(name=f'{name}_bn1')(y)
    y = layers.Activation('relu', name=f'{name}_relu1')(y)

    y = layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv2')(y)
    y = layers.BatchNormalization(name=f'{name}_bn2')(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters, 1, padding='same', use_bias=False, name=f'{name}_shortcut_conv'
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    y = layers.Add(name=f'{name}_add')([y, shortcut])
    return layers.Activation('relu', name=f'{name}_output')(y)


def build_resunet(input_shape=(256, 256, 3)):
    inputs = keras.Input(shape=input_shape, name='image')

    c1 = residual_block(inputs, 32, 'enc1')
    p1 = layers.MaxPooling2D(name='pool1')(c1)
    c2 = residual_block(p1, 64, 'enc2')
    p2 = layers.MaxPooling2D(name='pool2')(c2)
    c3 = residual_block(p2, 128, 'enc3')
    p3 = layers.MaxPooling2D(name='pool3')(c3)
    c4 = residual_block(p3, 256, 'enc4')
    p4 = layers.MaxPooling2D(name='pool4')(c4)

    bottleneck = residual_block(p4, 512, 'bottleneck')

    def decoder_step(x, skip, filters, name):
        x = layers.Conv2DTranspose(
            filters, 2, strides=2, padding='same', name=f'{name}_up'
        )(x)
        x = layers.Concatenate(name=f'{name}_concat')([x, skip])
        return residual_block(x, filters, name)

    d4 = decoder_step(bottleneck, c4, 256, 'dec4')
    d3 = decoder_step(d4, c3, 128, 'dec3')
    d2 = decoder_step(d3, c2, 64, 'dec2')
    d1 = decoder_step(d2, c1, 32, 'dec1')

    outputs = layers.Conv2D(
        1, 1, activation='sigmoid', dtype='float32', name='segmentation_mask'
    )(d1)
    return keras.Model(inputs, outputs, name='ResUNet')


tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
resunet = build_resunet(input_shape=(*SEG_IMAGE_SIZE, 3))
dummy_output = resunet(tf.zeros((1, *SEG_IMAGE_SIZE, 3), dtype=tf.float32), training=False)
assert dummy_output.shape == (1, 256, 256, 1)
print('ResUNet parameters:', f'{resunet.count_params():,}')
print('ResUNet synthetic shape test passed.')

resunet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=SEG_LEARNING_RATE),
    loss=combined_segmentation_loss,
    metrics=[dice_coefficient, iou_coefficient],
)

RESUNET_MODEL_PATH = MODELS_DIR / 'resunet_best.keras'
segmentation_callbacks = [
    keras.callbacks.ModelCheckpoint(
        RESUNET_MODEL_PATH,
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_dice_coefficient',
        mode='max',
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_dice_coefficient',
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ============================================================
# 10. TRAIN RESUNET
# ============================================================
print('\nSTARTING RESUNET TRAINING...')
seg_start = time.time()
resunet_fit = resunet.fit(
    train_seg_ds,
    validation_data=val_seg_ds,
    epochs=SEG_EPOCHS,
    callbacks=segmentation_callbacks,
    verbose=1,
)
resunet_training_seconds = time.time() - seg_start
resunet_history = resunet_fit.history

selected_segmenter = keras.models.load_model(
    RESUNET_MODEL_PATH,
    custom_objects=CUSTOM_OBJECTS,
)
print('Best ResUNet checkpoint reloaded.')

# Training curves
history = resunet_history
epochs = range(1, len(history['loss']) + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, title in [
    (axes[0], 'loss', 'ResUNet Loss'),
    (axes[1], 'dice_coefficient', 'ResUNet Dice'),
    (axes[2], 'iou_coefficient', 'ResUNet IoU / Jaccard'),
]:
    ax.plot(epochs, history[metric], label='Training')
    ax.plot(epochs, history[f'val_{metric}'], linestyle='--', label='Validation')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.25)
    ax.legend()
plt.tight_layout()
RESUNET_CURVES_PATH = FIGURES_DIR / 'resunet_training_curves.png'
plt.savefig(RESUNET_CURVES_PATH, dpi=180, bbox_inches='tight')
plt.show()

# ============================================================
# 11. FINAL RESUNET TEST METRICS
# ============================================================
def predict_mask_for_pil(model, image_pil):
    resized = image_pil.resize(
        (SEG_IMAGE_SIZE[1], SEG_IMAGE_SIZE[0]), Image.Resampling.BILINEAR
    )
    x = np.asarray(resized, dtype=np.float32) / 255.0
    probability = model.predict(x[None, ...], verbose=0)[0, ..., 0]
    binary = (probability >= SEG_THRESHOLD).astype(np.uint8)
    return probability, binary


keras_test_results = selected_segmenter.evaluate(test_seg_ds, verbose=0, return_dict=True)


def evaluate_resunet_per_image(model, frame):
    rows = []
    total_tp = total_tn = total_fp = total_fn = 0

    for _, row in tqdm(frame.iterrows(), total=len(frame), desc='Final ResUNet test'):
        image = load_rgb(row['image_path'])
        gt = np.asarray(
            load_mask(row['mask_path']).resize(
                (SEG_IMAGE_SIZE[1], SEG_IMAGE_SIZE[0]), Image.Resampling.NEAREST
            )
        ) > 127
        _, pred = predict_mask_for_pil(model, image)
        pred = pred.astype(bool)

        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        dice = (2.0 * intersection + EPS) / (gt.sum() + pred.sum() + EPS)
        iou = (intersection + EPS) / (union + EPS)

        tp_pixels = np.logical_and(gt, pred).sum()
        tn_pixels = np.logical_and(~gt, ~pred).sum()
        fp_pixels = np.logical_and(~gt, pred).sum()
        fn_pixels = np.logical_and(gt, ~pred).sum()
        total_tp += int(tp_pixels)
        total_tn += int(tn_pixels)
        total_fp += int(fp_pixels)
        total_fn += int(fn_pixels)

        rows.append({
            'image_id': row['image_id'],
            'diagnosis': row['diagnosis'],
            'image_path': row['image_path'],
            'mask_path': row['mask_path'],
            'dice': float(dice),
            'iou': float(iou),
        })

    results_df = pd.DataFrame(rows)
    sensitivity = total_tp / (total_tp + total_fn + EPS)
    specificity = total_tn / (total_tn + total_fp + EPS)
    return results_df, float(sensitivity), float(specificity)


resunet_test_per_image_df, pixel_sensitivity, pixel_specificity = evaluate_resunet_per_image(
    selected_segmenter, test_df
)

TEST_DICE = float(resunet_test_per_image_df['dice'].mean())
TEST_JACCARD = float(resunet_test_per_image_df['iou'].mean())
TEST_LOSS = float(keras_test_results['loss'])
BEST_TEST_DICE = float(resunet_test_per_image_df['dice'].max())
MEDIAN_TEST_DICE = float(resunet_test_per_image_df['dice'].median())
WORST_TEST_DICE = float(resunet_test_per_image_df['dice'].min())
BEST_VALIDATION_DICE = float(np.max(resunet_history['val_dice_coefficient']))
RESUNET_PARAMETERS = int(selected_segmenter.count_params())

resunet_final_metrics = pd.DataFrame({
    'Metric': [
        'Test Dice', 'Test Jaccard / IoU', 'Test Loss',
        'Best Test Dice', 'Median Test Dice', 'Worst Test Dice',
        'Best Validation Dice', 'Parameters',
        'Pixel Sensitivity', 'Pixel Specificity', 'Training Time (min)',
    ],
    'Value': [
        TEST_DICE, TEST_JACCARD, TEST_LOSS,
        BEST_TEST_DICE, MEDIAN_TEST_DICE, WORST_TEST_DICE,
        BEST_VALIDATION_DICE, RESUNET_PARAMETERS,
        pixel_sensitivity, pixel_specificity, resunet_training_seconds / 60.0,
    ],
})

print('\n' + '=' * 70)
print('RESUNET - FINAL TEST RESULTS')
print('=' * 70)
display(resunet_final_metrics)
resunet_final_metrics.to_csv(RESULTS_DIR / 'FINAL_resunet_metrics.csv', index=False)
resunet_test_per_image_df.to_csv(RESULTS_DIR / 'resunet_test_case_scores.csv', index=False)

# ============================================================
# 12. BEST / MEDIAN / WORST RESUNET SEGMENTATIONS
# ============================================================
sorted_seg = resunet_test_per_image_df.sort_values('dice').reset_index(drop=True)
worst_row = sorted_seg.iloc[0]
best_row = sorted_seg.iloc[-1]
median_target = float(sorted_seg['dice'].median())
median_row = sorted_seg.loc[(sorted_seg['dice'] - median_target).abs().idxmin()]
representative_rows = [('BEST', best_row), ('MEDIAN', median_row), ('WORST', worst_row)]

fig, axes = plt.subplots(3, 4, figsize=(16, 13))
for r, (case_name, result_row) in enumerate(representative_rows):
    image = load_rgb(result_row['image_path']).resize(
        (SEG_IMAGE_SIZE[1], SEG_IMAGE_SIZE[0]), Image.Resampling.BILINEAR
    )
    gt = (np.asarray(load_mask(result_row['mask_path']).resize(
        (SEG_IMAGE_SIZE[1], SEG_IMAGE_SIZE[0]), Image.Resampling.NEAREST
    )) > 127).astype(np.uint8)
    _, pred = predict_mask_for_pil(selected_segmenter, image)

    axes[r, 0].imshow(image)
    axes[r, 0].set_title(f"{case_name} - Original\n{result_row['image_id']}")
    axes[r, 1].imshow(gt, cmap='gray')
    axes[r, 1].set_title('Ground-Truth Mask')
    axes[r, 2].imshow(pred, cmap='gray')
    axes[r, 2].set_title('ResUNet Predicted Mask')
    axes[r, 3].imshow(segmentation_overlay(image, pred))
    axes[r, 3].set_title(f"Dice={result_row['dice']:.3f} | IoU={result_row['iou']:.3f}")
    for ax in axes[r]:
        ax.axis('off')

plt.tight_layout()
SEG_EXAMPLES_PATH = FIGURES_DIR / 'resunet_best_median_worst.png'
plt.savefig(SEG_EXAMPLES_PATH, dpi=180, bbox_inches='tight')
plt.show()

# ============================================================
# 13. GENERATE LESION ROIs FROM RESUNET PREDICTIONS
# Crop the ORIGINAL RGB image; do not classify masks
# ============================================================
def extract_roi_from_binary_mask(image_pil, binary_mask, padding_fraction=0.10):
    image_array = np.asarray(image_pil.convert('RGB'))
    h, w = image_array.shape[:2]
    mask = np.asarray(
        Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(
            (w, h), Image.Resampling.NEAREST
        )
    ) > 0

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image_pil.copy(), (0, 0, w, h), True

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    box_w, box_h = max(1, x_max - x_min + 1), max(1, y_max - y_min + 1)
    pad_x = int(round(box_w * padding_fraction))
    pad_y = int(round(box_h * padding_fraction))
    x0, y0 = max(0, x_min - pad_x), max(0, y_min - pad_y)
    x1, y1 = min(w, x_max + pad_x + 1), min(h, y_max + pad_y + 1)

    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return image_pil.copy(), (0, 0, w, h), True

    return image_pil.crop((x0, y0, x1, y1)), (x0, y0, x1, y1), False


roi_records = []
for _, row in tqdm(selected_cases.iterrows(), total=len(selected_cases), desc='Generating ROIs'):
    original = load_rgb(row['image_path'])
    _, predicted_mask = predict_mask_for_pil(selected_segmenter, original)
    roi, bbox, fallback = extract_roi_from_binary_mask(original, predicted_mask)
    roi_path = ROI_DIR / f"{row['image_id']}.jpg"
    roi.save(roi_path, quality=95)
    roi_records.append({
        'image_id': row['image_id'],
        'roi_path': str(roi_path),
        'roi_fallback': bool(fallback),
        'bounding_box': str(bbox),
    })

roi_manifest = pd.DataFrame(roi_records)
print('ROI generation complete.')
print('Fallback ROIs:', int(roi_manifest['roi_fallback'].sum()), 'out of', len(roi_manifest))

train_cls_df = train_df.merge(roi_manifest, on='image_id', validate='one_to_one')
val_cls_df = val_df.merge(roi_manifest, on='image_id', validate='one_to_one')
test_cls_df = test_df.merge(roi_manifest, on='image_id', validate='one_to_one')


# ============================================================
# 14A. TARGETED MALIGNANT-ONLY ROTATION AUGMENTATION
# Purpose: reduce benign bias AFTER segmentation / ROI extraction
# Only the TRAINING set is augmented. Validation and test remain untouched.
# Each malignant training ROI gets 90°, 180°, and 270° rotated copies.
# 31 malignant originals -> 124 malignant training samples total.
# Benign samples remain at 129, giving an almost 1:1 classifier training set.
# ============================================================
AUGMENTED_ROI_DIR.mkdir(parents=True, exist_ok=True)

train_cls_original_df = train_cls_df.copy()
train_cls_original_df['source_image_id'] = train_cls_original_df['image_id'].astype(str)
train_cls_original_df['augmentation'] = 'original'

rotation_ops = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}

augmented_rows = []
malignant_train_df = train_cls_df[train_cls_df['label'] == 1].copy()

for _, row in tqdm(
    malignant_train_df.iterrows(),
    total=len(malignant_train_df),
    desc='Creating malignant-only rotations'
):
    with Image.open(row['roi_path']) as img:
        roi_rgb = img.convert('RGB')

        for angle, transpose_op in rotation_ops.items():
            rotated = roi_rgb.transpose(transpose_op)
            aug_id = f"{row['image_id']}_rot{angle}"
            aug_path = AUGMENTED_ROI_DIR / f"{aug_id}.jpg"
            rotated.save(aug_path, quality=95)

            new_row = row.copy()
            new_row['image_id'] = aug_id
            new_row['roi_path'] = str(aug_path)
            new_row['source_image_id'] = str(row['image_id'])
            new_row['augmentation'] = f'rotation_{angle}'
            augmented_rows.append(new_row)

augmented_malignant_df = pd.DataFrame(augmented_rows)

train_cls_aug_df = pd.concat(
    [train_cls_original_df, augmented_malignant_df],
    ignore_index=True,
    sort=False,
)

before_counts = (
    train_cls_df['diagnosis']
    .value_counts()
    .reindex(['benign', 'malignant'])
    .fillna(0)
    .astype(int)
)

after_counts = (
    train_cls_aug_df['diagnosis']
    .value_counts()
    .reindex(['benign', 'malignant'])
    .fillna(0)
    .astype(int)
)

print('\nCLASSIFIER TRAINING DISTRIBUTION BEFORE TARGETED AUGMENTATION')
display(before_counts.to_frame('count'))

print('\nCLASSIFIER TRAINING DISTRIBUTION AFTER TARGETED AUGMENTATION')
display(after_counts.to_frame('count'))

print(
    f"\nTraining set changed from {len(train_cls_df)} to {len(train_cls_aug_df)} ROI samples."
)
print(
    f"Malignant: {before_counts['malignant']} -> {after_counts['malignant']} "
    f"| Benign: {before_counts['benign']} -> {after_counts['benign']}"
)
print('Validation and final test sets were NOT augmented.')

assert len(val_cls_df) == 40
assert len(test_cls_df) == 100
assert after_counts['benign'] == before_counts['benign']
assert after_counts['malignant'] == before_counts['malignant'] * 4

train_cls_aug_df.to_csv(
    RESULTS_DIR / 'targeted_augmented_training_manifest.csv',
    index=False
)

# Save a visual example for the presentation.
sample_malignant = malignant_train_df.iloc[0]
with Image.open(sample_malignant['roi_path']) as img:
    sample_roi = img.convert('RGB')
    sample_variants = [
        ('Original', sample_roi),
        ('90°', sample_roi.transpose(Image.Transpose.ROTATE_90)),
        ('180°', sample_roi.transpose(Image.Transpose.ROTATE_180)),
        ('270°', sample_roi.transpose(Image.Transpose.ROTATE_270)),
    ]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (title, image_variant) in zip(axes, sample_variants):
    ax.imshow(image_variant)
    ax.set_title(title)
    ax.axis('off')
plt.suptitle('Targeted malignant-only ROI augmentation')
plt.tight_layout()
AUGMENTATION_EXAMPLE_PATH = FIGURES_DIR / 'targeted_melanoma_rotation_example.png'
plt.savefig(AUGMENTATION_EXAMPLE_PATH, dpi=180, bbox_inches='tight')
plt.show()



# ============================================================
# 14. CLASSIFICATION tf.data + CLASS WEIGHTS
# EfficientNetB0 receives 0-255 float input (built-in rescaling)
# ============================================================
def decode_classification_pair(roi_path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(roi_path), channels=3)
    image = tf.image.resize(image, CLS_IMAGE_SIZE, method='bilinear')
    image = tf.cast(image, tf.float32)
    image.set_shape((*CLS_IMAGE_SIZE, 3))
    return image, tf.cast(label, tf.float32)


def make_cls_dataset(frame, training=False):
    ds = tf.data.Dataset.from_tensor_slices((
        frame['roi_path'].astype(str).values,
        frame['label'].astype(np.float32).values,
    ))
    if training:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_classification_pair, num_parallel_calls=AUTOTUNE)
    return ds.batch(CLS_BATCH_SIZE).prefetch(AUTOTUNE)


train_cls_ds = make_cls_dataset(train_cls_aug_df, training=True)
val_cls_ds = make_cls_dataset(val_cls_df, training=False)
test_cls_ds = make_cls_dataset(test_cls_df, training=False)

train_labels = train_cls_aug_df['label'].astype(int).values
classes = np.unique(train_labels)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
CLASS_WEIGHTS = {int(c): float(w) for c, w in zip(classes, weights)}
print('Class weights:', CLASS_WEIGHTS)

# ============================================================
# 15. EFFICIENTNETB0 CLASSIFIER
# Same overall classifier design as Oneli
# ============================================================
classification_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical', seed=SEED),
    layers.RandomRotation(0.05, seed=SEED),
    layers.RandomZoom(0.10, seed=SEED),
    layers.RandomContrast(0.10, seed=SEED),
], name='classification_augmentation')


def build_efficientnet_classifier():
    inputs = keras.Input(shape=(*CLS_IMAGE_SIZE, 3), name='roi_image')
    x = classification_augmentation(inputs)
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*CLS_IMAGE_SIZE, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.30, name='dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='prediction')(x)
    return keras.Model(inputs, outputs, name='Melanoma_EfficientNetB0')


def compile_classifier(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )


classifier = build_efficientnet_classifier()
compile_classifier(classifier, CLS_HEAD_LR)
assert classifier(tf.zeros((1, 224, 224, 3), dtype=tf.float32), training=False).shape == (1, 1)
print('EfficientNetB0 synthetic shape test passed.')

# Phase 1: frozen backbone
HEAD_MODEL_PATH = MODELS_DIR / 'efficientnet_b0_head_best.keras'
head_callbacks = [
    keras.callbacks.ModelCheckpoint(HEAD_MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]
head_fit = classifier.fit(
    train_cls_ds,
    validation_data=val_cls_ds,
    epochs=CLS_HEAD_EPOCHS,
    class_weight=CLASS_WEIGHTS,
    callbacks=head_callbacks,
    verbose=1,
)
head_history = head_fit.history

# Phase 2: fine-tune upper 20 layers, BatchNorm frozen
FINE_MODEL_PATH = MODELS_DIR / 'efficientnet_b0_finetuned_best.keras'
fine_history = None
if DO_FINE_TUNING:
    classifier = keras.models.load_model(HEAD_MODEL_PATH)
    backbone = classifier.get_layer('efficientnetb0')
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_classifier(classifier, CLS_FINE_TUNE_LR)

    fine_callbacks = [
        keras.callbacks.ModelCheckpoint(FINE_MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]
    fine_fit = classifier.fit(
        train_cls_ds,
        validation_data=val_cls_ds,
        epochs=CLS_FINE_TUNE_EPOCHS,
        class_weight=CLASS_WEIGHTS,
        callbacks=fine_callbacks,
        verbose=1,
    )
    fine_history = fine_fit.history

# ============================================================
# 16. SELECT CLASSIFIER USING VALIDATION ROC-AUC ONLY
# ============================================================
def predict_probabilities(model, dataset):
    return model.predict(dataset, verbose=0).reshape(-1)


val_true = val_cls_df['label'].astype(int).values
head_model = keras.models.load_model(HEAD_MODEL_PATH)
head_val_probs = predict_probabilities(head_model, val_cls_ds)
head_val_auc = roc_auc_score(val_true, head_val_probs)
candidates = [('Frozen EfficientNetB0', head_model, head_val_auc)]

if DO_FINE_TUNING and FINE_MODEL_PATH.exists():
    fine_model = keras.models.load_model(FINE_MODEL_PATH)
    fine_val_probs = predict_probabilities(fine_model, val_cls_ds)
    fine_val_auc = roc_auc_score(val_true, fine_val_probs)
    candidates.append(('Fine-tuned EfficientNetB0', fine_model, fine_val_auc))

classifier_comparison = pd.DataFrame([
    {'Model': name, 'Validation ROC-AUC': auc_val}
    for name, _, auc_val in candidates
])
print('\nClassifier validation comparison:')
display(classifier_comparison)

SELECTED_CLASSIFIER_NAME, final_classifier, _ = max(candidates, key=lambda item: item[2])
print('Selected classifier:', SELECTED_CLASSIFIER_NAME)
final_classifier.save(MODELS_DIR / 'efficientnet_b0_final.keras')

# ============================================================
# 17. CLASSIFIER TRAINING CURVES
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, title in [
    (axes[0], 'loss', 'EfficientNet Loss'),
    (axes[1], 'accuracy', 'EfficientNet Accuracy'),
    (axes[2], 'auc', 'EfficientNet ROC-AUC'),
]:
    head_epochs = range(1, len(head_history[metric]) + 1)
    ax.plot(head_epochs, head_history[metric], label='Head Train')
    ax.plot(head_epochs, head_history[f'val_{metric}'], '--', label='Head Validation')

    if fine_history is not None:
        fine_epochs = range(
            len(head_history[metric]) + 1,
            len(head_history[metric]) + len(fine_history[metric]) + 1,
        )
        ax.plot(fine_epochs, fine_history[metric], label='Fine-Tune Train')
        ax.plot(fine_epochs, fine_history[f'val_{metric}'], '--', label='Fine-Tune Validation')

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.25)
    ax.legend()

plt.tight_layout()
CLASSIFIER_CURVES_PATH = FIGURES_DIR / 'classifier_training_curves.png'
plt.savefig(CLASSIFIER_CURVES_PATH, dpi=180, bbox_inches='tight')
plt.show()

# ============================================================
# 18. FINAL CLASSIFICATION RESULTS
# ============================================================
test_true = test_cls_df['label'].astype(int).values
test_probs = predict_probabilities(final_classifier, test_cls_ds)
test_pred = (test_probs >= CLS_THRESHOLD).astype(int)
cm = confusion_matrix(test_true, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = float(accuracy_score(test_true, test_pred))
sensitivity = float(recall_score(test_true, test_pred, zero_division=0))
specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
precision = float(precision_score(test_true, test_pred, zero_division=0))
f1 = float(f1_score(test_true, test_pred, zero_division=0))
roc_auc = float(roc_auc_score(test_true, test_probs))
balanced_accuracy = float(balanced_accuracy_score(test_true, test_pred))

classification_results = pd.DataFrame({
    'Metric': [
        'Accuracy', 'Sensitivity', 'Specificity', 'Balanced Accuracy',
        'Test ROC-AUC', 'Precision', 'F1-score',
        'True Positive', 'True Negative', 'False Positive', 'False Negative',
    ],
    'Value': [
        accuracy, sensitivity, specificity, balanced_accuracy,
        roc_auc, precision, f1, tp, tn, fp, fn,
    ],
})

print('\n' + '=' * 70)
print('EFFICIENTNETB0 - FINAL TEST RESULTS')
print('=' * 70)
display(classification_results)
print(classification_report(
    test_true, test_pred,
    labels=[0, 1],
    target_names=['Benign', 'Melanoma'],
    zero_division=0,
))
classification_results.to_csv(RESULTS_DIR / 'FINAL_classifier_metrics.csv', index=False)

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
image_plot = ax.imshow(cm, cmap='Blues')
labels_matrix = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{labels_matrix[i][j]}\n{cm[i, j]}', ha='center', va='center', fontsize=16)
ax.set_xticks([0, 1], ['Predicted Benign', 'Predicted Melanoma'])
ax.set_yticks([0, 1], ['Actual Benign', 'Actual Melanoma'])
ax.set_xlabel('Model Prediction')
ax.set_ylabel('Ground Truth')
ax.set_title('Final Test Confusion Matrix (n=100)')
plt.colorbar(image_plot, ax=ax)
plt.tight_layout()
CONFUSION_PATH = FIGURES_DIR / 'classification_confusion_matrix.png'
plt.savefig(CONFUSION_PATH, dpi=180, bbox_inches='tight')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(test_true, test_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'EfficientNetB0 AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate / Sensitivity')
plt.title('Final Test ROC Curve')
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
ROC_PATH = FIGURES_DIR / 'classification_roc_curve.png'
plt.savefig(ROC_PATH, dpi=180, bbox_inches='tight')
plt.show()

# Save per-case classifier predictions
test_predictions_df = test_cls_df.copy()
test_predictions_df['malignancy_probability'] = test_probs
test_predictions_df['predicted_label'] = test_pred
test_predictions_df['predicted_diagnosis'] = np.where(test_pred == 1, 'malignant', 'benign')
test_predictions_df['absolute_probability_error'] = np.abs(test_true - test_probs)
conditions = [
    (test_true == 0) & (test_pred == 0),
    (test_true == 0) & (test_pred == 1),
    (test_true == 1) & (test_pred == 0),
    (test_true == 1) & (test_pred == 1),
]
test_predictions_df['outcome'] = np.select(conditions, ['TN', 'FP', 'FN', 'TP'], default='?')
test_predictions_df.to_csv(RESULTS_DIR / 'test_predictions.csv', index=False)

# ============================================================
# 19. GRAD-CAM
# ============================================================
def make_gradcam_heatmap(model, image_batch):
    backbone = model.get_layer('efficientnetb0')
    gap_layer = model.get_layer('gap')
    dropout_layer = model.get_layer('dropout')
    prediction_layer = model.get_layer('prediction')

    with tf.GradientTape() as tape:
        feature_maps = backbone(image_batch, training=False)
        tape.watch(feature_maps)
        x = gap_layer(feature_maps)
        x = dropout_layer(x, training=False)
        prediction = prediction_layer(x)

    gradients = tape.gradient(prediction, feature_maps)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))[0]
    feature_maps = feature_maps[0]
    heatmap = tf.reduce_sum(feature_maps * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def prepare_roi_for_classifier(roi_pil):
    resized = roi_pil.convert('RGB').resize(
        (CLS_IMAGE_SIZE[1], CLS_IMAGE_SIZE[0]), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32)
    return resized, array[None, ...]


def heatmap_overlay(image_pil, heatmap, alpha=0.40):
    image = image_pil.convert('RGB')
    heatmap_resized = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    ).resize(image.size, Image.Resampling.BILINEAR)
    heat = np.asarray(heatmap_resized, dtype=np.float32) / 255.0
    base = np.asarray(image, dtype=np.float32) / 255.0
    coloured = plt.get_cmap('jet')(heat)[..., :3]
    overlay = np.clip((1 - alpha) * base + alpha * coloured, 0, 1)
    return Image.fromarray(np.uint8(overlay * 255))


# Best / median / worst classification cases by probability error
sorted_cls = test_predictions_df.sort_values('absolute_probability_error').reset_index(drop=True)
gradcam_cases = [
    ('BEST', sorted_cls.iloc[0]),
    ('MEDIAN', sorted_cls.iloc[len(sorted_cls) // 2]),
    ('WORST', sorted_cls.iloc[-1]),
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for r, (case_name, row) in enumerate(gradcam_cases):
    original = load_rgb(row['image_path'])
    roi_pil = load_rgb(row['roi_path'])
    roi_resized, roi_batch = prepare_roi_for_classifier(roi_pil)
    heatmap = make_gradcam_heatmap(final_classifier, tf.convert_to_tensor(roi_batch))
    overlay = heatmap_overlay(roi_resized, heatmap)

    axes[r, 0].imshow(original)
    axes[r, 0].set_title(f"{case_name} - Original\nTrue: {row['diagnosis']}")
    axes[r, 1].imshow(roi_resized)
    axes[r, 1].set_title(f"Lesion ROI\nPredicted: {row['predicted_diagnosis']}")
    axes[r, 2].imshow(overlay)
    axes[r, 2].set_title(f"Grad-CAM\nP(malignant)={row['malignancy_probability']:.3f}")
    for ax in axes[r]:
        ax.axis('off')

plt.tight_layout()
GRADCAM_PATH = FIGURES_DIR / 'gradcam_best_median_worst.png'
plt.savefig(GRADCAM_PATH, dpi=180, bbox_inches='tight')
plt.show()

# ============================================================
# 20. FINAL RESULTS DASHBOARD
# ============================================================
print('\n\n' + '=' * 78)
print('FINAL RESULTS - COPY THESE INTO THE PRESENTATION')
print('=' * 78)

print('\nTARGETED CLASSIFIER AUGMENTATION')
print('-' * 40)
print(f"Benign training ROIs:    {int(after_counts['benign'])}")
print(f"Malignant training ROIs: {int(after_counts['malignant'])}")
print('Malignant transforms:   90°, 180°, 270° rotations')
print('Validation/test:         untouched')

print('\nRESUNET SEGMENTATION')
print('-' * 40)
print(f'Test Dice:             {TEST_DICE:.4f}')
print(f'Test Jaccard / IoU:    {TEST_JACCARD:.4f}')
print(f'Test Loss:             {TEST_LOSS:.4f}')
print(f'Best Test Dice:        {BEST_TEST_DICE:.4f}')
print(f'Median Test Dice:      {MEDIAN_TEST_DICE:.4f}')
print(f'Worst Test Dice:       {WORST_TEST_DICE:.4f}')
print(f'Best Validation Dice:  {BEST_VALIDATION_DICE:.4f}')
print(f'Parameters:            {RESUNET_PARAMETERS:,}')
print(f'Pixel Sensitivity:     {pixel_sensitivity:.4f}')
print(f'Pixel Specificity:     {pixel_specificity:.4f}')

print('\nEFFICIENTNETB0 CLASSIFICATION')
print('-' * 40)
print(f'Accuracy:              {accuracy:.4f}')
print(f'Sensitivity:           {sensitivity:.4f}')
print(f'Specificity:           {specificity:.4f}')
print(f'Test ROC-AUC:          {roc_auc:.4f}')
print(f'Precision:             {precision:.4f}')
print(f'F1-score:              {f1:.4f}')
print(f'True Positive (TP):    {tp}')
print(f'True Negative (TN):    {tn}')
print(f'False Positive (FP):   {fp}')
print(f'False Negative (FN):   {fn}')
print('Selected classifier:', SELECTED_CLASSIFIER_NAME)

print('\nFINAL RESUNET TABLE')
display(resunet_final_metrics)
print('\nFINAL CLASSIFIER TABLE')
display(classification_results)

# Re-display presentation figures
for title, path in [
    ('0. TARGETED MELANOMA AUGMENTATION EXAMPLE', AUGMENTATION_EXAMPLE_PATH),
    ('1. RESUNET TRAINING CURVES', RESUNET_CURVES_PATH),
    ('2. BEST / MEDIAN / WORST RESUNET RESULTS', SEG_EXAMPLES_PATH),
    ('3. EFFICIENTNET TRAINING CURVES', CLASSIFIER_CURVES_PATH),
    ('4. CONFUSION MATRIX', CONFUSION_PATH),
    ('5. ROC CURVE', ROC_PATH),
    ('6. BEST / MEDIAN / WORST GRAD-CAM', GRADCAM_PATH),
]:
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)
    with Image.open(path) as final_figure:
        display(final_figure.copy())

print('\nPIPELINE COMPLETE.')
print('Results saved to:', PROJECT_DIR)

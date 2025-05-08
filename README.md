# End-to-End 3D CAD Reconstruction

## 1. Installing the Repository
```
git clone <this repository>
cd treecad-280
conda create -n treecad python=3.11 -y
conda activate treecad
pip install cadquery # to manually install the dependencies of it
pip uninstall cadquery -y
git clone git@github.com:CadQuery/cadquery.git
cd cadquery
pip install -e .
cd ..
pip install -r requirements.txt
```

**All steps following here may require changing the dataset address in the script, since this is different per user.
## 2. Generating the Dataset
```
// Do this clone at a place you like, since VSCode tracks dataset files in Git and causes large lags.
git clone https://huggingface.co/datasets/filapro/cad-recode-v1.5

// The following step renders some images for each existing cadquery script within a range.
DISPLAY=:0 python ./src/generate_dataset.py
```

## 3. Training
The training config is at `./src/config/default_config.yaml`
```
DISPLAY=:0 python ./src/naive_autoregressive/train_autoregressive.py
```

## 4. Model Inference
The model inference function is currently written as a `model_inference(model, renders, roll, elevations, masks, tokenizer)` function inside `train_autoregressive.py` as a standalone piece, and can run conventionally with the evaluation framework. Vint wrote `src/naive_autoregressive/test_inference.ipynb` but this is an incompelte script for inference, since the model checkpoints take too long to load from `torch.load`. This is another point of investigation.
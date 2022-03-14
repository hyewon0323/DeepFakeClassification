# 2021 Real/Fake image classification

## Challenge Details
* [Kaggle Challenge Page](https://www.kaggle.com/c/deepfake-statml2021s-smwu/)
* In-class Competition of Sookmyung Women's University
* Apr 27, 2021 - Jun 10, 2021

## Team
* Team Name : Group 2 (✨Rank 1✨)
* Members : [Gahee Kim](https://github.com/GaHeeKim), [Hyewon Ryu](https://github.com/hyewon0323), [Haram Lee](https://github.com/hrxorxm)

## Solution Description
* Source : [solution.ipynb](https://github.com/hrxorxm/DeepFakeClassification/blob/main/solution.ipynb)
  * Reference
    * [Competition baseline code](https://www.kaggle.com/hy2rim/cnnmodel)

* Augmentations
  * Reference
    * [DeepFake Detection (DFDC) Solution by @selimsef](https://github.com/selimsef/dfdc_deepfake_challenge)
    * [Albumentations documentation](https://albumentations.ai/docs/)
  * Code
    ```python
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transforms_tr = A.Compose([
        A.Resize(256, 256),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.05),
        A.HorizontalFlip(),
        A.OneOf([
            A.RandomBrightnessContrast(), 
            A.FancyPCA(), 
            A.HueSaturationValue(),
        ], p=0.7),
        A.ToGray(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.CenterCrop(156, 156),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    ```
    * `Resize(256, 256)` : Required to resize input image consistently
    * `ImageCompression` ~ `ShiftScaleRotate` : Added some modifications to the code in the reference
    * `CenterCrop(156, 156)` : Features of fake images appear mainly in the center of the face. Most of the datasets have faces in the center front. Therefore, only the center part of the image is extracted.

* Model
  * Reference
    * [Facenet-pytorch](https://github.com/timesler/facenet-pytorch)
  * Code
    ```python
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained=None, classify=True, num_classes=2, dropout_prob=0.6)
    ```

* Optimizer and Scheduler
  * Reference
    * [Pytorch-optimizer](https://github.com/jettify/pytorch-optimizer)
  * Code
    ```python
    import torch_optimizer as optim
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=T_max)
    ```

* K-fold and Soft voting
  * Reference
    * [scikit-learn K-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
  * Code
    ```python
    # Training
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    best_models = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_df)):
      ...
      for e in range(num_epochs) :
        ...
        best_models[fold] = best_model
    
    # Inference
    scores_result = []
    for fold in range(n_splits):
      model = best_models[fold]
      ...
      scores_result.append(score_list)
    
    # Soft voting
    import torch.nn.functional as F
    myresult = F.softmax(torch.tensor(scores_result), dim=2)
    myresult = torch.sum(myresult, dim=0)
    _, preds = myresult.max(dim=1)
    ```

## Contributors
<div>
<a href="https://github.com/hrxorxm">
  <img src="https://github.com/hrxorxm.png" width="50" height="50" >
</a>
<a href="https://github.com/GaHeeKim">
  <img src="https://github.com/GaHeeKim.png" width="50" height="50" >
</a>
<a href="https://github.com/hyewon0323">
  <img src="https://github.com/hyewon0323.png" width="50" height="50" >
</a>
</div>

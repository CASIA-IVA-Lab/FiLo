import torch
import argparse
import os
import numpy as np
import torch.nn.functional as F
from datasets.mvtec_supervised import MVTecDataset
from datasets.visa_supervised import VisaDataset
import models.vv_open_clip as open_clip
import torchvision.transforms as transforms
from utils.loss import FocalLoss, BinaryDiceLoss
from models.FiLo import FiLo
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

mvtec_obj_list = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

visa_obj_list = [
    "candle",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "capsules",
]

positions_list = ['top left', 'top', 'top right', 'left', 'center', 'right', 'bottom left', 'bottom', 'bottom right']


if __name__ == "__main__":

    parser = argparse.ArgumentParser("FiLo Train", add_help=True)
    parser.add_argument(
        "--clip_model", type=str, default="ViT-L-14-336", help="model used"
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="openai",
        help="pretrained weight used",
    )
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="features used",
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./data/visa",
        help="train dataset path",
    )
    parser.add_argument("--image_size", type=int, default=518, help="image size")

    parser.add_argument(
        "--dataset", type=str, default="visa", help="train dataset name"
    )
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--decoder_learning_rate", type=float, default=0.0001, help="learning rate for decoder"
    )
    parser.add_argument(
        "--adapter_learning_rate", type=float, default=0.00001, help="learning rate for adapter"
    )
    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--n_ctx", type=int, default=12, help="epochs")
    parser.add_argument("--adapter_epoch", type=int, default=5, help="epochs")

    parser.add_argument(
        "--save_path",
        type=str,
        default="./ckpt",
        help="path to save results",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="running on cpu only!, default=False"
    )
    args = parser.parse_args()


    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = args.device
    image_size = args.image_size
    batch_size = args.batch_size
    epochs = args.epoch
    dataset_name = args.dataset

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    _, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, image_size, pretrained=args.clip_pretrained
    )

    if dataset_name == "visa":
        train_data = VisaDataset(
            root=args.train_data_path,
            transform=preprocess,
            target_transform=transform,
        )
    else:
        train_data = MVTecDataset(
            root=args.train_data_path,
            transform=preprocess,
            target_transform=transform,
            aug_rate=args.aug_rate,
        )

    train_dataloader = DataLoaderX(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8
    )
    
    obj_list = [x.replace("_", " ") for x in train_data.get_cls_names()]

    filo_model = FiLo(obj_list, args, device).to(device)
    
    main_part_param_groups = [
        {'params': filo_model.decoder_cov.parameters(), 'lr': args.decoder_learning_rate},
        {'params': filo_model.decoder_linear.parameters(), 'lr': args.decoder_learning_rate},
        {'params': filo_model.normal_prompt_learner.parameters(), 'lr': args.learning_rate},
        {'params': filo_model.abnormal_prompt_learner.parameters(), 'lr': args.learning_rate}
    ]

    optimizer_main_part = torch.optim.AdamW(
        main_part_param_groups,
        betas=(0.5, 0.999),
    )

    adapter_param_groups = [
        {'params': filo_model.adapter.parameters(), 'lr': args.adapter_learning_rate},
    ]

    optimizer_adapter = torch.optim.AdamW(
        adapter_param_groups,
        betas=(0.5, 0.999),
    )

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    with torch.no_grad():
        obj_list = [x.replace("_", " ") for x in train_data.get_cls_names()]


    for epoch in range(epochs):
        loss_list = []
        for items in tqdm(train_dataloader):
            image = items["img"].to(device)
            cls_name = items["cls_name"][0]
            image_path = items["img_path"]
            anomaly_cls = items["anomaly_class"][0]
            label = items['anomaly'].to(device)
            text_probs, anomaly_maps = filo_model(items, with_adapter=False)

            # losses
            gt = items["img_mask"].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
            loss = 0
            for num in range(len(anomaly_maps)):
                loss += loss_focal(anomaly_maps[num], gt)
                loss += loss_dice(anomaly_maps[num][:, 1, :, :], gt)
                loss += loss_dice(anomaly_maps[num][:, 0, :, :], 1 - gt)

            optimizer_main_part.zero_grad()
            loss.backward()
            optimizer_main_part.step()

            loss_list.append(loss.item())
        # logs
        if (epoch + 1) % 1 == 0:
            print(
                "epoch [{}/{}], loss:{:.4f}".format(
                    epoch + 1, epochs, np.mean(loss_list)
                )
            )

    for epoch in range(args.adapter_epoch):
        loss_list = []
        for items in tqdm(train_dataloader):
            image = items["img"].to(device)
            cls_name = items["cls_name"][0]
            image_path = items["img_path"]
            anomaly_cls = items["anomaly_class"][0]
            label = items['anomaly'][0].to(device)
            text_probs, anomaly_maps = filo_model(items, only_train_adapter=True, with_adapter=True)

            # losses
            text_probs = text_probs[:, 0, ...] / 0.07
            loss = F.cross_entropy(text_probs.squeeze(), label)
            loss_list.append(loss.item())

            optimizer_adapter.zero_grad()
            loss.backward()
            optimizer_adapter.step()

            loss_list.append(loss.item())

        # logs
        print(
            "adapter epoch [{}/{}], loss:{:.4f}".format(
                epoch + 1, args.adapter_epoch, np.mean(loss_list)
            )
        )

        # save mode
        save_name =  + "filo_train_on_" + args.dataset
        ckp_path = os.path.join(
            save_path,
            f"{save_name}.pth",
        )
        torch.save(
            {
                "filo": filo_model.state_dict(),
            },
            ckp_path,
        )

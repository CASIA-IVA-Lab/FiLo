import torch
import torch.nn as nn
import os
from typing import Union, List, OrderedDict
import torch
import numpy as np
from torch.nn import functional as F
from . import vv_open_clip as open_clip
# from utils.loss import FocalLoss, BinaryDiceLoss
import cv2
from matplotlib import pyplot as plt

status_normal = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]

status_abnormal_winclip = [
    "damaged {}",
    "broken {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

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


cls_map = {}
for c_name in visa_obj_list:
    cls_map[c_name] = c_name

for c_name in mvtec_obj_list:
    cls_map[c_name] = c_name

cls_map["pcb1"] = "pcb"
cls_map["pcb2"] = "pcb"
cls_map["pcb3"] = "pcb"
cls_map["pcb4"] = "pcb"
cls_map["macaroni1"] = "macaroni"
cls_map["macaroni2"] = "macaroni"

detail_status_abnormal = {}

mvtec_anomaly_detail_gpt = {
    "carpet": "discoloration in a specific area,irregular patch or section with a different texture,frayed edges or unraveling fibers,burn mark or scorching",
    "grid": "crooked,cracks,excessive gaps,discoloration,deformation,missing,inconsistent spacing between grid elements,corrosion,visible signs,chipping",
    "leather": "scratches,discoloration,creases,uneven texture,tears,brittleness,damage,seams,heat damage,mold",
    "tile": "chipped,irregularities,discoloration,efflorescence,warping,missing,depressions,lippage,fungus,damage",
    "wood": "knots,warping,cracks along the grain,mold growth on the surface,staining from water damage,wood rot,woodworm holes,rough patches,protruding knots",
    "bottle": "cracked large,cracked small,dented large,dented small,leaking,discolored,deformed,missing cap,excessive condensation,unusual odor",
    "cable": "twisted,knotted cable strands,detached connectors,excessive stretching,dents,corrosion,scorching along the cable,exposed conductive material",
    "capsule": "irregular shape,discoloration coloring,crinkled,uneven seam,condensation inside the capsule,foreign particles,unusually soft or hard",
    "hazelnut": "fungal growth,Unusual discoloration,rotten or foul odor emanating,insect infestation,wetness,misshapen shell,unusually thin,contaminants,unusual texture",
    "metal nut": "cracks,irregular threading,corrosion,missing,distortion,signs of discoloration,excessive wear on contact surfaces,inconsistent texture",
    "pill": "irregular shape,crumbling texture,excessive powder,Uneven coating,presence of air bubbles,disintegration,abnormal specks",
    "screw": "rust on the surface,bent,damaged threads,stripped threads,deformed top,coating damage,uneven grooves,inconsistent size",
    "toothbrush": "loose bristles,uneven bristle distribution,excessive shedding of bristles,staining on the bristles,abrasive texture,irregularities in the shape",
    "transistor": "burn marks,detached leads,signs of corrosion,irregularities in the shape,presence of cracks or fractures,signs of physical trauma,irregularities in the surface texture",
    "zipper": "bent,frayed,misaligned,excessive stiffness,corroded,detaches,loose,warped",
}

visa_anomaly_detail_gpt = {
    "candle": "cracks or fissures in the wax,Wax pooling unevenly around the wick,tunneling,incomplete wax melt pool,irregular or flickering flame,other,extra wax in candle,wax melded out of the candle",
    "capsules": "uneven capsule size,capsule shell appears brittle,excessively soft,dents,condensation,irregular seams or joints,specks",
    "cashew": "uneven coloring,fungal growth,presence of foreign objects,unusual texture,empty shells,signs of moisture,stuck together",
    "chewinggum": "consistency,presence of foreign objects,uneven coloring,excessive hardness,similar colour spot",
    "fryum": "irregular shape,unusual odor,uneven coloring,unusual texture,small scratches,different colour spot,fryum stuck together,other",
    "macaroni1": "uneven shape ,small scratches,small cracks,uneven coloring,signs of insect infestation,uneven texture,Unusual consistency",
    "macaroni2": "irregular shape,small scratches,presence of foreign particles,excessive moisture,Signs of infestation,small cracks,unusual texture",
    "pcb1": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb2": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb3": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pcb4": "oxidation on the copper traces,separation of layers,presence of solder bridges,excessive solder residue,discoloration,Uneven solder joints,bowing of the board,missing vias",
    "pipe fryum": "uneven shape,presence of foreign objects,different colour spot,unusual odor,empty interior,unusual texture,similar colour spot,stuck together",
}

for cls_name in mvtec_anomaly_detail_gpt.keys():
    mvtec_anomaly_detail_gpt[cls_name] = (
        mvtec_anomaly_detail_gpt[cls_name].split(",")
    )

for cls_name in visa_anomaly_detail_gpt.keys():
    visa_anomaly_detail_gpt[cls_name] = (
        visa_anomaly_detail_gpt[cls_name].split(",")
    )

status_abnormal = {}

for cls_name in mvtec_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in mvtec_anomaly_detail_gpt[cls_name]] + status_abnormal_winclip

for cls_name in visa_anomaly_detail_gpt.keys():
    status_abnormal[cls_name] = ['abnormal {} ' + 'with {}'.format(x) for x in visa_anomaly_detail_gpt[cls_name]] + status_abnormal_winclip


positions_list = [
    "top left",
    "top",
    "top right",
    "left",
    "center",
    "right",
    "bottom left",
    "bottom",
    "bottom right",
]

location_map = {"top left": [(0, 0),(172, 172)],
                "top": [(173, 0),(344, 172)],
                "top right": [(345, 0), (517, 172)],
                "left": [(0, 173), (172, 344)],
                "center": [(173, 173), (344, 344)],
                "right": [(345, 173), (517, 344)],
                "bottom left": [(0, 345), (172, 517)],
                "bottom": [(173, 345), (344, 517)],
                "bottom right": [(345, 345), (517, 517)]}


class PromptLearner_normal(nn.Module):
    def __init__(self, classnames, status, clip_model, tokenizer, dim, n_ctx, device):
        super().__init__()
        vis_dim = dim
        ctx_dim = dim

        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        classnames = [name.replace("_", " ") for name in classnames]

        self.tokenized_prompts = {}
        embedding = {}

        self.token_prefix = {}
        self.token_suffix = {}

        for class_name in classnames:

            class_name = cls_map[class_name]

            p = [
                prompt + " " + status_i.format(class_name) + "." for status_i in status
            ]

            self.tokenized_prompts[class_name] = tokenizer(p)
            self.tokenized_prompts[class_name].requires_grad = False

            with torch.no_grad():
                embedding[class_name] = clip_model.token_embedding(
                    self.tokenized_prompts[class_name]
                )

            self.token_prefix[class_name] = embedding[class_name][:, :1, :].to(device)
            self.token_prefix[class_name].requires_grad = False

            self.token_suffix[class_name] = embedding[class_name][:, 1 + n_ctx :, :].to(device)
            self.token_suffix[class_name].requires_grad = False

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, class_name):

        ctx = self.ctx
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        prefix = self.token_prefix[class_name]
        suffix = self.token_suffix[class_name]

        n_cls = prefix.shape[0]

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)

        return prompts


class PromptLearner_abnormal(nn.Module):
    def __init__(
        self,
        classnames,
        status,
        clip_model,
        tokenizer,
        dim,
        n_ctx,
        device,
        positions=None,
    ):
        super().__init__()
        vis_dim = dim
        ctx_dim = dim

        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim)

        nn.init.normal_(ctx_vectors, std=0.02)
        prompt = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )
        if positions == None:
            self.positions = [
                "top left",
                "top",
                "top right",
                "left",
                "center",
                "right",
                "bottom left",
                "bottom",
                "bottom right",
            ]
        else:
            self.positions = positions

        classnames = [name.replace("_", " ") for name in classnames]

        self.tokenized_prompts = {}
        embedding = {}

        self.token_prefix = {}
        self.token_suffix = {}

        for origin_class_name in classnames:

            class_name = cls_map[origin_class_name]

            p = [
                prompt + " " + status_i.format(class_name) + " at " + position + "."
                for status_i in status[origin_class_name]
                for position in self.positions
            ]

            # print(p)

            self.tokenized_prompts[class_name] = tokenizer(p)
            with torch.no_grad():
                embedding[class_name] = clip_model.token_embedding(
                    self.tokenized_prompts[class_name]
                )

            self.token_prefix[class_name] = embedding[class_name][:, :1, :].to(device)
            self.token_prefix[class_name].requires_grad = False

            self.token_suffix[class_name] = embedding[class_name][:, 1 + n_ctx :, :].to(device)
            self.token_suffix[class_name].requires_grad = False

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features, class_name):

        ctx = self.ctx
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        prefix = self.token_prefix[class_name]
        suffix = self.token_suffix[class_name]

        n_cls = prefix.shape[0]

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(
                ctx_i, prefix, suffix
            )  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)

        prompts = torch.stack(prompts)
        return prompts


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                assert 0 == 1  # error
        return tokens

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=2):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.SiLU(),
        )

    def forward(self, x):
        y = self.fc(x) + x
        return y


class CovLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(CovLayer, self).__init__()
        self.fc_33 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_11 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_77 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_55 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same")
                for _ in range(k)
            ]
        )
        self.fc_51 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(5, 1), padding="same")
                for _ in range(k)
            ]
        )
        self.fc_15 = nn.ModuleList(
            [
                nn.Conv2d(dim_in, dim_out, kernel_size=(1, 5), padding="same")
                for _ in range(k)
            ]
        )

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                x = tokens[i][:, 1:, :]
                x = x.view(
                    x.shape[0],
                    int(np.sqrt(x.shape[1])),
                    int(np.sqrt(x.shape[1])),
                    x.shape[2],
                )
                # print(x.shape)
                x_temp = (
                    self.fc_11[i](x.permute(0, 3, 1, 2))
                    + self.fc_33[i](x.permute(0, 3, 1, 2))
                    + self.fc_55[i](x.permute(0, 3, 1, 2))
                    + self.fc_77[i](x.permute(0, 3, 1, 2))
                    + self.fc_15[i](x.permute(0, 3, 1, 2))
                    + self.fc_51[i](x.permute(0, 3, 1, 2))
                )
                tokens[i] = x_temp
                tokens[i] = (
                    tokens[i]
                    .permute(0, 2, 3, 1)
                    .view(tokens[i].shape[0], -1, tokens[i].shape[1])
                )
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](
                    tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous()
                )
        return tokens


class FiLo(nn.Module):
    def __init__(self, obj_list, args, device) -> None:
        super().__init__()

        self.args = args

        self.device = device

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            args.clip_model, args.image_size, pretrained=args.clip_pretrained
        )
        self.clip_model.eval()

        self.tokenizer = open_clip.get_tokenizer(args.clip_model)

        self.decoder_cov = CovLayer(1024, 768, 3)
        self.decoder_linear =  LinearLayer(1024, 768, 4)
        self.text_encoder = TextEncoder(self.clip_model)
        self.text_encoder.eval()

        self.normal_prompt_learner = PromptLearner_normal(
            obj_list,
            status_normal,
            self.clip_model,
            self.tokenizer,
            768,
            args.n_ctx,
            args.device
        )

        self.abnormal_prompt_learner = PromptLearner_abnormal(
            obj_list,
            status_abnormal,
            self.clip_model,
            self.tokenizer,
            768,
            args.n_ctx,
            args.device
        )

        self.adapter = Adapter(768)

    def forward(self, items, with_adapter=False, only_train_adapter=False, positions=None):
        image = items["img"].to(self.device)
        cls_name = items["cls_name"][0]


        with torch.no_grad():
            image_features, patch_tokens = self.clip_model.encode_image(
                image, self.args.features_list
            )


        if with_adapter:
            image_features = self.adapter(image_features)
        image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)



        if only_train_adapter:
            with torch.no_grad():
                normal_prompts = self.normal_prompt_learner(image_features, cls_map[cls_name])
                normal_tokenized_prompts = self.normal_prompt_learner.tokenized_prompts[
                    cls_map[cls_name]
                ]

                abnormal_prompts = self.abnormal_prompt_learner(image_features, cls_map[cls_name])
                abnormal_tokenized_prompts = self.abnormal_prompt_learner.tokenized_prompts[
                    cls_map[cls_name]
                ]

                normal_text_features = self.text_encoder(
                    normal_prompts[0], normal_tokenized_prompts
                )
                normal_text_features = normal_text_features / normal_text_features.norm(
                    dim=-1, keepdim=True
                )

                abnormal_text_features = self.text_encoder(
                    abnormal_prompts[0], abnormal_tokenized_prompts
                )
                abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(
                    dim=-1, keepdim=True
                )

                normal_text_features = normal_text_features.mean(dim=0, keepdim=True)
                normal_text_features = normal_text_features / normal_text_features.norm()
                normal_text_features = normal_text_features.unsqueeze(1)

        else:

            normal_prompts = self.normal_prompt_learner(image_features, cls_map[cls_name])
            normal_tokenized_prompts = self.normal_prompt_learner.tokenized_prompts[
                cls_map[cls_name]
            ]

            abnormal_prompts = self.abnormal_prompt_learner(image_features, cls_map[cls_name])
            abnormal_tokenized_prompts = self.abnormal_prompt_learner.tokenized_prompts[
                cls_map[cls_name]
            ]

            normal_text_features = self.text_encoder(
                normal_prompts[0], normal_tokenized_prompts
            )
            normal_text_features = normal_text_features / normal_text_features.norm(
                dim=-1, keepdim=True
            )

            abnormal_text_features = self.text_encoder(
                abnormal_prompts[0], abnormal_tokenized_prompts
            )
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(
                dim=-1, keepdim=True
            )

            normal_text_features = normal_text_features.mean(dim=0, keepdim=True)
            normal_text_features = normal_text_features / normal_text_features.norm()
            normal_text_features = normal_text_features.unsqueeze(1)


        ab_position = []
        if positions != None:
            ab_position = positions


        if len(ab_position) > 0:
            tmp_abnormal_text_features = []
            for ab_p in ab_position:
                position_idx = positions_list.index(ab_p)
                tmp_abnormal_text_features.append(abnormal_text_features[position_idx::9])
        
            abnormal_text_features = torch.cat(tmp_abnormal_text_features,dim=0)
        

        abnormal_text_features = abnormal_text_features.mean(dim=0, keepdim=True)
        abnormal_text_features = abnormal_text_features / abnormal_text_features.norm()
        abnormal_text_features = abnormal_text_features.unsqueeze(1)

        text_features = torch.cat([normal_text_features, abnormal_text_features], dim=1)

        text_features = text_features / text_features.norm()

        text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)

        anomaly_maps = None

        if not only_train_adapter:

            patch_tokens_qkv = self.decoder_linear(patch_tokens[::2])
            patch_tokens_vv = self.decoder_cov(patch_tokens[1::2])

            anomaly_maps = []
            for layer in range(len(patch_tokens_qkv)):

                patch_tokens_qkv[layer] = patch_tokens_qkv[layer] / patch_tokens_qkv[
                    layer
                ].norm(dim=-1, keepdim=True)

                anomaly_map = (
                    100.0 * patch_tokens_qkv[layer] @ text_features.transpose(-2, -1)
                )

                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                    size=self.args.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

            for layer in range(len(patch_tokens_vv)):
                patch_tokens_vv[layer] = patch_tokens_vv[layer] / patch_tokens_vv[
                    layer
                ].norm(dim=-1, keepdim=True)

                anomaly_map = (
                    100.0 * patch_tokens_vv[layer] @ text_features.transpose(-2, -1)
                )

                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                    size=self.args.image_size,
                    mode="bilinear",
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

        return text_probs, anomaly_maps

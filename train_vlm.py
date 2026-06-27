import os
import json
import torch
import autoroot
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from typing import List, Tuple, Dict
from ssr.models.midi import MIDI
from ssr.models.vlm import SSRVLM
from accelerate import Accelerator
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from peft import LoraConfig, get_peft_model
from ssr.utils.prompt import SSRSpecialToken
from ssr.data.ssr_cot import SSRCoTDataset4VLM
from argparse import ArgumentParser, Namespace
from ssr.utils.misc import quiet, str_datetime, count_params, freeze_module
from transformers import AutoTokenizer, Qwen2_5_VLProcessor, CLIPProcessor, CLIPVisionModel, SiglipProcessor, SiglipVisionModel, get_cosine_schedule_with_warmup

from ssr.data.PointCloudTokenizer_new import SonataPointEncoder
from ssr.data.ssr_cot import get_point_embeds

import gc  # 垃圾回收
# 引入模型层定义，用于配置FSDP的模型参数分片策略（让fsdp知道分片处理的最小单元(一个一个分片)，而不是把整个模型直接做分片）
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools
# 引入 Qwen2 的层定义 (对应 midi)
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
# 引入 Qwen2.5-VL 的层定义 (对应 vlm)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.sep, "/data/dsq", "ScanNet/raw_select"))  # "ScanNet/raw_select"
    # ========== 新增：第二个数据集与混合策略 ==========
    parser.add_argument("--data_dir2", type=str, default="/data3/lxp/argoverse2/val")
    parser.add_argument("--jsonl1", type=str, default="scannet_reasoning_dataset_5w.jsonl")  # scannet数据集用的jsonl
    parser.add_argument("--jsonl2", type=str, default="argoverse2_reasoning_dataset_correct_5w.jsonl")  # av2数据集用的jsonl
    # mix_strategy: "concat" 按样本数比例训练；"balanced" 让两个数据集采样更均衡
    parser.add_argument("--mix_strategy", type=str, default="concat", choices=["concat", "balanced"])
    # ========== 新增结束 ==========
    parser.add_argument("--mamba", type=str, default=None)
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.sep, "/data/dsq/Models", "clip-vit-large-patch14-336"))
    parser.add_argument("--pretrained_midi", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-Reasoning", "2ds_image-point-fusion", "fusion1_cat_2qformer_newencoder_prompt_1.5b"))
    parser.add_argument("--pretrained_vlm", type=str, default=os.path.join(os.sep, "/data/dsq/Models", "Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--image_size", type=Tuple[int, int], default=(256, 256))
    parser.add_argument("--n_tor", type=int, default=10)
    parser.add_argument("--max_length", type=Tuple[int, int, int], default=(256, 1024, 256))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "checkpoints", "SSR-VLM", "2ds_image-point-fusion", "fusion1_cat_2qformer_newencoder_prompt_1.5b"))
    parser.add_argument("--llava", action="store_true")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def train(
    midi: nn.Module
    , vlm: nn.Module
    , dataloader: DataLoader
    , optimizer: torch.optim.Optimizer
    , scheduler: torch.optim.lr_scheduler.LRScheduler
    , accelerator: Accelerator
    , tor_token_id: Tuple[int, int]
    , epochs: int
    , clip_model: CLIPVisionModel      # <<< 新增
    , point_models: Dict[int, nn.Module]  # <<< 修改：传入编码器字典
) -> List[float]:
    losses = []
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"{str_datetime()} [Epoch {epoch + 1}/{epochs}]", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            device = accelerator.device
            # 1) CLIP 特征
            clip_pixel_values = batch.pop("clip_pixel_values").to(device)  # [B,3,Hc,Wc]
            with torch.no_grad():
                image_embeds = clip_model(pixel_values=clip_pixel_values).last_hidden_state  # [B,Li,Di]
            # 2) 点云特征
            # 先取出 dataset_type，用于选择对应的编码器
            dataset_type = batch.pop("dataset_type")  # [B]，在 CPU 上
            points_list = batch.pop("points_b1xn6")
            point_embeds_list = []
            for i, pts in enumerate(points_list):
                # 获取当前样本的数据集类型：0=scannet(室内), 1=argoverse2(室外)
                sample_dataset_type = dataset_type[i].item()
                # 根据 dataset_type 选择对应的编码器
                point_model = point_models[sample_dataset_type]
                # 确保输入是 float32，并放到正确的 device
                pts = pts.to(device=device, dtype=torch.float32)  # [1,N,6]
                # 对点云编码器显式关闭 autocast，强制 fp32
                with torch.cuda.amp.autocast(enabled=False):
                    pe, pd = get_point_embeds(
                        raw_pointcloud=pts,
                        point_model=point_model,   # 根据 dataset_type 选择的编码器
                        device=device,
                        compute_normals=True,
                    )
                # pe, pd 默认是 float32，在后续传给 MIDI 前你可以随它被 cast 成 bf16
                point_embeds_list.append(pe)          # [576,C]
            point_embeds = torch.stack(point_embeds_list, dim=0)
            # 3) 塞回 batch，供 MIDI 使用
            batch["image_embeds"] = image_embeds
            batch["point_embeds"] = point_embeds
            # dataset_type 转换为 tensor 并放到 device 上，用于后续模型前向
            dataset_type = dataset_type.to(device)  # [B]
            
            # 如果不需要计算MIDI loss，只获取tor_embeds
            tor_embeds = midi(
                mamba_input_ids=batch["mamba_input_ids"]
                , mamba_attention_mask=batch["mamba_attention_mask"]
                , image_embeds=batch["image_embeds"]
                , point_embeds=batch["point_embeds"]
                , tor_token_id=tor_token_id
                , alignment=False
                , dataset_type=dataset_type
            ).tor_embeds
            # tor_embeds = projector(tor_embeds)
            outputs = vlm(
                input_ids=batch["vlm_input_ids"]
                , attention_mask=batch["vlm_attention_mask"]
                , pixel_values=batch["vlm_pixel_values"]
                , image_grid_thw=batch["vlm_image_grid_thw"]
                , labels=batch["vlm_labels"]
                , tor_embeds=tor_embeds
                , tor_token_id=tor_token_id[1]
            )
            loss = outputs.loss
            accelerator.backward(loss)

            with torch.no_grad():
                embedding = vlm.get_input_embeddings()
                grad = embedding.weight.grad
                if grad is not None:
                    mask = torch.zeros_like(grad)
                    mask[tor_token_id[1]] = 1.0
                    grad *= mask
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(accelerator.gather(loss).mean().item())
            progress_bar.set_description(f"{str_datetime()} [Epoch {epoch + 1}/{epochs}] | Loss: {losses[-1]:.4f}")
    return losses


def main(args: Namespace) -> None:
    # 1. 定义需要 Wrap 的层集合
    transformer_layer_cls = {
        Qwen2DecoderLayer,       # MIDI 的层
        Qwen2_5_VLDecoderLayer,  # VLM 的语言层
        Qwen2_5_VLVisionBlock    # VLM 的视觉层
    }

    # 2. 构建 auto_wrap_policy 函数
    # 使用 functools.partial 将层列表绑定到 PyTorch 的默认策略上
    fsdp_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )

    # 3. 初始化插件，使用 'auto_wrap_policy' 参数
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        limit_all_gathers=True,
        auto_wrap_policy=fsdp_auto_wrap_policy  # <--- 注意参数名是 auto_wrap_policy
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    # accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.print(f"{str_datetime()} Loading Tokenizer & Processor...")
    args.mamba = json.load(open(os.path.join(args.pretrained_midi, "args.json")))["mamba"]
    mamba_tokenizer = AutoTokenizer.from_pretrained(args.mamba)
    mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
    vlm_processor = Qwen2_5_VLProcessor.from_pretrained(args.pretrained_vlm)
    vlm_processor.tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)

    accelerator.print(f"{str_datetime()} Loading CLIP and Siglip Models...")
    clip_processor, clip_model = CLIPProcessor.from_pretrained(args.clip_path), (CLIPVisionModel.from_pretrained(args.clip_path))
    clip_model = accelerator.prepare(clip_model)

    # 点云编码器：不要走 accelerator.prepare，不要 FSDP，不要混合精度
    # 创建两个编码器实例：室内和室外
    point_model_indoor = SonataPointEncoder(dataset_type="indoor", device=str(accelerator.device)).to(accelerator.device)  # 室内编码器
    point_model_outdoor = SonataPointEncoder(dataset_type="outdoor", device=str(accelerator.device)).to(accelerator.device)  # 室外编码器
    freeze_module(point_model_indoor)
    freeze_module(point_model_outdoor)
    point_model_indoor.eval()  # 只做推理，用不到梯度
    point_model_outdoor.eval()  # 只做推理，用不到梯度
    # 创建编码器字典，方便根据 dataset_type 选择
    point_models = {
        0: point_model_indoor,   # 0 = scannet (室内)
        1: point_model_outdoor,  # 1 = argoverse2 (室外)
    }

    accelerator.print(f"{str_datetime()} Loading Dataset...")
    dataset_scannet = SSRCoTDataset4VLM(
        data_dir=args.data_dir
        , n_tor=args.n_tor
        , mamba_tokenizer=mamba_tokenizer
        , vlm_processor=vlm_processor
        , max_length=args.max_length
        , image_size=args.image_size
        , clip_processor=clip_processor
        , llava=args.llava
        , jsonl_name=args.jsonl1          # <<< 新增: 使用 scannet 对应 jsonl
        , dataset_type="scannet"          # <<< 新增
    )

    # Argoverse2 数据集
    dataset_av2 = SSRCoTDataset4VLM(
        data_dir=args.data_dir2
        , n_tor=args.n_tor
        , mamba_tokenizer=mamba_tokenizer
        , vlm_processor=vlm_processor
        , max_length=args.max_length
        , image_size=args.image_size
        , clip_processor=clip_processor
        , llava=args.llava
        , jsonl_name=args.jsonl2          # <<< 新增: 使用 av2 对应 jsonl
        , dataset_type="argoverse2"       # <<< 新增
    )

    # 合并两个数据集
    full_dataset = ConcatDataset([dataset_scannet, dataset_av2])

    accelerator.print(
        f"{str_datetime()} ScanNet samples = {len(dataset_scannet)}, "
        f"Argoverse2 samples = {len(dataset_av2)}, "
        f"Total = {len(full_dataset)}"
    )

    accelerator.print(f"{str_datetime()} Loading Model...")
    midi = MIDI.from_pretrained(args.pretrained_midi, device_map="cpu") # 确保MIDI放到cpu

    # 彻底垃圾回收
    del midi.llm
    gc.collect()
    torch.cuda.empty_cache() 
    accelerator.print(f"Deleted midi.llm and cleaned memory.")

    tor_token_id = (
        mamba_tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
        , vlm_processor.tokenizer._tokenizer.token_to_id(SSRSpecialToken.TOR_TOKEN)
    )

    # vlm加载也强制到cpu
    vlm = SSRVLM.from_pretrained(args.pretrained_vlm, device_map="cpu")

    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r
            , lora_alpha=args.lora_alpha
            , lora_dropout=args.lora_dropout
            , task_type="CAUSAL_LM"
            , target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        vlm = get_peft_model(vlm, lora_config)
        vlm.to("cpu")  
        vlm.get_input_embeddings().weight.requires_grad_(True)

    accelerator.print(f"{str_datetime()} VLM: {count_params(vlm)}")
    # 再次清理，确保 prepare 之前 GPU 是干净的
    torch.cuda.empty_cache()
    gc.collect()
    accelerator.print(f"开始 Prepare...")
    midi, vlm = accelerator.prepare(midi, vlm)

    # ================= 修复维度不匹配() =================
    # 获取维度
    # midi_dim = 1536 # 打印查看tor_embeds的shape得到
    # vlm_dim = 2048
    # accelerator.print(f"Checking dimensions: MIDI={midi_dim}, VLM={vlm_dim}")
    # projector = None
    # if midi_dim != vlm_dim:
    #     accelerator.print(f"Detected dimension mismatch! Creating Projector: {midi_dim} -> {vlm_dim}")
    #     # 定义投影层：1536 -> 2048
    #     projector = nn.Linear(midi_dim, vlm_dim, bias=False)
    #     # 初始化
    #     nn.init.xavier_uniform_(projector.weight)
    #     projector.requires_grad_(True)
    # else:
    #     accelerator.print("Dimensions match, no projector needed.")

    accelerator.print(f"{str_datetime()} Preparing Optimizer, Dataloader, Scheduler...")
    # 把projector也加入可训练，同时加入loss权重参数
    optimizer = torch.optim.AdamW(
        params=list(midi.parameters()) + list(vlm.parameters()),
        lr=args.lr
    )

    # 先根据 CPU 数量和并行进程数，自动算一个比较稳妥的 num_workers
    cpu_num = os.cpu_count() or 8
    world_size = accelerator.num_processes  # accelerate 控制的总进程数
    # 每个 rank 上的 worker 数：2~8 之间
    num_workers = min(8, max(2, cpu_num // world_size))
    # ========== 根据 mix_strategy 构建 DataLoader ==========
    if args.mix_strategy == "concat":
        # 简单拼接：按样本总数随机打乱
        dataloader = DataLoader(
            full_dataset,
            batch_size=args.batch_size_per_gpu,
            shuffle=True,
            num_workers=num_workers,          # <<< 新增
            pin_memory=True,                  # <<< 建议新增
            persistent_workers=True,          # <<< 建议新增（需要 num_workers>0）
            collate_fn=dataset_scannet.collate_fn,   # 两个数据集的 collate_fn 一致
        )
    elif args.mix_strategy == "balanced":
        # 平衡采样：两个数据集出现概率尽量接近
        num_scannet = len(dataset_scannet)
        num_av2 = len(dataset_av2)
        # 每个数据集内部均匀，但整体上两者权重相近
        weights_scannet = np.full(num_scannet, 1.0 / num_scannet, dtype=np.float64)
        weights_av2 = np.full(num_av2, 1.0 / num_av2, dtype=np.float64)
        all_weights = np.concatenate([weights_scannet, weights_av2], axis=0)
        sampler = WeightedRandomSampler(
            torch.as_tensor(all_weights, dtype=torch.double),
            num_samples=len(full_dataset),          # 每个 epoch 采这么多样本
            replacement=True
        )
        dataloader = DataLoader(
            full_dataset,
            batch_size=args.batch_size_per_gpu,
            sampler=sampler,
            shuffle=False,                    # <<< sampler 存在时必须关掉 shuffle
            num_workers=num_workers,          # <<< 新增
            pin_memory=True,                  # <<< 建议新增
            persistent_workers=True,          # <<< 建议新增
            collate_fn=dataset_scannet.collate_fn,
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer
        , num_warmup_steps=int((len(dataloader) * args.epochs) * args.warmup_ratio)
        , num_training_steps=(len(dataloader) * args.epochs)
    )
    # 准备optimizer, dataloader, scheduler
    # loss_weights的参数已经包含在optimizer中，不需要单独prepare
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)

    accelerator.print(f"{str_datetime()} Training...")
    losses = train(
        midi, vlm, dataloader, optimizer, scheduler, accelerator, tor_token_id, args.epochs, 
        clip_model=clip_model, point_models=point_models,
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        np.save(os.path.join(args.output_dir, "losses.npy"), losses)
    accelerator.print(f"{str_datetime()} Saving Checkpoint into {args.output_dir} ...")

    accelerator.unwrap_model(midi).save_pretrained(
        os.path.join(args.output_dir, "MIDI")
        , is_main_process=accelerator.is_main_process
        , save_function=accelerator.save
        , state_dict=accelerator.get_state_dict(midi)
    )
    accelerator.print(f"{str_datetime()} MIDI Save Completed.")

    accelerator.unwrap_model(vlm).save_pretrained(
        os.path.join(args.output_dir, "SSRVLM")
        , is_main_process=accelerator.is_main_process
        , save_function=accelerator.save
        , state_dict=accelerator.get_state_dict(vlm)
    )
    accelerator.print(f"{str_datetime()} VLM Save Completed.")
    
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
    accelerator.print(f"{str_datetime()} Done.")


if __name__ == "__main__":
    quiet()
    args = get_args()
    main(args)

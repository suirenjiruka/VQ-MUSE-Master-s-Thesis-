import torch
import math
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from os.path import join as pjoin
from transformers import get_cosine_schedule_with_warmup
from evaluator.evaluator import evaluation_generation_hml, evaluation_generation_hml_mixed, evaluation_motion_editing_hml
from SnapMogen_model.transformer.tools import *

'''
Training code is borrowed from the framework of SnapMogen, including VQ-model, traing process, and most relatted configuratioon
'''

class ModelEMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and not name.startswith("text_emb."):
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.decay = float(state.get("decay", self.decay))
        loaded = state.get("shadow", {})
        for name in list(self.shadow.keys()):
            if name in loaded:
                self.shadow[name] = loaded[name].detach().clone().to(self.shadow[name].device)

    @torch.no_grad()
    def store(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()

    @torch.no_grad()
    def copy_to(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


class VA_motion_trainer:
    def __init__(self, cfg, vq_model, va_transformer, eval_wrapper, device, edit_eval_wrapper=None):
        #config setup
        self.config = cfg
        self.vq_model = vq_model
        self.vq_model.eval()
        self.training_model = va_transformer
        self.eval_wrapper = eval_wrapper
        self.edit_eval_wrapper = edit_eval_wrapper if edit_eval_wrapper is not None else eval_wrapper
        #self.eval_wrapper.eval()
        self.device = device
        lr = float(cfg.training.lr)
        weight_decay = float(cfg.training.weight_decay)
        # frozen branches (two-stage plan) stay out of the optimizer; also keeps optimizer state resume-compatible
        self.optimizer = optim.AdamW([p for p in self.training_model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        self.ema_decay = float(getattr(cfg.training, "ema_decay", 0.0))
        self.use_ema = 0.0 < self.ema_decay < 1.0
        self.ema = None
        # setup logger
        self.logger = SummaryWriter(cfg.exp.log_dir)

    def forward(self, data_batch):
        # data fetch
        caption, src_motion, tgt_motion, m_length, src_joints, tgt_joints, has_source, _task_type, _name = data_batch[:9]
        src_m_length = data_batch[12]
        same_src_text = data_batch[13] if len(data_batch) > 13 else None
        same_src_flag = data_batch[14] if len(data_batch) > 14 else None
        same_tgt_motion = data_batch[15] if len(data_batch) > 15 else None
        same_m_length = data_batch[16] if len(data_batch) > 16 else None
        src_motion = src_motion.detach().float().to(self.device)
        tgt_motion = tgt_motion.detach().float().to(self.device)
        src_joints = src_joints.detach().float().to(self.device)
        tgt_joints = tgt_joints.detach().float().to(self.device)
        m_length = m_length.detach().long().to(self.device)
        src_m_length = src_m_length.detach().long().to(self.device)
        has_source = has_source.detach().long().to(self.device)

        # VQ encode with own len
        target_code_idx, _ = self.vq_model.encode(tgt_motion[..., :self.config.data.dim_pose], m_length)
        source_code_idx, _ = self.vq_model.encode(src_motion[..., :self.config.data.dim_pose], src_m_length)
        m_lens = m_length // self.config.data.unit_length
        src_m_lens = src_m_length // self.config.data.unit_length
        same_target_code_idx = None
        same_m_lens = None
        use_same_metric = getattr(self.config.loss, "weight_same_src", 0.0) > 0
        use_same_metric = use_same_metric or ((not self.training_model.training) and getattr(self.config.loss, "eval_same_src_metric", True))
        if same_src_flag is not None and use_same_metric:
            same_flag = same_src_flag.detach().long().to(self.device) if torch.is_tensor(same_src_flag) else None
            if same_flag is not None and same_flag.sum() > 0:
                same_tgt_motion = same_tgt_motion.detach().float().to(self.device)
                same_m_length = same_m_length.detach().long().to(self.device)
                same_target_code_idx, _ = self.vq_model.encode(same_tgt_motion[..., :self.config.data.dim_pose], same_m_length)
                same_m_lens = same_m_length // self.config.data.unit_length

        #ensure all input device consistant
        caption = caption.to(self.device).float() if torch.is_tensor(caption) else caption

        _loss, loss_mt, delta_loss, latent_loss, rank_loss, same_loss, same_gap, _acc = self.training_model(
            target_code_idx, source_code_idx, caption, m_lens, has_source,
            source_m_lens=src_m_lens, source_joints=src_joints, target_joints=tgt_joints,
            same_src_text=same_src_text, same_src_flag=same_src_flag,
            same_target_input=same_target_code_idx, same_m_lens=same_m_lens
        )

        return _loss, loss_mt, delta_loss, latent_loss, rank_loss, same_loss, same_gap, _acc
    
    def save(self, save_pth, epoch: int):
        # save model
        model_state_dict = self.training_model.state_dict()
        # del T5 model parameter (to large & unnecessary)
        t5_weights = [e for e in model_state_dict.keys() if e.startswith('text_emb.')]
        for e in t5_weights:
            del model_state_dict[e]

        state = {
            'training_model': model_state_dict, # model parameter
            'optimizer': self.optimizer.state_dict(), # save optimizer
            'lr_scheduler':self.lr_scheduler.state_dict(), # save scheduler
            'epoch': epoch,
        }
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
        torch.save(state, save_pth)

    

    def train(self, train_loader, val_loader, eval_loader, gen_eval_loader=None):
        self.training_model.to(self.device)
        self.vq_model.to(self.device)

        max_epoch = self.config.training.max_epoch
        epoch = 0
        current_iter = 0

        #set up scheduler
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.training.warm_up_iter,
                                                            num_training_steps= max_epoch * len(train_loader))
        #repair if cfg.exp.is_continue = true
        if self.config.exp.is_continue == True:
            model_dir = pjoin(self.config.exp.model_dir, 'latest.tar')  # we save latest everyy 1 epochs, and index_based checkpointt everyy 50 epochs
            checkpoint = torch.load(model_dir, map_location=self.device, weights_only=True)

            _, unexpected_keys = self.training_model.load_state_dict(checkpoint['training_model'], strict=False)
            old_keys = ("text_delta_encoder.", "edit_map_head.", "latent_MLP.", "edit_delta_head.")
            unexpected_keys = [k for k in unexpected_keys if not k.startswith(old_keys)]
            assert len(unexpected_keys) == 0
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer']) # Optimizer
            except ValueError:
                print("optimizer state is incompatible with current trainable params; reset optimizer")
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) # Scheduler
            epoch = checkpoint['epoch']
            if self.use_ema:
                self.ema = ModelEMA(self.training_model, self.ema_decay)
                if 'ema' in checkpoint:
                    self.ema.load_state_dict(checkpoint['ema'])
                else:
                    print("EMA state not found in checkpoint; initialize EMA from loaded model")

            del checkpoint
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        elif self.use_ema:
            self.ema = ModelEMA(self.training_model, self.ema_decay)

        if self.use_ema:
            print(f"EMA enabled: decay={self.ema_decay}")

        #Output training Info
        # time setup
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        #total trining epoch & iteration
        total_iter = self.config.training.max_epoch * len(train_loader)
        print(f"training epoch: {self.config.training.max_epoch}, {len(train_loader)} iterations in each epoch.")
        print(f"total training iteration: {total_iter}.") 
        # loss weight setup
        w_trans = self.config.loss.weight_transformer_loss
        w_mtext  = self.config.loss.weight_motion_text_InfoNCE
        w_delta = getattr(self.config.loss, "weight_delta", 0.0)
        w_latent = getattr(self.config.loss, "weight_latent", 0.0)
        w_rank = getattr(self.config.loss, "weight_rank", 0.0)
        w_same = getattr(self.config.loss, "weight_same_src", 0.0)

        # tensorboard log format initialiozation
        logs = {"total_loss": 0.0, "transformer_loss": 0.0, "InfoNCE_text": 0.0,
                "vq_delta": 0.0, "vq_latent": 0.0, "vq_rank": 0.0,
                "same_src": 0.0, "same_gap": 0.0, "accuracy": 0.0}
        log_period = self.config.training.log_every

        # init training parameter
        best_g2t_r1 = 0.0
        best_g2t_avgr = float("inf")
        best_fid = float("inf")

        #setup iterative epoch
        while epoch < max_epoch:
            self.training_model.train()
            start.record()

            #data processing & training
            for i, batch in tqdm(enumerate(train_loader)):
                current_iter += 1
                # calculatte loss & optimization
                transformer_loss, InfoNCE_mt_loss, delta_loss, latent_loss, rank_loss, same_loss, same_gap, acc = self.forward(batch)
                loss = (w_trans * transformer_loss + w_mtext * InfoNCE_mt_loss
                        + w_delta * delta_loss + w_latent * latent_loss + w_rank * rank_loss
                        + w_same * same_loss)
                self.optimizer.zero_grad()
                if math.isnan(loss.item()):
                    continue # loss = nan, skip
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), 1.0)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.training_model)
                self.lr_scheduler.step()

                logs['total_loss'] += loss.item()
                logs['transformer_loss'] += transformer_loss.item()
                logs['InfoNCE_text'] += InfoNCE_mt_loss.item()
                logs['vq_delta'] += delta_loss.item()
                logs['vq_latent'] += latent_loss.item()
                logs['vq_rank'] += rank_loss.item()
                logs['same_src'] += same_loss.item()
                logs['same_gap'] += same_gap.item()
                logs['accuracy'] += acc

                #print log every n steps 
                if current_iter % log_period == 0:
                    for key, value in logs.items():
                        self.logger.add_scalar(f"train/{key}", value / log_period, current_iter)
                    # print logs
                    print(f"cuurent iter: {current_iter}. Avg total loss: {logs['total_loss'] / log_period:.4f}, transformer loss: {logs['transformer_loss'] / log_period:.4f}, InfoNCE_text: {logs['InfoNCE_text'] / log_period:.4f}, vq_delta: {logs['vq_delta'] / log_period:.4f}, vq_latent: {logs['vq_latent'] / log_period:.4f}, vq_rank: {logs['vq_rank'] / log_period:.4f}, same_src: {logs['same_src'] / log_period:.4f}, same_gap: {logs['same_gap'] / log_period:.4f}, accuracy: {logs['accuracy'] / log_period:.3f}")
                    # reset logs
                    logs =  {"total_loss": 0.0, "transformer_loss": 0.0, "InfoNCE_text": 0.0,
                             "vq_delta": 0.0, "vq_latent": 0.0, "vq_rank": 0.0,
                             "same_src": 0.0, "same_gap": 0.0, "accuracy": 0.0}
                    
                    
            #save checkpoints
            if epoch % 50 == 0:  # save regular check point every 50 epochs
                self.save(pjoin(self.config.exp.model_dir, f'_{epoch}.tar'), epoch)
            self.save(pjoin(self.config.exp.model_dir, 'latest.tar'), epoch) # save latest check point every epoch

            print('Validation time:')
            self.training_model.eval()
            if self.ema is not None:
                self.ema.store(self.training_model)
                self.ema.copy_to(self.training_model)

            val_loss = 0.0
            val_acc =0.0
            val_delta = 0.0
            val_latent = 0.0
            val_rank = 0.0
            val_same = 0.0
            val_same_gap = 0.0
            val_num = len(val_loader)

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    transformer_loss, InfoNCE_mt_loss, delta_loss, latent_loss, rank_loss, same_loss, same_gap, acc = self.forward(batch)
                    loss = (w_trans * transformer_loss + w_mtext * InfoNCE_mt_loss
                            + w_delta * delta_loss + w_latent * latent_loss + w_rank * rank_loss
                            + w_same * same_loss)
                    val_loss += loss.item()
                    val_acc += acc
                    val_delta += delta_loss.item()
                    val_latent += latent_loss.item()
                    val_rank += rank_loss.item()
                    val_same += same_loss.item()
                    val_same_gap += same_gap.item()
                    # round to 3 decimals
                val_loss = val_loss / val_num
                val_acc = val_acc / val_num
                val_delta = val_delta / val_num
                val_latent = val_latent / val_num
                val_rank = val_rank / val_num
                val_same = val_same / val_num
                val_same_gap = val_same_gap / val_num
                print(f"validation result, loss: {val_loss:.3f}, vq_delta: {val_delta:.3f}, vq_latent: {val_latent:.3f}, vq_rank: {val_rank:.3f}, same_src: {val_same:.3f}, same_gap: {val_same_gap:.3f}, accuracy: {val_acc:.3f}")

                self.logger.add_scalar('Val/loss', val_loss, epoch)
                self.logger.add_scalar('Val/acc', val_acc, epoch)
                self.logger.add_scalar('Val/vq_delta', val_delta, epoch)
                self.logger.add_scalar('Val/vq_latent', val_latent, epoch)
                self.logger.add_scalar('Val/vq_rank', val_rank, epoch)
                self.logger.add_scalar('Val/same_src', val_same, epoch)
                self.logger.add_scalar('Val/same_gap', val_same_gap, epoch)
            
            gen_eval_fn = evaluation_generation_hml if gen_eval_loader is not None else evaluation_generation_hml_mixed
            gen_metrics = gen_eval_fn(
                gen_eval_loader if gen_eval_loader is not None else eval_loader,
                self.training_model, self.vq_model, self.logger, epoch,
                eval_wrapper=self.eval_wrapper, device=self.device,
                time_steps=10, cond_scale=4, source_cond_scale=getattr(self.config.inference, "source_cond_scale", 1.0), unit_length=self.config.data.unit_length,
                temperature=1, topk_filter_thres=0.95, gsample=self.config.training.gumbel_sample
            )
            eval_metrics = evaluation_motion_editing_hml(
                eval_loader, self.training_model, self.vq_model, self.logger, epoch,
                eval_wrapper=self.edit_eval_wrapper, device=self.device,
                # edit dual CFG expands to cond*C - (cond-source)*B; cond=4 extrapolates the undertrained B branch 3x -> off-manifold FID spikes
                time_steps=10, cond_scale=getattr(self.config.inference, "edit_cond_scale", 2), source_cond_scale=getattr(self.config.inference, "source_cond_scale", 1.0), unit_length=self.config.data.unit_length,
                temperature=1, topk_filter_thres=0.95, gsample=self.config.training.gumbel_sample
            )
            if eval_metrics is not None:
                improve_g2t = eval_metrics["g2t_r1"] > best_g2t_r1
                tie_break = eval_metrics["g2t_r1"] == best_g2t_r1 and eval_metrics["g2t_avgr"] < best_g2t_avgr
                if improve_g2t or tie_break:
                    print(
                        f"G2T improved: R@1 {best_g2t_r1:.5f}->{eval_metrics['g2t_r1']:.5f}, "
                        f"AvgR {best_g2t_avgr:.2f}->{eval_metrics['g2t_avgr']:.2f}"
                    )
                    self.save(pjoin(self.config.exp.model_dir, 'best.tar'), epoch)
                    best_g2t_r1 = eval_metrics["g2t_r1"]
                    best_g2t_avgr = eval_metrics["g2t_avgr"]
                if eval_metrics["fid"] < best_fid:
                    print(f"FID improved from {best_fid:.5f} to {eval_metrics['fid']:.5f}")
                    self.save(pjoin(self.config.exp.model_dir, 'best_fid.tar'), epoch)
                    best_fid = eval_metrics["fid"]

            # timer close
            end.record()
            torch.cuda.synchronize()
            epoch_duration = start.elapsed_time(end) * 0.001
            print(f"epoch {epoch} duration: {epoch_duration} sec")
            if self.ema is not None:
                self.ema.restore(self.training_model)

            epoch += 1

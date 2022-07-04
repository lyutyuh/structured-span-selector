import logging
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from torch.optim import Adam
from tensorize import CorefDataProcessor
import util
import time
from os.path import join
from metrics import CorefEvaluator
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR

import json

import model as Model

import conll
import sys

torch.autograd.set_detect_anomaly(False)
USE_AMP = True

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

class Runner:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')
        self.scaler = torch.cuda.amp.GradScaler(init_scale=256.0)

        # Set up data
        self.data = CorefDataProcessor(self.config)

    def initialize_model(self, saved_suffix=None):
        model = Model.CorefModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)
        
        # Get model parameters for grad clipping
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1, max_f1_test = 0, 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            print("EPOCH", epo)
            random.shuffle(examples_train)  # Shuffle training set
            for doc_key, example in examples_train:
                # Forward pass
                model.train()
                example_gpu = [d.to(self.device) if d is not None else None for d in example]
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    _, loss = model(*example_gpu)

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                
                if USE_AMP:
                    scaled_loss = self.scaler.scale(loss)
                    scaled_loss.backward()
                else:
                    loss.backward()
                
                loss_during_accum.append(loss.item())
                
                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if USE_AMP:
                        self.scaler.unscale_(optimizers[0])
                        self.scaler.unscale_(optimizers[1])
                    
                    if conf['max_grad_norm']:
                        norm_bert = torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'], error_if_nonfinite=False)
                        norm_task = torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'], error_if_nonfinite=False)
                        
                    for optimizer in optimizers:
                        if USE_AMP:
                            self.scaler.step(optimizer)
                        else:
                            optimizer.step()
                    
                    if USE_AMP:
                        self.scaler.update()
                        
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        # Testing Dev
                        logger.info('Dev')
                        f1, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False, conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)
                        # Testing Test
                        logger.info('Test')
                        f1_test = 0.
                        if f1 > max_f1 or f1_test > max_f1_test:
                            max_f1 = max(max_f1, f1)
                            max_f1_test = 0. # max(max_f1_test, f1_test)
                            self.save_model_checkpoint(model, len(loss_history))
                        
                        logger.info('Eval max f1: %.2f' % max_f1)
                        logger.info('Test max f1: %.2f' % max_f1_test)
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None, predict=False):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        
        model.to(self.device)
        evaluator = CorefEvaluator()
        doc_to_prediction = {}

        model.eval()
        
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            current_json = {}

            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:7]  # Strip out gold
            example_gpu = [d.to(self.device) for d in tensor_example]

            with torch.no_grad():
                returned_tuple = model(*example_gpu)

            if len(returned_tuple) == 10:
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, score_j_i, input_ids, head_cond_score = returned_tuple

                current_json["score_j_i"] = score_j_i.tolist()
                current_json["input_ids"] = input_ids.tolist()
                current_json["head_cond_score"] = head_cond_score.tolist()

            elif len(returned_tuple) == 7:
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = returned_tuple

            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx, antecedent_scores

            predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator)
            doc_to_prediction[doc_key] = predicted_clusters
            

        p, r, f, m_recall, (blanc_p, blanc_r, blanc_f) = evaluator.get_prf()
        all_metrics = evaluator.get_all()
        
        metrics = {
            'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100, 
            "Eval_Men_Recall": m_recall*100,
            "Eval_Blanc_Precision": blanc_p * 100, "Eval_Blanc_Recall": blanc_r * 100, "Eval_Blanc_F1": blanc_f * 100
        }
        
        for k,v in all_metrics.items():
            logger.info('%s: %.4f'%(k, v))
            
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        return f * 100, metrics

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        model.eval()
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        for i,  (doc_key, tensor_example)  in enumerate(tensor_examples):
            tensor_example = tensor_example[:7]
            example_gpu = [d.to(self.device) for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores = model(*example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

        return predicted_clusters, predicted_spans, predicted_antecedents

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'], weight_decay=0)
        ]
        return optimizers

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers

    def save_model_checkpoint(self, model, step):
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)


if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    saved_suffix = sys.argv[3] if len(sys.argv) >= 4 else None
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    runner.train(model)

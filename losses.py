# # Copyright (c) 2015-present, Facebook, Inc.
# # All rights reserved.
# """
# Implements the knowledge distillation loss
# """
# import torch
# from torch.nn import functional as F


# class DistillationLoss(torch.nn.Module):
#     """
#     This module wraps a standard criterion and adds an extra knowledge distillation loss by
#     taking a teacher model prediction and using it as additional supervision.
#     """
#     def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
#                  distillation_type: str, alpha: float, tau: float):
#         super().__init__()
#         self.base_criterion = base_criterion
#         self.teacher_model = teacher_model
#         assert distillation_type in ['none', 'soft', 'hard']
#         self.distillation_type = distillation_type
#         self.alpha = alpha
#         self.tau = tau
    
#     # def _KD_loss(pred, soft, T=float(2.0)):
#     #     pred = torch.log_softmax(pred/T, dim=1)
#     #     soft = torch.softmax(soft/T, dim=1)
#     #     return -1*torch.mul(soft, pred).sum()/pred.shape[0]

#     def forward(self, inputs, outputs, labels):
#         """
#         Args:
#             inputs: The original inputs that are feed to the teacher model
#             outputs: the outputs of the model to be trained. It is expected to be
#                 either a Tensor, or a Tuple[Tensor, Tensor], with the original output
#                 in the first position and the distillation predictions as the second output
#             labels: the labels for the base criterion
#         """
#         # # 得到教师与学生模型每一层的输出结果
#         # outputs_student = []
#         # outputs_teacher = []
#         # # Hook函数
#         # def hook_fn_student(module, input, output):
#         #     outputs_student.append(output)
#         # def hook_fn_teacher(module, input, output):
#         #     outputs_teacher.append(output)
#         # # 注册hook到每个Transformer block
#         # for i, block in enumerate(self.teacher_model.blocks):
#         #     block.register_forward_hook(hook_fn_teacher)

#         outputs_kd = outputs[1]
#         outputs_layer_kd = outputs[2]
#         output_base = outputs[0]
#         # if not isinstance(outputs, torch.Tensor):
#         #     # assume that the model outputs a tuple of [outputs, outputs_kd]
#         #     outputs, outputs_kd = outputs
#         base_loss = self.base_criterion(output_base, labels)
#         if self.distillation_type == 'none':
#             return base_loss
#         # from IPython import embed
#         # embed()
#         if outputs_kd is None:
#             raise ValueError("When knowledge distillation is enabled, the model is "
#                              "expected to return a Tuple[Tensor, Tensor] with the output of the "
#                              "class_token and the dist_token")
#         # don't backprop throught the teacher
#         with torch.no_grad():
#             teacher_outputs = self.teacher_model(inputs)
#             teacher_outputs_logit = teacher_outputs[0]
#             teacher_outputs_kd = teacher_outputs[1]
#             teacher_outputs_layer_kd = teacher_outputs[2]

#         if self.distillation_type == 'soft':
#             T = self.tau
#             # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
#             # with slight modifications
#             # from IPython import embed
            
#             # embed()
#             # # Loss 1
#             # distillation_loss = F.kl_div(
#             #     F.log_softmax(outputs_kd / T, dim=1),
#             #     #We provide the teacher's targets in log probability because we use log_target=True 
#             #     #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
#             #     #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
#             #     F.log_softmax(teacher_outputs_kd / T, dim=1),
#             #     reduction='sum',
#             #     log_target=True
#             # ) * (T * T) / outputs_kd.numel()

#             # Loss 2
#             pred = torch.log_softmax(outputs_kd/T, dim=1)
#             soft = torch.softmax(teacher_outputs_kd/T, dim=1)
#             distillation_loss = -1*torch.mul(soft, pred).sum()/pred.shape[0]
#             #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
#             #But we also experiments output_kd.size(0) 
#             #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
#         elif self.distillation_type == 'hard':
#             distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs_kd.argmax(dim=1))

#         # # layer feature distill
#         # layer_distill_loss = 0
#         # for i in range(len(outputs_layer_kd) - 1):
#         #     pred = torch.log_softmax(outputs_layer_kd[i]/T, dim=1)
#         #     soft = torch.softmax(teacher_outputs_layer_kd[i]/T, dim=1)
#         #     layer_distill_loss += -1*torch.mul(soft, pred).sum()/pred.shape[0]
        
#         # layer logit distill
#         layer_distill_loss = 0
#         for i in range(len(outputs_layer_kd) - 1):
#             pred = torch.log_softmax(outputs_layer_kd[i]/T, dim=1)
#             soft = torch.softmax(teacher_outputs_logit/T, dim=1)
#             layer_distill_loss += -1*torch.mul(soft, pred).sum()/pred.shape[0]
#         # loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
#         # loss = (base_loss  + distillation_loss * 8) / 9
#         loss = (base_loss  + distillation_loss + layer_distill_loss) 
#         return loss
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.num_classes = 1000
        self.eps = 1
    
    # def _KD_loss(pred, soft, T=float(2.0)):
    #     pred = torch.log_softmax(pred/T, dim=1)
    #     soft = torch.softmax(soft/T, dim=1)
    #     return -1*torch.mul(soft, pred).sum()/pred.shape[0]

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        # # 得到教师与学生模型每一层的输出结果
        # outputs_student = []
        # outputs_teacher = []
        # # Hook函数
        # def hook_fn_student(module, input, output):
        #     outputs_student.append(output)
        # def hook_fn_teacher(module, input, output):
        #     outputs_teacher.append(output)
        # # 注册hook到每个Transformer block
        # for i, block in enumerate(self.teacher_model.blocks):
        #     block.register_forward_hook(hook_fn_teacher)

        outputs_kd = outputs[1]
        outputs_layer_kd = outputs[2]
        output_base = outputs[0]
        # if not isinstance(outputs, torch.Tensor):
        #     # assume that the model outputs a tuple of [outputs, outputs_kd]
        #     outputs, outputs_kd = outputs
        base_loss = self.base_criterion(output_base, labels)
        if self.distillation_type == 'none':
            return base_loss
        # from IPython import embed
        # embed()
        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            teacher_outputs_kd = teacher_outputs[1]
            teacher_outputs_layer_kd = teacher_outputs[2]

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            # from IPython import embed
            
            # embed()
            # # Loss 1
            # distillation_loss = F.kl_div(
            #     F.log_softmax(outputs_kd / T, dim=1),
            #     #We provide the teacher's targets in log probability because we use log_target=True 
            #     #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
            #     #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
            #     F.log_softmax(teacher_outputs_kd / T, dim=1),
            #     reduction='sum',
            #     log_target=True
            # ) * (T * T) / outputs_kd.numel()

            # Loss 2
            pred = torch.log_softmax(outputs_kd/T, dim=1)
            soft = torch.softmax(teacher_outputs_kd/T, dim=1)
            distillation_loss = -1*torch.mul(soft, pred).sum()/pred.shape[0]
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs_kd.argmax(dim=1))
        
        elif self.distillation_type == 'ofa':

            if len(labels.shape) != 1:  # label smoothing
                target_mask = F.one_hot(labels.argmax(-1), self.num_classes)
            else:
                target_mask = F.one_hot(labels, self.num_classes)
            pred_student = F.softmax(outputs_kd / T, dim=1)
            pred_teacher = F.softmax(teacher_outputs_kd / T, dim=1)
            prod = (pred_teacher + target_mask) * self.eps
            distillation_loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
        # # # layer distill
        # layer_distill_loss = 0
        # for i in range(len(outputs_layer_kd) - 1):
        #     pred = torch.log_softmax(outputs_layer_kd[i]/T, dim=1)
        #     soft = torch.softmax(teacher_outputs_layer_kd[i]/T, dim=1)
        #     layer_distill_loss += -1*torch.mul(soft, pred).sum()/pred.shape[0]
            
        # layer distill(only use last teacher's output)
        layer_distill_loss = 0
        ii = 1
        for i in range(len(outputs_layer_kd) - 1):
            pred = torch.log_softmax(outputs_layer_kd[i]/T, dim=1)
            soft = torch.softmax(teacher_outputs_kd/T, dim=1)
            tem_loss = -1*torch.mul(soft, pred).sum()/pred.shape[0]
            layer_distill_loss += tem_loss * ii
            ii += 1
            # layer_distill_loss += self._KD_loss(outputs_layer_kd[i], teacher_outputs_layer_kd[i])
        # loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        # loss = (base_loss  + distillation_loss * 8) / 9
        loss = (base_loss  + distillation_loss * 120    + layer_distill_loss) 
        # loss = base_loss  + distillation_loss 
        return loss
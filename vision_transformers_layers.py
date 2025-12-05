import torch

from timm.models.vision_transformer import VisionTransformer
class VisionTransformer_layers(VisionTransformer):
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        output = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            # # Norm
            # output.append((x[:, 0] - x[:, 0].mean()) / (x[:, 0].std() + 1e-8))
            # # cls token
            # output.append(x[:,0])
            # average pooling
            output.append(torch.mean(x, dim=1))
        # x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0]), output
        else:
            return x[:, 0], x[:, 1], output
    def forward(self, x):
        x, output = self.forward_features(x)
        out = x
        # out = (out - out.mean()) / (out.std() + 1e-8)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, out, output
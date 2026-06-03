import torch
import torch.nn as nn
import torch.nn.functional as F


class AITRFusion(nn.Module):
    def __init__(self, embed_dim=768, num_scalars=9, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.scalar_proj = nn.Sequential(
            nn.Linear(num_scalars, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.type_embedding = nn.Embedding(5, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=1536, dropout=dropout, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, img_emb, txt_emb, scalars):
        B = img_emb.size(0)
        img_times_txt = img_emb * txt_emb
        img_minus_txt = img_emb - txt_emb
        scalar_token  = self.scalar_proj(scalars)
        type_ids  = torch.arange(5, device=img_emb.device)
        type_embs = self.type_embedding(type_ids)
        img_tok    = img_emb       + type_embs[0]
        txt_tok    = txt_emb       + type_embs[1]
        cross_tok  = img_times_txt + type_embs[2]
        diff_tok   = img_minus_txt + type_embs[3]
        scalar_tok = scalar_token  + type_embs[4]
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([
            cls,
            img_tok.unsqueeze(1),
            txt_tok.unsqueeze(1),
            cross_tok.unsqueeze(1),
            diff_tok.unsqueeze(1),
            scalar_tok.unsqueeze(1)
        ], dim=1)
        tokens = tokens.transpose(0, 1)
        out     = self.transformer(tokens)
        cls_out = out[0]
        logits  = self.classifier(cls_out).squeeze(-1)
        return logits


def load_aitr(weights_path, device):
    model  = AITRFusion()
    state  = torch.load(weights_path, map_location=device)
    try:
        result = model.load_state_dict(state, strict=True)
        print(f"Loaded AITR (strict). Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}")
    except Exception as e:
        print(f"strict=True failed: {e}\nRetrying with strict=False")
        result = model.load_state_dict(state, strict=False)
        print(f"Loaded AITR (non-strict). Missing: {result.missing_keys}, Unexpected: {result.unexpected_keys}")
    model.to(device).eval()
    return model

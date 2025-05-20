from transformers import LlamaForCausalLM, MistralForCausalLM
import torch
import torch.nn as nn

def get_embedding_matrix(model):
    if isinstance(model, (LlamaForCausalLM, MistralForCausalLM)):
        return model.model.embed_tokens.weight

    raise ValueError(f'Unknown model type: {type(model)}')


def check_legal_input(tokens, slices):
    assert 'adv_slice' in slices
    assert 'target_slice' in slices

    length = tokens.shape[1]
    adv_start = slices['adv_slice'].start
    adv_stop = slices['adv_slice'].stop
    assert adv_start < adv_stop < length

    target_start = slices['target_slice'].start
    target_stop = slices['target_slice'].stop
    assert target_start < target_stop <= length
    return


def get_illegal_tokens(tokenizer):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    ascii_toks = tuple(set(ascii_toks))
    return ascii_toks


class HighDimFuzzyAttention(nn.Module):
    def __init__(self, hidden_dim=32000, num_heads=8):
        super().__init__()
        # ¿ÉÑ§Ï°²ÎÊý (ÊÊÓ¦ÈýÎ¬½á¹¹)
        self.k = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.a = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(3)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(3)])

        # Í¶Ó°²ã
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x , step_before, max_steps):
        """ÊäÈëxÐÎ×´: (batch_size, seq_len, hidden_dim)"""
        # --- Ä£ºý»¯½×¶Î ---
        # ¼ÆËãÈý¸öÄ£ºý¼¯µÄÁ¥Êô¶È (²¢ÐÐ´¦Àí)
        alpha = torch.tensor(step_before / max_steps, dtype=torch.float32)
        alpha = torch.clamp(alpha, 0.0, 1.0)

        mu = [1 / (1 + torch.exp(self.a[i] * (x + self.b[i]))) for i in range(3)]
        mu[2] = mu[2] * alpha
        # ²»È·¶¨ÐÔÈ¨ÖØ¼ÆËã
        w = [2 * torch.abs(mu_i - 0.5) for mu_i in mu]
        w_c = torch.max(torch.stack(w, dim=0), dim=0)[0]  # (B,S,D)

        # --- Í¨µÀÄ£ºý»¯ ---
        m = x.mean(dim=1, keepdim=True)  # (B,1,D)
        I_c = 1 / (1 + torch.exp(self.k * (m - x)))  # (B,S,D)

        # Ä£ºýÈÚºÏ
        y_c = I_c + w_c - I_c * w_c

        # --- È¥Ä£ºý»¯ ---
        numerator = torch.sum(y_c * x, dim=1, keepdim=True)  # (B,1,D)
        denominator = torch.sum(y_c, dim=1, keepdim=True) + 1e-6
        O_c = numerator / denominator

        # ÌØÕ÷ÔöÇ¿
        return x * self.proj(O_c)  # ¹ã²¥³Ë·¨
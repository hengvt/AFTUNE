import importlib
import inspect
import os

import torch


def use_qwen3_position_ids(model_name):
    return model_name is not None and "qwen3" in model_name.lower()


def qwen3_text_model_position_ids(batch, seq_len, device, past_seen_tokens=0):
    # Qwen3 RoPE expects 1D positions with optional KV offset
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long) + past_seen_tokens
    position_ids = position_ids.unsqueeze(0)
    if batch != 1:
        position_ids = position_ids.expand(batch, -1)
    return position_ids


def position_ids_for_rope(hidden_states, use_qwen3):
    batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    device = hidden_states.device
    if use_qwen3:
        return qwen3_text_model_position_ids(batch, seq_len, device)
    # Llama-style models use simple 0..seq_len-1 positions
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if batch != 1:
        position_ids = position_ids.expand(batch, -1)
    return position_ids


def ensure_rope_state(rotary_emb_module, hidden_states, rope_state):
    # Compute (cos, sin) once per replay step and reuse across decoder layers
    if rope_state["embeddings"] is None:
        text_pos = position_ids_for_rope(hidden_states, rope_state["use_qwen3"])
        rope_state["text_pos"] = text_pos
        rope_state["embeddings"] = rotary_emb_module(hidden_states, text_pos)


def llm_decoder_rope(model, hidden_states):
    # Full-model path: use live global rotary_emb from the training model
    rotary_emb_module = model.model.rotary_emb if hasattr(model.model, "rotary_emb") else None
    use_qwen3 = "qwen3" in model.config.model_type.lower()
    text_pos = position_ids_for_rope(hidden_states, use_qwen3)
    rope_embeddings = None
    if rotary_emb_module is not None:
        rope_embeddings = rotary_emb_module(hidden_states, text_pos)
    return rotary_emb_module, text_pos, rope_embeddings


def load_saved_rotary_emb(recorder, model_name, device):
    # Load rotary_emb module structure saved during finetune for isolated replay
    rotary_emb_path = os.path.join(recorder.module_structure_dir, model_name, "module_structures", "rotary_emb.pt")
    if os.path.isfile(rotary_emb_path):
        return recorder.load_module_structure("rotary_emb", device=device)
    return None


def new_rope_state(rotary_emb_module, model_name):
    if rotary_emb_module is None:
        return None
    return {"text_pos": None, "embeddings": None, "use_qwen3": use_qwen3_position_ids(model_name)}


def expand_layernorm_output_grad(model_name, last_layer, last_output, expected_output_grad):
    if last_layer != "layernorm":
        return expected_output_grad
    if model_name == "dinov2_giant":
        hidden = last_output.shape[-1]
        grad = torch.zeros_like(last_output)
        grad[:, 0, :] = expected_output_grad[:, :hidden]
        grad[:, 1:, :] = expected_output_grad[:, hidden:].unsqueeze(1) / (last_output.shape[1] - 1)
        return grad
    if model_name == "vit_large":
        grad = torch.zeros_like(last_output)
        grad[:, 0, :] = expected_output_grad
        return grad
    return expected_output_grad


def replay_decoder_layer(layer_module, hidden_states, rotary_emb_module, rotary_cache, rope_state):
    if rotary_emb_module is not None and rope_state is not None:
        ensure_rope_state(rotary_emb_module, hidden_states, rope_state)
        output = decoder_layer_forward(
            layer_module, hidden_states, rope_state["text_pos"], rotary_cache, rotary_emb_module,
            position_embeddings_precomputed=rope_state["embeddings"],
        )
    else:
        # Fallback when no saved global rotary_emb exists
        position_ids = position_ids_for_rope(hidden_states, False)
        output = decoder_layer_forward(layer_module, hidden_states, position_ids, rotary_cache, None)
    return output[0] if isinstance(output, tuple) else output


def rotary_embedding_class_for_decoder_layer(layer_module):
    # Resolve HF rotary class from the decoder layer module name
    mod = importlib.import_module(type(layer_module).__module__)
    layer_name = type(layer_module).__name__
    class_candidates = [
        layer_name.replace("DecoderLayer", "TextRotaryEmbedding"),
        layer_name.replace("DecoderLayer", "RotaryEmbedding"),
    ]
    for cls_name in class_candidates:
        if hasattr(mod, cls_name):
            return getattr(mod, cls_name)
    raise AttributeError(f"Cannot resolve rotary embedding class for {layer_name} in module {mod.__name__}")


def decoder_layer_forward(layer_module, hidden_states, position_ids, rotary_cache, rotary_emb_module=None, position_embeddings_precomputed=None):
    sig = inspect.signature(layer_module.forward)
    if "position_embeddings" not in sig.parameters:
        return layer_module(hidden_states, position_ids=position_ids)
    if position_embeddings_precomputed is not None:
        return layer_module(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings_precomputed,
        )
    cache_key = id(rotary_emb_module) if rotary_emb_module is not None else id(layer_module)
    if cache_key not in rotary_cache:
        if rotary_emb_module is not None:
            rotary_cache[cache_key] = rotary_emb_module
        else:
            # Last resort: build a per-layer rotary module from layer config
            rcls = rotary_embedding_class_for_decoder_layer(layer_module)
            if hasattr(layer_module, "self_attn") and hasattr(layer_module.self_attn, "config"):
                config = layer_module.self_attn.config
            elif hasattr(layer_module, "mlp") and hasattr(layer_module.mlp, "config"):
                config = layer_module.mlp.config
            elif hasattr(layer_module, "config"):
                config = layer_module.config
            else:
                raise AttributeError(f"Cannot resolve config for {type(layer_module).__name__}")
            rotary_cache[cache_key] = rcls(config, device=hidden_states.device)
    position_embeddings = rotary_cache[cache_key](hidden_states, position_ids)
    return layer_module(
        hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )

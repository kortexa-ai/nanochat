"""
Penumbra Introspection Layer for nanochat.

Wraps a nanochat Engine to provide runtime activation observation
and weight modification capabilities. All interaction is external
via PyTorch's hook system — no modifications to gpt.py or engine.py.

This file is a Penumbra addition and does not exist upstream.
See penumbra.server/docs/INTROSPECTION.md for architecture details.

Activations are captured as "frames" — one per forward pass, tagged with
metadata (phase, step number, sequence length). Analysis code can then
slice by phase (prefill vs decode) without losing any data.
"""

import torch
import torch.nn.functional as F


class IntrospectableEngine:
    """
    Wraps a nanochat Engine with activation capture and weight modification.

    Activations are stored as a list of frames. Each frame captures one forward
    pass with metadata:
        {"step": 0, "phase": "prefill", "seq_len": 45, "taps": {"layer_0_attn": tensor, ...}}

    Phase is detected automatically: seq_len > 1 = prefill, seq_len == 1 = decode.

    Usage:
        engine = IntrospectableEngine(base_engine)
        engine.tap_layer(0, "attn")
        engine.tap_layer(5, "mlp")

        for tokens, masks in engine.generate(input_tokens):
            prefill = engine.get_prefill()       # first forward pass (the question)
            last    = engine.get_decode_step(-1)  # most recent decode step (the answer)
    """

    def __init__(self, engine):
        self.engine = engine
        self.model = engine.model
        self.tokenizer = engine.tokenizer
        self._hooks = []          # registered hook handles (for cleanup)
        self._frames = []         # list of activation frames (one per forward pass)
        self._current_taps = {}   # accumulates taps for the current forward pass
        self._step = 0            # forward pass counter (reset on clear)
        self._neuron_hooks = []   # neuron intervention hook handles (separate from observation)
        self._neuron_scales = {}  # {layer_idx: {neuron_idx: factor}} for runtime intervention
        self._head_hooks = []     # attention head hook handles
        self._head_scales = {}    # {layer_idx: {head_idx: factor}} for head-level intervention
        self._head_tap_layers = set()  # layers with active head taps
        self._residual_hook = None       # handle for lm_head capture hook
        self._injection_hook = None      # handle for Block[0] injection hook
        self._captured_residual = None   # tensor: last-token residual from lm_head input
        self._injection_vector = None    # tensor to inject into a block
        self._injection_alpha = 0.0      # blending strength
        self._injection_layer = 0        # which block to inject into

    # -------------------------------------------------------------------------
    # Activation observation
    # -------------------------------------------------------------------------

    def tap_layer(self, layer_idx, component="block", clone=False):
        """
        Register a forward hook to capture activations from a specific layer.

        Args:
            layer_idx: which transformer layer (0-indexed)
            component: what to tap — "block" (full output), "attn", or "mlp"
            clone: if True, clone the tensor (costs a memcpy but survives past next forward)
        """
        block = self.model.transformer.h[layer_idx]
        if component == "block":
            module = block
        elif component == "attn":
            module = block.attn
        elif component == "mlp":
            module = block.mlp
        else:
            raise ValueError(f"Unknown component '{component}'. Use 'block', 'attn', or 'mlp'.")

        name = f"layer_{layer_idx}_{component}"

        def _hook(mod, inp, out, _name=name, _clone=clone):
            t = out.detach().clone() if _clone else out.detach()
            self._current_taps[_name] = t
            # Finalize frame when last layer's last component fires.
            # We detect this by checking if this is a block-level hook on the last layer.
            # For non-block hooks, the frame is finalized in _maybe_finalize_frame().

        handle = module.register_forward_hook(_hook)
        self._hooks.append(handle)
        # Register the frame finalizer on the lm_head (fires once per forward pass, after all layers)
        self._ensure_finalizer()
        return name

    def tap_embeddings(self, clone=False):
        """Tap the token embedding layer output."""
        name = "embeddings"

        def _hook(mod, inp, out, _clone=clone):
            t = out.detach().clone() if _clone else out.detach()
            self._current_taps[name] = t

        handle = self.model.transformer.wte.register_forward_hook(_hook)
        self._hooks.append(handle)
        self._ensure_finalizer()
        return name

    def tap_qkv(self, layer_idx, clone=False):
        """
        Tap Q, K, V projections individually (before they enter Flash Attention).

        Returns three activation keys: layer_N_q, layer_N_k, layer_N_v
        """
        attn = self.model.transformer.h[layer_idx].attn
        names = []
        for proj_name, module in [("q", attn.c_q), ("k", attn.c_k), ("v", attn.c_v)]:
            name = f"layer_{layer_idx}_{proj_name}"
            names.append(name)

            def _hook(mod, inp, out, _name=name, _clone=clone):
                t = out.detach().clone() if _clone else out.detach()
                self._current_taps[_name] = t

            handle = module.register_forward_hook(_hook)
            self._hooks.append(handle)
        self._ensure_finalizer()
        return names

    def tap_mlp_hidden(self, layer_idx, clone=False):
        """
        Capture post-activation MLP hidden states (the 5120-dim neuron activations).

        Hooks c_fc output and applies relu^2 to match what c_proj actually sees.
        The captured tensor has shape (B, T, 4*n_embd) — one value per neuron.
        """
        c_fc = self.model.transformer.h[layer_idx].mlp.c_fc
        name = f"layer_{layer_idx}_mlp_hidden"

        def _hook(mod, inp, out, _name=name, _clone=clone):
            # Apply the same activation as MLP.forward: relu(x)^2
            hidden = F.relu(out).square()
            t = hidden.detach().clone() if _clone else hidden.detach()
            self._current_taps[_name] = t

        handle = c_fc.register_forward_hook(_hook)
        self._hooks.append(handle)
        self._ensure_finalizer()
        return name

    def _ensure_finalizer(self):
        """
        Register a hook on lm_head to finalize frames. lm_head fires exactly once
        per forward pass, after all transformer layers, making it the ideal place
        to bundle accumulated taps into a frame.

        Only registers once — subsequent calls are no-ops.
        """
        if hasattr(self, '_finalizer_registered') and self._finalizer_registered:
            return
        self._finalizer_registered = True

        def _finalize(mod, inp, out):
            if not self._current_taps:
                return
            # Detect sequence length from any captured tensor (B, T, ...) or (B, T)
            seq_len = 0
            for t in self._current_taps.values():
                if t.dim() >= 2:
                    seq_len = t.shape[1]
                    break
            phase = "prefill" if seq_len > 1 else "decode"
            frame = {
                "step": self._step,
                "phase": phase,
                "seq_len": seq_len,
                "taps": dict(self._current_taps),
            }
            self._frames.append(frame)
            self._current_taps.clear()
            self._step += 1

        handle = self.model.lm_head.register_forward_hook(_finalize)
        self._hooks.append(handle)

    # ---- Frame accessors ----

    def get_frames(self):
        """Return all captured frames (list of dicts with step/phase/seq_len/taps)."""
        return self._frames

    def get_prefill(self):
        """Return the prefill frame (first forward pass over the full prompt), or None."""
        for f in self._frames:
            if f["phase"] == "prefill":
                return f
        return None

    def get_decode_step(self, n):
        """
        Return a specific decode frame. n is 0-indexed among decode frames.
        Negative indexing works: get_decode_step(-1) returns the last decode frame.
        """
        decode_frames = [f for f in self._frames if f["phase"] == "decode"]
        if not decode_frames:
            return None
        return decode_frames[n]

    def get_decode_frames(self):
        """Return all decode frames."""
        return [f for f in self._frames if f["phase"] == "decode"]

    def get_activations(self):
        """
        Backwards-compatible: return taps dict from the most recent frame.
        Equivalent to get_frames()[-1]["taps"] if frames exist.
        """
        if self._frames:
            return self._frames[-1]["taps"]
        return {}

    def clear_activations(self):
        """Free all captured frames and reset step counter."""
        self._frames.clear()
        self._current_taps.clear()
        self._step = 0

    def remove_hooks(self):
        """Remove all registered hooks (returns engine to zero-overhead state)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        for handle in self._neuron_hooks:
            handle.remove()
        self._neuron_hooks.clear()
        self._neuron_scales.clear()
        for handle in self._head_hooks:
            handle.remove()
        self._head_hooks.clear()
        self._head_scales.clear()
        self._head_tap_layers.clear()
        if self._residual_hook is not None:
            self._residual_hook.remove()
            self._residual_hook = None
        self._captured_residual = None
        self.clear_residual_injection()
        self._frames.clear()
        self._current_taps.clear()
        self._step = 0
        self._finalizer_registered = False

    # -------------------------------------------------------------------------
    # Weight observation
    # -------------------------------------------------------------------------

    def get_weight(self, path):
        """
        Read a weight tensor by dot-path.

        Example paths:
            "transformer.h.0.attn.c_q.weight"
            "transformer.h.5.mlp.c_fc.weight"
            "lm_head.weight"
        """
        parts = path.split(".")
        obj = self.model
        for part in parts:
            if part.isdigit():
                # ModuleDict uses string keys, ModuleList uses int indices
                try:
                    obj = obj[int(part)]
                except (KeyError, TypeError):
                    obj = obj[part]
            else:
                obj = getattr(obj, part)
        return obj.data

    def get_layer_weights(self, layer_idx):
        """Return a dict of all weight tensors in a given layer."""
        block = self.model.transformer.h[layer_idx]
        weights = {}
        for name, param in block.named_parameters():
            weights[f"transformer.h.{layer_idx}.{name}"] = param.data
        return weights

    def get_resid_lambdas(self):
        """Return the per-layer residual scaling parameters."""
        return self.model.resid_lambdas.data

    def get_x0_lambdas(self):
        """Return the per-layer skip-connection scaling parameters."""
        return self.model.x0_lambdas.data

    # -------------------------------------------------------------------------
    # Weight modification
    # -------------------------------------------------------------------------

    def scale_layer(self, layer_idx, factor):
        """
        Scale a layer's contribution to the residual stream.
        Uses the model's built-in resid_lambdas (no extra parameters needed).

        factor > 1.0 = boost, factor < 1.0 = suppress
        """
        self.model.resid_lambdas.data[layer_idx] *= factor

    def set_resid_lambda(self, layer_idx, value):
        """Set a layer's residual scaling to an exact value."""
        self.model.resid_lambdas.data[layer_idx] = value

    def set_x0_lambda(self, layer_idx, value):
        """Set a layer's skip-connection strength to an exact value."""
        self.model.x0_lambdas.data[layer_idx] = value

    def modify_weights(self, deltas):
        """
        Apply additive deltas to weight tensors.

        Args:
            deltas: dict mapping dot-paths to delta tensors.
                    e.g. {"transformer.h.0.attn.c_q.weight": delta_tensor}
        """
        for path, delta in deltas.items():
            weight = self.get_weight(path)
            weight.add_(delta)

    def scale_weights(self, scales):
        """
        Apply multiplicative scaling to weight tensors.

        Args:
            scales: dict mapping dot-paths to scalar or tensor factors.
                    e.g. {"transformer.h.3.attn.c_v.weight": 0.8}
        """
        for path, factor in scales.items():
            weight = self.get_weight(path)
            weight.mul_(factor)

    # -------------------------------------------------------------------------
    # Neuron-level intervention
    # -------------------------------------------------------------------------

    def set_neuron_scales(self, layer_idx, scales):
        """
        Scale specific MLP neurons during forward passes.

        Installs a pre-hook on c_proj that multiplies selected neurons in the
        post-relu^2 activation before down-projection. This is a runtime
        intervention — it doesn't change the stored weights.

        Args:
            layer_idx: which transformer layer
            scales: dict mapping neuron indices to scale factors.
                    e.g. {3847: 0.0, 1024: 2.0}
                    factor 0.0 = ablate, 1.0 = no-op, 2.0 = double
        """
        # Merge with any existing scales for this layer
        if layer_idx not in self._neuron_scales:
            self._neuron_scales[layer_idx] = {}
        self._neuron_scales[layer_idx].update(scales)

        # Remove old hook for this layer if it exists, then install fresh
        self._install_neuron_hook(layer_idx)

    def clear_neuron_scales(self, layer_idx=None):
        """
        Remove neuron scaling interventions.

        Args:
            layer_idx: which layer to clear, or None to clear all layers.
        """
        if layer_idx is not None:
            self._neuron_scales.pop(layer_idx, None)
            self._remove_neuron_hook(layer_idx)
        else:
            self._neuron_scales.clear()
            for handle in self._neuron_hooks:
                handle.remove()
            self._neuron_hooks.clear()

    def get_neuron_scales(self, layer_idx=None):
        """Return current neuron scales. If layer_idx given, return that layer's dict."""
        if layer_idx is not None:
            return dict(self._neuron_scales.get(layer_idx, {}))
        return {l: dict(s) for l, s in self._neuron_scales.items()}

    def _install_neuron_hook(self, layer_idx):
        """Install or replace the neuron scaling pre-hook for a layer."""
        # Remove existing hook for this layer
        self._remove_neuron_hook(layer_idx)

        c_proj = self.model.transformer.h[layer_idx].mlp.c_proj

        def _pre_hook(mod, args, _layer=layer_idx):
            scales = self._neuron_scales.get(_layer)
            if not scales:
                return args
            x = args[0]
            for neuron_idx, factor in scales.items():
                x[:, :, neuron_idx] = x[:, :, neuron_idx] * factor
            return (x,)

        handle = c_proj.register_forward_pre_hook(_pre_hook)
        # Tag the handle so we can find it later
        handle._neuron_layer_idx = layer_idx
        self._neuron_hooks.append(handle)

    def _remove_neuron_hook(self, layer_idx):
        """Remove the neuron scaling hook for a specific layer."""
        remaining = []
        for handle in self._neuron_hooks:
            if getattr(handle, '_neuron_layer_idx', None) == layer_idx:
                handle.remove()
            else:
                remaining.append(handle)
        self._neuron_hooks = remaining

    # -------------------------------------------------------------------------
    # Attention head observation + intervention
    # -------------------------------------------------------------------------

    def tap_attention_heads(self, layer_idx):
        """
        Tap per-head output norms from attention.

        Installs a combined pre-hook on attn.c_proj that captures per-head
        L2 norms BEFORE any scaling intervention (preserving the observation
        invariant). Returns tap name 'layer_N_heads'.

        The tap tensor has shape (B, T, n_head) — one L2 norm per head.
        """
        self._head_tap_layers.add(layer_idx)
        self._install_head_hook(layer_idx)
        self._ensure_finalizer()
        return f"layer_{layer_idx}_heads"

    def set_head_scales(self, layer_idx, scales):
        """
        Scale specific attention heads during forward passes.

        Installs a pre-hook on attn.c_proj that multiplies selected head
        slices in the concatenated attention output before down-projection.

        Args:
            layer_idx: which transformer layer
            scales: dict mapping head indices to scale factors.
                    e.g. {0: 0.0, 5: 2.0}
                    factor 0.0 = ablate, 1.0 = no-op, 2.0 = double
        """
        if layer_idx not in self._head_scales:
            self._head_scales[layer_idx] = {}
        self._head_scales[layer_idx].update(scales)
        self._install_head_hook(layer_idx)

    def clear_head_scales(self, layer_idx=None):
        """
        Remove head scaling interventions.

        Args:
            layer_idx: which layer to clear, or None to clear all layers.
        """
        if layer_idx is not None:
            self._head_scales.pop(layer_idx, None)
            # Reinstall hook if still tapping, otherwise remove
            if layer_idx in self._head_tap_layers:
                self._install_head_hook(layer_idx)
            else:
                self._remove_head_hook(layer_idx)
        else:
            self._head_scales.clear()
            # Reinstall tap-only hooks, remove scale-only hooks
            active = set()
            for handle in self._head_hooks:
                lid = getattr(handle, '_head_layer_idx', None)
                if lid is not None and lid not in self._head_tap_layers:
                    handle.remove()
                else:
                    active.add(lid)
            # Reinstall for tap layers (hooks now have no scales to apply)
            for lid in self._head_tap_layers:
                self._install_head_hook(lid)

    def get_head_scales(self, layer_idx=None):
        """Return current head scales. If layer_idx given, return that layer's dict."""
        if layer_idx is not None:
            return dict(self._head_scales.get(layer_idx, {}))
        return {l: dict(s) for l, s in self._head_scales.items()}

    def _install_head_hook(self, layer_idx):
        """Install or replace the combined head observation+intervention pre-hook."""
        self._remove_head_hook(layer_idx)

        attn = self.model.transformer.h[layer_idx].attn
        c_proj = attn.c_proj
        n_head = attn.n_head
        head_dim = attn.head_dim
        tapping = layer_idx in self._head_tap_layers

        def _pre_hook(mod, args, _layer=layer_idx, _n_head=n_head,
                      _head_dim=head_dim, _tapping=tapping):
            x = args[0]  # (B, T, n_embd)

            if _tapping:
                # Observation: per-head L2 norms BEFORE any scaling
                heads = x.view(x.shape[0], x.shape[1], _n_head, _head_dim)
                norms = heads.norm(dim=-1)  # (B, T, n_head)
                self._current_taps[f"layer_{_layer}_heads"] = norms.detach()

            # Intervention: scale specified heads
            scales = self._head_scales.get(_layer)
            if scales:
                for head_idx, factor in scales.items():
                    start = head_idx * _head_dim
                    end = start + _head_dim
                    x[:, :, start:end] = x[:, :, start:end] * factor
                return (x,)
            return args

        handle = c_proj.register_forward_pre_hook(_pre_hook)
        handle._head_layer_idx = layer_idx
        self._head_hooks.append(handle)

    def _remove_head_hook(self, layer_idx):
        """Remove the head hook for a specific layer."""
        remaining = []
        for handle in self._head_hooks:
            if getattr(handle, '_head_layer_idx', None) == layer_idx:
                handle.remove()
            else:
                remaining.append(handle)
        self._head_hooks = remaining

    # -------------------------------------------------------------------------
    # Inter-inference residual capture + injection
    # -------------------------------------------------------------------------

    def tap_final_residual(self):
        """
        Capture the final normalized residual at lm_head input.

        Installs a pre-hook on lm_head that grabs the last-token residual
        from each forward pass. During decode, this is the single-token
        state (1, n_embd) that becomes the trail's final state at EOS.
        """
        if self._residual_hook is not None:
            return  # already installed

        def _pre_hook(mod, args):
            x = args[0]  # (B, T, n_embd) — post-norm, pre-projection
            self._captured_residual = x[:, -1, :].detach().clone()  # (B, n_embd)

        self._residual_hook = self.model.lm_head.register_forward_pre_hook(_pre_hook)

    def get_captured_residual(self):
        """Return the most recently captured residual vector, or None."""
        return self._captured_residual

    def set_residual_injection(self, vector, alpha=1.0, layer_idx=0):
        """
        Inject a vector additively into a block's input on next forward pass.

        The injection is additive: x = x + alpha * v, preserving the
        prompt's embedding while adding prior-inference signal. The vector
        is broadcast across all token positions.

        Args:
            vector: (n_embd,) or (1, n_embd) tensor to inject
            alpha: blending strength (0.0 = no injection, 1.0 = full add)
            layer_idx: which block to inject into (default 0 = first block)
        """
        self._injection_vector = vector.detach()
        self._injection_alpha = alpha

        # If layer changed or no hook yet, (re)install
        if self._injection_hook is not None and self._injection_layer == layer_idx:
            return  # same layer, hook already installed, will use new vector/alpha
        # Remove old hook if switching layers
        if self._injection_hook is not None:
            self._injection_hook.remove()
            self._injection_hook = None

        self._injection_layer = layer_idx
        block = self.model.transformer.h[layer_idx]

        def _pre_hook(mod, args):
            if self._injection_vector is None or self._injection_alpha == 0.0:
                return args
            x = args[0]  # (B, T, n_embd)
            v = self._injection_vector
            if v.dim() == 1:
                v = v.unsqueeze(0).unsqueeze(0)  # (1, 1, n_embd)
            elif v.dim() == 2:
                v = v.unsqueeze(1)  # (B, 1, n_embd)
            x = x + self._injection_alpha * v
            return (x,) + args[1:]

        self._injection_hook = block.register_forward_pre_hook(_pre_hook)

    def clear_residual_injection(self):
        """Remove injection hook and clear state."""
        self._injection_vector = None
        self._injection_alpha = 0.0
        if self._injection_hook is not None:
            self._injection_hook.remove()
            self._injection_hook = None

    # -------------------------------------------------------------------------
    # Snapshot / restore (for safe experimentation)
    # -------------------------------------------------------------------------

    def snapshot_weights(self, paths=None):
        """
        Take a snapshot of specified weight tensors (or all if paths is None).
        Returns a dict that can be passed to restore_weights().
        """
        if paths is None:
            return {name: param.data.clone() for name, param in self.model.named_parameters()}
        return {path: self.get_weight(path).clone() for path in paths}

    def restore_weights(self, snapshot):
        """Restore weights from a snapshot taken by snapshot_weights()."""
        for path, data in snapshot.items():
            self.get_weight(path).copy_(data)

    # -------------------------------------------------------------------------
    # Pass-through to underlying engine
    # -------------------------------------------------------------------------

    def generate(self, *args, **kwargs):
        """Proxy to engine.generate(). Activations are captured automatically by hooks."""
        return self.engine.generate(*args, **kwargs)

    def generate_batch(self, *args, **kwargs):
        """Proxy to engine.generate_batch()."""
        return self.engine.generate_batch(*args, **kwargs)

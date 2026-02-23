import gc
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from videoprism import models as vp
from utils import decode


# videoprism model wrapper
class VideoPrismWrapper:
    def __init__(
        self,
        model_name="videoprism_lvt_public_v1_base",
        tokenizer_name="c4_en",
        num_frames=16,
        frame_size=288,
    ):
        self.model_name = model_name
        self.num_frames = num_frames
        self.frame_size = frame_size

        self.flax_model = vp.get_model(self.model_name, fprop_dtype=jnp.bfloat16)
        self.loaded_state = vp.load_pretrained_weights(self.model_name)
        self.text_tokenizer = vp.load_text_tokenizer(tokenizer_name)
        self.forward_fn = jax.jit(self._apply_forward)

    def _apply_forward(self, inputs, text_token_ids, text_paddings, train=False):
        return self.flax_model.apply(
            self.loaded_state,
            inputs,
            text_token_ids,
            text_paddings,
            train=train,
        )

    def embed(self, video_path, texts_v0, texts_v1, intervals):
        """Embed a single video over multiple intervals with two caption sets."""
        if not hasattr(intervals, "__len__") or len(intervals) == 0:
            raise ValueError("intervals must be a non-empty sequence.")
        if len(texts_v0) != len(intervals):
            raise ValueError(
                "texts_v0 length must match intervals length: "
                f"{len(texts_v0)} != {len(intervals)}"
            )
        if len(texts_v1) != len(intervals):
            raise ValueError(
                "texts_v1 length must match intervals length: "
                f"{len(texts_v1)} != {len(intervals)}"
            )

        video_embeddings = []
        text_embeddings_v0 = []
        text_embeddings_v1 = []

        for interval, text_v0, text_v1 in zip(intervals, texts_v0, texts_v1):
            if not hasattr(interval, "__len__") or len(interval) != 2:
                raise ValueError(f"each interval must be a length-2 sequence, got {interval}")
            frames, _ = decode(
                video_path,
                num_frames=self.num_frames,
                resolution=(self.frame_size, self.frame_size),
                interval=interval,
            )
            if getattr(frames, "numel", lambda: 0)() == 0:
                raise ValueError(f"no frames decoded for interval {interval}")
            frames_np = frames.numpy()[None, ...]
            del frames

            all_frames = jnp.asarray(frames_np)
            del frames_np

            text_token_ids_v0, text_paddings_v0 = vp.tokenize_texts(
                self.text_tokenizer,
                [text_v0],
                max_length=64,
                add_bos=False,
                canonicalize=False,
            )
            text_token_ids_v0 = jnp.asarray(text_token_ids_v0)
            text_paddings_v0 = jnp.asarray(text_paddings_v0, dtype=jnp.bfloat16)

            text_token_ids_v1, text_paddings_v1 = vp.tokenize_texts(
                self.text_tokenizer,
                [text_v1],
                max_length=64,
                add_bos=False,
                canonicalize=False,
            )
            text_token_ids_v1 = jnp.asarray(text_token_ids_v1)
            text_paddings_v1 = jnp.asarray(text_paddings_v1, dtype=jnp.bfloat16)

            per_video_emb, per_text_emb_v0, _ = self.forward_fn(
                all_frames,
                text_token_ids_v0,
                text_paddings_v0,
            )
            del text_token_ids_v0, text_paddings_v0

            _, per_text_emb_v1, _ = self.forward_fn(
                all_frames,
                text_token_ids_v1,
                text_paddings_v1,
            )
            del all_frames, text_token_ids_v1, text_paddings_v1
            gc.collect()
            if per_video_emb.ndim == 1:
                per_video_emb = per_video_emb[None, ...]
            if per_text_emb_v0.ndim == 1:
                per_text_emb_v0 = per_text_emb_v0[None, ...]
            if per_text_emb_v1.ndim == 1:
                per_text_emb_v1 = per_text_emb_v1[None, ...]
            video_embeddings.append(per_video_emb)
            text_embeddings_v0.append(per_text_emb_v0)
            text_embeddings_v1.append(per_text_emb_v1)

        video_embeddings = jnp.concatenate(video_embeddings, axis=0)
        text_embeddings_v0 = jnp.concatenate(text_embeddings_v0, axis=0)
        text_embeddings_v1 = jnp.concatenate(text_embeddings_v1, axis=0)

        return video_embeddings, text_embeddings_v0, text_embeddings_v1


def compute_paired_softmax(video_embeddings, text_embeddings_v0, text_embeddings_v1, temperature=0.01):
    """Compute row-wise softmax over paired text candidates for each video embedding."""
    video_norm = video_embeddings / jnp.linalg.norm(video_embeddings, axis=-1, keepdims=True)
    text_v0_norm = text_embeddings_v0 / jnp.linalg.norm(text_embeddings_v0, axis=-1, keepdims=True)
    text_v1_norm = text_embeddings_v1 / jnp.linalg.norm(text_embeddings_v1, axis=-1, keepdims=True)

    logits = jnp.concatenate(
        [
            jnp.sum(video_norm * text_v0_norm, axis=-1)[..., None],
            jnp.sum(video_norm * text_v1_norm, axis=-1)[..., None],
        ],
        axis=-1,
    )
    return jax.nn.softmax(logits / temperature, axis=-1)


def embed_and_score(video_path, texts_v0, texts_v1, intervals: Sequence[Sequence[float]], temperature=0.01):
    """Run the model on a video and two text lists, then return softmax probabilities."""
    video_embeddings, text_embeddings_v0, text_embeddings_v1 = _videoprism.embed(
        video_path,
        texts_v0,
        texts_v1,
        intervals,
    )
    softmax_probs = compute_paired_softmax(
        video_embeddings,
        text_embeddings_v0,
        text_embeddings_v1,
        temperature=temperature,
    )
    softmax_probs = np.asarray(softmax_probs)
    del video_embeddings, text_embeddings_v0, text_embeddings_v1
    gc.collect()
    return softmax_probs


_videoprism = VideoPrismWrapper()
MODEL_NAME = _videoprism.model_name
NUM_FRAMES = _videoprism.num_frames
FRAME_SIZE = _videoprism.frame_size

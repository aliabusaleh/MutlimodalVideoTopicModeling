from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.asr.whisper_transcribe import transcribe_with_whisper
from src.audio.speaker_embeddings import embed_segments
from src.clustering.cluster import run_umap_hdbscan
from src.config import ensure_dir, load_config
from src.evaluation.metrics import compute_numeric_metrics
from src.fusion.attention_gated_learning import MultiModalCoAttention, run
from src.fusion.co_sim_gated import (
    similarity_gated_concatenation,
    similarity_gated_concatenation_multimodal,
    naive_concatenation, _align_to_common_dim,
)
from src.preprocess.audio_extractor import extract_wav
from src.topic.bertopic_runner import encode_text_segments, run_bertopic
from src.topic.summarize_local import summarize_topics_extractively
from src.topic.topic_merge import reduce_similar_topics
from src.utils import load_json, save_json, save_numpy
from src.video.clip_embeddings import embed_frames_per_segment, to_segment_matrix
from src.video.frame_selector import extract_segment_frames
from src.visualization.timeline_plotly import build_timeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video topic modeling prototype")
    parser.add_argument("--video", default=None, help="Path to a single input video")
    parser.add_argument("--video-dir", default="data", help="Directory to search for MP4 files")
    parser.add_argument("--all-mp4", action="store_true", help="Process all MP4 files found under --video-dir")
    parser.add_argument(
        "--allowed-videos-file",
        default=None,
        help="Optional path to a file listing allowed video file paths (one per line). If set, only videos in this list will be processed.",
    )
    parser.add_argument("--datasets", nargs="+", default=[], help="List of dataset paths to process separately with per-dataset metrics")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--skip-existing", default=False, help="Skip processing videos that have already been processed based on presence of output files")
    parser.add_argument(
        "--stages",
        default="audio,asr,speaker,frames,clip,visual_cluster,fusion,topic,merge,summary,metrics,viz",
        help="Comma-separated stage list",
    )
    parser.add_argument("--output-dir", default=None, help="Optional output directory override")
    return parser.parse_args()


def apply_speaker_labels(segments: list[dict[str, Any]], labels: np.ndarray) -> None:
    for seg, label in zip(segments, labels):
        seg["speaker"] = "unknown" if int(label) == -1 else f"speaker_{int(label)}"


def apply_topic_labels(segments: list[dict[str, Any]], topics: list[int]) -> None:
    for seg, topic in zip(segments, topics):
        seg["topic"] = int(topic)


def _flatten_numeric(prefix: str, payload: Any, out: dict[str, float]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            nested = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(nested, value, out)
        return
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        out[prefix] = float(payload)


def _discover_videos(args: argparse.Namespace) -> list[Path]:
    videos: list[Path] = []
    # If an allowed-videos-file is provided, read that and return those paths (after validation).
    if getattr(args, "allowed_videos_file", None):
        allowed_file = Path(args.allowed_videos_file)
        if not allowed_file.exists():
            raise RuntimeError(f"Allowed videos file not found: {allowed_file}")
        with allowed_file.open("r", encoding="utf-8") as handle:
            lines = [l.strip() for l in handle.readlines() if l.strip()]
        videos = [Path(l) for l in lines]
        missing = [p for p in videos if not p.exists()]
        if missing:
            raise RuntimeError(f"Video file not found: {missing[0]}")
        return videos

    if args.all_mp4:
        root = Path(args.video_dir)
        if not root.exists():
            raise RuntimeError(f"Video directory not found: {root}")
        root = root.resolve()
        # Scan only one level deep: root/*.mp4 and root/*/*.mp4.
        direct_files = [p for p in root.glob("*.mp4") if p.is_file()]
        nested_files = [
            p
            for child in root.iterdir()
            if child.is_dir()
            for p in child.glob("*.mp4")
            if p.is_file()
        ]
        videos = sorted(direct_files + nested_files)
    elif args.video:
        videos = [Path(args.video)]

    if not videos:
        raise RuntimeError("No videos found. Use --video <file> or --all-mp4 --video-dir <dir>.")

    missing = [p for p in videos if not p.exists()]
    if missing:
        raise RuntimeError(f"Video file not found: {missing[0]}")

    return videos


def _run_stem_for_video(video_path: Path, args: argparse.Namespace) -> str:
    stem = video_path.stem
    if not args.all_mp4:
        return stem

    root = Path(args.video_dir).resolve()
    video_resolved = video_path.resolve()
    parent = video_resolved.parent

    # If discovered from an immediate subfolder, prefix parent folder name.
    if parent != root and parent.parent == root:
        return f"{parent.name}_{stem}"
    return stem


def _is_already_processed(
    run_stem: str,
    stages: set[str],
    cfg: Any,
    output_root_override: Path | None,
    is_batch: bool,
) -> bool:
    processed_root = Path(cfg.get("paths", "processed_dir", default="data/processed"))
    output_root = output_root_override or Path(cfg.get("paths", "output_dir", default="data/output"))

    processed_dir = processed_root / run_stem
    if output_root_override is None:
        output_dir = output_root / run_stem
    else:
        output_dir = (output_root / run_stem) if is_batch else output_root

    markers: list[Path] = []
    if "audio" in stages:
        markers.append(processed_dir / "audio.wav")
    if "asr" in stages:
        markers.append(processed_dir / "segments.json")
    if "speaker" in stages:
        markers.extend(
            [
                processed_dir / "audio_embeddings.npy",
                processed_dir / "audio_umap.npy",
            ]
        )
    if "frames" in stages:
        markers.append(processed_dir / "segment_frames.json")
    if "clip" in stages:
        markers.append(processed_dir / "visual_embeddings.npy")
    if "visual_cluster" in stages:
        markers.append(processed_dir / "visual_umap.npy")
    if "fusion" in stages:
        markers.extend(
            [
                processed_dir / "fused_embeddings.npy",
                processed_dir / "fused_umap.npy",
            ]
        )
    if "topic" in stages:
        markers.extend(
            [
                output_dir / "segments_enriched.json",
                output_dir / "topic_info.json",
            ]
        )
    if "merge" in stages:
        markers.append(output_dir / "topic_info_merged.json")
    if "summary" in stages:
        markers.append(output_dir / "topic_summaries.json")
    if "metrics" in stages:
        topic_embedding_source = str(cfg.get("topic", "embedding_source", default="text")).strip().lower()
        if "topic" in stages and topic_embedding_source == "both":
            markers.extend(
                [
                    output_dir / "metrics_text.json",
                    output_dir / "metrics_multimodal.json",
                    output_dir / "metrics_comparison.json",
                ]
            )
        else:
            markers.append(output_dir / "metrics.json")
    if "viz" in stages:
        markers.append(output_dir / "timeline.html")

    if not markers:
        return False
    return all(marker.exists() for marker in markers)


def _aggregate_metrics(metric_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    flattened: list[dict[str, float]] = []
    for payload in metric_payloads:
        metric_values = payload.get("metrics", payload) if isinstance(payload, dict) else payload
        row: dict[str, float] = {}
        _flatten_numeric("", metric_values, row)
        flattened.append(row)

    keys = sorted({k for row in flattened for k in row.keys()})
    mean: dict[str, float] = {}
    summation: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in flattened if key in row]
        if not values:
            continue
        mean[key] = float(np.mean(values))
        if key.endswith("num_segments") or key.endswith("total_duration_s") or key.endswith("noise_segment_count"):
            summation[key] = float(np.sum(values))

    return {
        "video_count": len(metric_payloads),
        "mean": mean,
        "sum": summation,
    }


def _metrics_with_context(
    metrics: dict[str, Any],
    *,
    timestamp_utc: str,
    config_path: str,
    cfg: Any,
    video_path: Path,
    run_stem: str,
    source_name: str,
    stages: set[str],
) -> dict[str, Any]:
    return {
        "meta": {
            "timestamp_utc": timestamp_utc,
            "config_path": config_path,
            "config": cfg.raw,
            "video": str(video_path),
            "run_stem": run_stem,
            "source": source_name,
            "stages": sorted(stages),
        },
        "metrics": metrics,
    }


def _run_video_pipeline(
    video_path: Path,
    run_stem: str,
    cfg: Any,
    stages: set[str],
    output_root_override: Path | None,
    is_batch: bool,
    timestamp_utc: str,
    config_path: str,
) -> dict[str, Any]:
    stem = run_stem
    # if path contains 2026-03-22 skip it
    if "2026-03-22" in video_path.as_posix() or "2026-04-09" in video_path.as_posix() or "2026-03-30" in video_path.as_posix():
        print(f"Skipping video: {video_path}")
        return {}
    processed_root = Path(cfg.get("paths", "processed_dir", default="data/processed"))
    output_root = output_root_override or Path(cfg.get("paths", "output_dir", default="data/output"))

    processed_dir = processed_root / stem
    if output_root_override is None:
        output_dir = output_root / stem
    else:
        output_dir = (output_root / stem) if is_batch else output_root
    ensure_dir(processed_dir)
    ensure_dir(output_dir)

    wav_path = processed_dir / "audio.wav"
    segments_path = processed_dir / "segments.json"

    segments: list[dict[str, Any]] = []

    if "audio" in stages:
        extract_wav(
            video_path=video_path,
            wav_path=wav_path,
            sample_rate=int(cfg.get("audio", "sample_rate", default=16000)),
            channels=int(cfg.get("audio", "channels", default=1)),
            codec=str(cfg.get("audio", "codec", default="pcm_s16le")),
        )

    if "asr" in stages:
        segments = transcribe_with_whisper(
            wav_path=wav_path,
            model_name=str(cfg.get("asr", "model_name", default="large-v3")),
            device=str(cfg.get("asr", "device", default="cuda")),
            language=cfg.get("asr", "language", default=None),
        )
        save_json(segments_path, segments)
    else:
        if segments_path.exists():
            segments = load_json(segments_path)
        else:
            raise RuntimeError("ASR segments not found. Include stage 'asr' or provide segments.json.")

    if not segments:
        raise RuntimeError("No ASR segments found. Include stage 'asr' and verify audio content.")

    if "speaker" in stages:
        audio_vectors = embed_segments(
            wav_path=wav_path,
            segments=segments,
            backend=str(cfg.get("speaker", "backend", default="pyannote_fallback")),
            model_name=str(cfg.get("speaker", "model_name", default="pyannote/embedding")),
            hf_token_env=str(cfg.get("speaker", "use_auth_token_env", default="HF_TOKEN")),
            clap_model_name=str(cfg.get("speaker", "clap_model_name", default="laion/clap-htsat-unfused")),
            clap_device=str(cfg.get("speaker", "clap_device", default="cuda")),
            clap_sampling_rate=int(cfg.get("speaker", "clap_sampling_rate", default=48000)),
            fallback_bins=int(cfg.get("speaker", "fallback_mfcc_bins", default=64)),
        )
        save_numpy(processed_dir / "audio_embeddings.npy", audio_vectors)

        audio_cluster = run_umap_hdbscan(
            audio_vectors,
            umap_n_neighbors=int(cfg.get("cluster", "umap_n_neighbors", default=15)),
            umap_n_components=int(cfg.get("cluster", "umap_n_components", default=8)),
            umap_metric=str(cfg.get("cluster", "umap_metric", default="cosine")),
            hdbscan_min_cluster_size=int(cfg.get("cluster", "hdbscan_min_cluster_size", default=5)),
            hdbscan_metric=str(cfg.get("cluster", "hdbscan_metric", default="euclidean")),
            random_state=cfg.seed,
        )
        apply_speaker_labels(segments, audio_cluster["labels"])
        save_numpy(processed_dir / "audio_umap.npy", audio_cluster["reduced"])

    segment_to_frames: dict[int, list[str]] = {}
    if "frames" in stages:
        segment_to_frames = extract_segment_frames(
            video_path=video_path,
            segments=segments,
            out_dir=processed_dir / "frames",
            top_k=int(cfg.get("frames", "top_k_per_segment", default=3)),
            candidate_multiplier=int(cfg.get("frames", "candidate_multiplier", default=4)),
            min_candidate_frames=int(cfg.get("frames", "min_candidate_frames", default=8)),
            dedup_similarity_threshold=float(cfg.get("frames", "dedup_similarity_threshold", default=0.96)),
            diversity_lambda=float(cfg.get("frames", "diversity_lambda", default=0.35)),
            image_format=str(cfg.get("frames", "image_format", default="jpg")),
            max_width=int(cfg.get("frames", "max_width", default=640)),
        )
        save_json(processed_dir / "segment_frames.json", segment_to_frames)

    visual_vectors = None
    if "clip" in stages:
        if "frames" not in stages:
                frames_path = processed_dir / "segment_frames.json"
                if not frames_path.exists():
                    raise RuntimeError("Segment frames not found. Include stage 'frames' before 'clip'.")
                segment_to_frames = load_json(frames_path)
        if not segment_to_frames:
            raise RuntimeError("No frames found. Include stage 'frames' before 'clip'.")
        clip_backend = str(cfg.get("clip", "backend", default="clip"))
        visual_map = embed_frames_per_segment(
            segment_to_frames=segment_to_frames,
            model_name=str(cfg.get("clip", "model_name", default="ViT-B-32")),
            pretrained=str(cfg.get("clip", "pretrained", default="laion2b_s34b_b79k")),
            signlip_model_name=str(cfg.get("clip", "signlip_model_name", default="ViT-SO400M-14-SigLIP-384")),
            signlip_pretrained=str(cfg.get("clip", "signlip_pretrained", default="webli")),
            device=str(cfg.get("clip", "device", default="cuda")),
            backend=clip_backend,
            vllm_base_url=str(cfg.get("clip", "vllm_base_url", default="http://localhost:8000/v1")),
            vllm_endpoint=str(cfg.get("clip", "vllm_endpoint", default="/embeddings")),
            vllm_model=str(cfg.get("clip", "vllm_model", default="Qwen/Qwen2.5-VL-7B-Instruct")),
            vllm_api_key_env=cfg.get("clip", "vllm_api_key_env", default="VLLM_API_KEY"),
            vllm_timeout_s=float(cfg.get("clip", "vllm_timeout_s", default=60.0)),
            fallback_embedding_dim=int(cfg.get("clip", "fallback_embedding_dim", default=512)),
        )
        visual_vectors = to_segment_matrix(segments, visual_map)
        save_numpy(processed_dir / "visual_embeddings.npy", visual_vectors)

    if "visual_cluster" in stages and visual_vectors is not None:
        visual_cluster = run_umap_hdbscan(
            visual_vectors,
            umap_n_neighbors=int(cfg.get("cluster", "umap_n_neighbors", default=15)),
            umap_n_components=int(cfg.get("cluster", "umap_n_components", default=8)),
            umap_metric=str(cfg.get("cluster", "umap_metric", default="cosine")),
            hdbscan_min_cluster_size=int(cfg.get("cluster", "hdbscan_min_cluster_size", default=5)),
            hdbscan_metric=str(cfg.get("cluster", "hdbscan_metric", default="euclidean")),
            random_state=cfg.seed,
        )
        save_numpy(processed_dir / "visual_umap.npy", visual_cluster["reduced"])

    if "fusion" in stages:
        audio_vectors = np.load(processed_dir / "audio_embeddings.npy")
        if visual_vectors is None:
            visual_vectors = np.load(processed_dir / "visual_embeddings.npy")

        # TODO -  gated fusion
        fused = similarity_gated_concatenation(
            audio_vectors,
            visual_vectors,
            weight_audio=float(cfg.get("fusion", "weight_audio", default=0.5)),
            weight_visual=float(cfg.get("fusion", "weight_visual", default=0.5)),
        )
        save_numpy(processed_dir / "fused_embeddings.npy", fused)

        fused_cluster = run_umap_hdbscan(
            fused,
            umap_n_neighbors=int(cfg.get("cluster", "umap_n_neighbors", default=15)),
            umap_n_components=int(cfg.get("cluster", "umap_n_components", default=8)),
            umap_metric=str(cfg.get("cluster", "umap_metric", default="cosine")),
            hdbscan_min_cluster_size=int(cfg.get("cluster", "hdbscan_min_cluster_size", default=5)),
            hdbscan_metric=str(cfg.get("cluster", "hdbscan_metric", default="euclidean")),
            random_state=cfg.seed,
        )
        save_numpy(processed_dir / "fused_umap.npy", fused_cluster["reduced"])

    topic_model = None
    topics: list[int] = []
    topic_outputs: dict[str, dict[str, Any]] = {}
    if "topic" in stages:
        topic_mode = str(cfg.get("topic", "mode", default="unsupervised")).strip().lower()
        topic_embedding_source = str(cfg.get("topic", "embedding_source", default="text")).strip().lower()
        raw_seed_topics = cfg.get("topic", "seed_topic_list", default=[])
        seed_topic_list = None
        if topic_mode in {"guided", "semi-supervised", "semi_supervised"} and isinstance(raw_seed_topics, list) and raw_seed_topics:
            seed_topic_list = raw_seed_topics

        multimodal_embeddings = None
        tv_concat = ta_concat = tav_concat = None
        # TODO - Attention model
        if topic_embedding_source in {"multimodal", "coattention", "co_attention", "both"}:
            audio_vectors = np.load(processed_dir / "audio_embeddings.npy")
            if visual_vectors is None:
                visual_vectors = np.load(processed_dir / "visual_embeddings.npy")
            text_vectors = encode_text_segments(
                segments,
                sentence_model_name=str(cfg.get("topic", "sentence_model", default="all-mpnet-base-v2")),
                device=str(cfg.get("topic", "sentence_model_device", default="cuda")),
                show_progress=True,
            )

            # multimodal_embeddings = similarity_gated_concatenation_multimodal(
            #     text_vectors=text_vectors,
            #     audio_vectors=audio_vectors,
            #     visual_vectors=visual_vectors,
            #     weight_text=float(cfg.get("topic", "weight_text", default=0.34)),
            #     weight_audio=float(cfg.get("topic", "weight_audio", default=0.33)),
            #     weight_visual=float(cfg.get("topic", "weight_visual", default=0.33)),
            # )


            print("using the new co attention implementation ")
            text_vectors, audio_vectors, visual_vectors = _align_to_common_dim(text_vectors, audio_vectors,
                                                                              visual_vectors)

            # model needs torch tensors no np arrays
            text_vectors = torch.tensor(text_vectors, dtype=torch.float32)
            audio_vectors = torch.tensor(audio_vectors, dtype=torch.float32)
            visual_vectors = torch.tensor(visual_vectors, dtype=torch.float32)
            print("in the run: ")
            print(text_vectors.shape, audio_vectors.shape, visual_vectors.shape)  # all should match


            # initiate model and self supervised training
            model = run(text_vectors, audio_vectors, visual_vectors)

            multimodal_embeddings = model(text_vectors, audio_vectors, visual_vectors)
            multimodal_embeddings = multimodal_embeddings.detach().numpy()


            save_numpy(processed_dir / "multimodal_topic_embeddings.npy", multimodal_embeddings)

            # Baseline: simple L2-normalized concatenations (no gate, no interactions)
            try:
                tv_concat = naive_concatenation(text_vectors, visual_vectors)
                save_numpy(processed_dir / "multimodal_topic_embeddings_concat_text_visual.npy", tv_concat)
            except ValueError:
                tv_concat = None

            try:
                ta_concat = naive_concatenation(text_vectors, audio_vectors)
                save_numpy(processed_dir / "multimodal_topic_embeddings_concat_text_audio.npy", ta_concat)
            except ValueError:
                ta_concat = None

            try:
                tav_concat = naive_concatenation(text_vectors, audio_vectors, visual_vectors)
                save_numpy(processed_dir / "multimodal_topic_embeddings_concat_text_audio_visual.npy", tav_concat)
            except ValueError:
                tav_concat = None

        sources: list[tuple[str, np.ndarray | None]]
        # Build sources to run BERTopic on. If multimodal embeddings were computed,
        # include the gated multimodal representation and simple concatenation baselines.
        sources: list[tuple[str, np.ndarray | None]]
        if topic_embedding_source == "both":
            sources = [("text", None)]
            # fall through to append multimodal variants below
        elif topic_embedding_source in {"multimodal", "coattention", "co_attention"}:
            sources = []
        else:
            sources = [("text", None)]

        if multimodal_embeddings is not None:
            sources.append(("multimodal", multimodal_embeddings))
            if tv_concat is not None:
                sources.append(("concat_text_visual", tv_concat))
            if ta_concat is not None:
                sources.append(("concat_text_audio", ta_concat))
            if tav_concat is not None:
                sources.append(("concat_text_audio_visual", tav_concat))

        for source_name, source_embeddings in sources:
            source_model, source_topics, _ = run_bertopic(
                segments,
                sentence_model_name=str(cfg.get("topic", "sentence_model", default="all-mpnet-base-v2")),
                min_topic_size=int(cfg.get("topic   ", "min_topic_size", default=5)),
                seed_topic_list=seed_topic_list,
                precomputed_embeddings=source_embeddings,
                device=str(cfg.get("topic", "sentence_model_device", default="cuda")),
            )
            source_segments = [dict(seg) for seg in segments]
            apply_topic_labels(source_segments, source_topics)

            topic_info = source_model.get_topic_info().to_dict(orient="records")
            suffix = "" if len(sources) == 1 else f"_{source_name}"
            save_json(output_dir / f"topic_info{suffix}.json", topic_info)
            save_json(output_dir / f"segments_enriched{suffix}.json", source_segments)

            topic_outputs[source_name] = {
                "topic_model": source_model,
                "topics": source_topics,
                "segments": source_segments,
                "topic_info": topic_info,
                "topic_info_path": str(output_dir / f"topic_info{suffix}.json"),
            }

        selected_source = "multimodal" if "multimodal" in topic_outputs else next(iter(topic_outputs.keys()))
        topic_model = topic_outputs[selected_source]["topic_model"]
        topics = topic_outputs[selected_source]["topics"]
        segments = topic_outputs[selected_source]["segments"]

    if "merge" in stages and topic_model is not None:
        docs = [s["text"] for s in segments]
        topic_model = reduce_similar_topics(topic_model, docs)
        try:
            merged = topic_model.get_topic_info().to_dict(orient="records")
            save_json(output_dir / "topic_info_merged.json", merged)
        except Exception:
            pass

    if "summary" in stages and topics:
        summaries = summarize_topics_extractively(
            segments,
            topics,
            max_sentences_per_topic=int(cfg.get("summary", "max_sentences_per_topic", default=4)),
        )
        save_json(output_dir / "topic_summaries.json", summaries)

    if "topic" not in stages:
        save_json(output_dir / "segments_enriched.json", segments)
    else:
        selected_segments = topic_outputs[selected_source]["segments"]
        save_json(output_dir / "segments_enriched.json", selected_segments)
        selected_topic_info = topic_outputs[selected_source]["topic_info"]
        save_json(output_dir / "topic_info.json", selected_topic_info)

    metric_files: dict[str, str] = {}
    if "metrics" in stages:
        if topic_outputs:
            for source_name, payload in topic_outputs.items():
                suffix = "" if len(topic_outputs) == 1 else f"_{source_name}"
                metric_path = output_dir / f"metrics{suffix}.json"
                metrics = compute_numeric_metrics(
                    segments=payload["segments"],
                    processed_dir=processed_dir,
                    output_dir=output_dir,
                    topic_info_override=payload["topic_info"],
                )
                metric_payload = _metrics_with_context(
                    metrics,
                    timestamp_utc=timestamp_utc,
                    config_path=config_path,
                    cfg=cfg,
                    video_path=video_path,
                    run_stem=run_stem,
                    source_name=source_name,
                    stages=stages,
                )
                save_json(metric_path, metric_payload)
                metric_files[source_name] = str(metric_path)

            if "text" in metric_files and "multimodal" in metric_files:
                text_payload = load_json(metric_files["text"])
                multimodal_payload = load_json(metric_files["multimodal"])
                text_metrics = text_payload.get("metrics", text_payload)
                multimodal_metrics = multimodal_payload.get("metrics", multimodal_payload)
                comparison = {
                    "noise_ratio_delta_multimodal_minus_text": float(multimodal_metrics.get("noise_ratio", 0.0) - text_metrics.get("noise_ratio", 0.0)),
                    "topic_coherence_npmi_delta_multimodal_minus_text": (
                        None
                        if text_metrics.get("topic_coherence_npmi") is None or multimodal_metrics.get("topic_coherence_npmi") is None
                        else float(multimodal_metrics["topic_coherence_npmi"] - text_metrics["topic_coherence_npmi"])
                    ),
                    "topic_entropy_delta_multimodal_minus_text": float(
                        multimodal_metrics.get("topic_entropy_normalized", 0.0)
                        - text_metrics.get("topic_entropy_normalized", 0.0)
                    ),
                }
                save_json(output_dir / "metrics_comparison.json", comparison)
        else:
            metric_path = output_dir / "metrics.json"
            metrics = compute_numeric_metrics(
                segments=segments,
                processed_dir=processed_dir,
                output_dir=output_dir,
            )
            metric_payload = _metrics_with_context(
                metrics,
                timestamp_utc=timestamp_utc,
                config_path=config_path,
                cfg=cfg,
                video_path=video_path,
                run_stem=run_stem,
                source_name="default",
                stages=stages,
            )
            save_json(metric_path, metric_payload)
            metric_files["default"] = str(metric_path)

    if "viz" in stages:
        build_timeline(
            segments,
            out_html=output_dir / "timeline.html",
            title=str(cfg.get("visualization", "timeline_title", default="Topic Timeline by Speaker")),
            processed_dir=processed_dir,
            output_dir=output_dir,
        )

    return {
        "video": str(video_path),
        "stem": stem,
        "output_dir": str(output_dir),
        "metrics_files": metric_files,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    config_path = str(Path(args.config).resolve())
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    is_batch = args.all_mp4 or len(_discover_videos(args)) > 1
    output_root_override = Path(args.output_dir) if args.output_dir else None

    # Handle multiple datasets if provided
    if args.datasets:
        datasets_results: dict[str, dict[str, Any]] = {}
        output_root = output_root_override or Path(cfg.get("paths", "output_dir", default="data/output"))

        for dataset_path in args.datasets:
            dataset_name = Path(dataset_path).name or Path(dataset_path).stem
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset_name} ({dataset_path})")
            print(f"{'='*60}")

            # Override video-dir for this dataset
            original_video_dir = args.video_dir
            args.video_dir = dataset_path
            args.all_mp4 = True

            try:
                videos = _discover_videos(args)
            except Exception as e:
                print(f"Error discovering videos in {dataset_path}: {e}")
                args.video_dir = original_video_dir
                continue

            is_batch_dataset = len(videos) > 1
            dataset_output_root = output_root / dataset_name if output_root_override is None else output_root

            runs: list[dict[str, Any]] = []
            for video_path in videos:
                run_stem = _run_stem_for_video(video_path, args)
                if args.skip_existing:
                    if _is_already_processed(
                        run_stem=run_stem,
                        stages=stages,
                        cfg=cfg,
                        output_root_override=dataset_output_root if output_root_override is None else output_root_override,
                        is_batch=is_batch_dataset,
                    ):
                        print(f"[SKIP] Already processed: {video_path} (run stem: {run_stem})")
                        continue

                run_info = _run_video_pipeline(
                    video_path=video_path,
                    run_stem=run_stem,
                    cfg=cfg,
                    stages=stages,
                    output_root_override=dataset_output_root if output_root_override is None else output_root_override,
                    is_batch=is_batch_dataset,
                    timestamp_utc=timestamp_utc,
                    config_path=config_path,
                )
                runs.append(run_info)

            # Aggregate metrics for this dataset
            if "metrics" in stages and runs:
                per_source_metrics: dict[str, list[dict[str, Any]]] = {}
                for run in runs:
                    for source_name, metric_file in run.get("metrics_files", {}).items():
                        payload = load_json(metric_file)
                        per_source_metrics.setdefault(source_name, []).append(payload)

                dataset_aggregate = {
                    "meta": {
                        "timestamp_utc": timestamp_utc,
                        "config_path": config_path,
                        "config": cfg.raw,
                        "stages": sorted(stages),
                        "dataset_name": dataset_name,
                        "dataset_path": dataset_path,
                    },
                    "video_count": len(runs),
                    "videos": [{"stem": run["stem"], "output_dir": run["output_dir"]} for run in runs if run.get("stem") and run.get("output_dir")],
                    "sources": {},
                }
                for source_name, payloads in per_source_metrics.items():
                    dataset_aggregate["sources"][source_name] = _aggregate_metrics(payloads)

                datasets_results[dataset_name] = dataset_aggregate

                # Save per-dataset metrics
                ensure_dir(dataset_output_root)
                save_json(dataset_output_root / f"metrics_{dataset_name}.json", dataset_aggregate)
                safe_timestamp = timestamp_utc.replace(":", "-")
                save_json(dataset_output_root / f"metrics_{dataset_name}_{safe_timestamp}.json", dataset_aggregate)

            args.video_dir = original_video_dir

        # Aggregate all datasets
        if datasets_results:
            all_datasets_aggregate = {
                "meta": {
                    "timestamp_utc": timestamp_utc,
                    "config_path": config_path,
                    "config": cfg.raw,
                    "stages": sorted(stages),
                    "num_datasets": len(datasets_results),
                },
                "datasets": datasets_results,
            }
            save_json(output_root / "metrics_all_datasets.json", all_datasets_aggregate)
            safe_timestamp = timestamp_utc.replace(":", "-")
            save_json(output_root / f"metrics_all_datasets_{safe_timestamp}.json", all_datasets_aggregate)
    else:
        # Original single dataset/video logic
        videos = _discover_videos(args)

        runs: list[dict[str, Any]] = []
        for video_path in videos:
            
            try:
                run_stem = _run_stem_for_video(video_path, args)
                if args.skip_existing == True:
                    if _is_already_processed(
                        run_stem=run_stem,
                        stages=stages,
                        cfg=cfg,
                        output_root_override=output_root_override,
                        is_batch=is_batch,
                    ):
                        print(f"[SKIP] Already processed: {video_path} (run stem: {run_stem})")
                        continue

                run_info = _run_video_pipeline(
                    video_path=video_path,
                    run_stem=run_stem,
                    cfg=cfg,
                    stages=stages,
                    output_root_override=output_root_override,
                    is_batch=is_batch,
                    timestamp_utc=timestamp_utc,
                    config_path=config_path,
                )
                runs.append(run_info)
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                continue
            
        if "metrics" in stages and runs:
            output_root = output_root_override or Path(cfg.get("paths", "output_dir", default="data/output"))
            per_source_metrics: dict[str, list[dict[str, Any]]] = {}
            for run in runs:
                for source_name, metric_file in run.get("metrics_files", {}).items():
                    payload = load_json(metric_file)
                    per_source_metrics.setdefault(source_name, []).append(payload)

            aggregate = {
                "meta": {
                    "timestamp_utc": timestamp_utc,
                    "config_path": config_path,
                    "config": cfg.raw,
                    "stages": sorted(stages),
                },
                "video_count": len(runs),
                "videos": [{"stem": run["stem"], "output_dir": run["output_dir"]} for run in runs if run.get("stem") and run.get("output_dir")],
                "sources": {},
            }
            for source_name, payloads in per_source_metrics.items():
                aggregate["sources"][source_name] = _aggregate_metrics(payloads)

            save_json(output_root / "metrics_all_videos.json", aggregate)
            safe_timestamp = timestamp_utc.replace(":", "-")
            save_json(output_root / f"metrics_all_videos_{safe_timestamp}.json", aggregate)


if __name__ == "__main__":
    main()

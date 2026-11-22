"""Reviewed adapters from existing execution contracts to common task profiles."""

from __future__ import annotations

from oryn_protocol.task_contracts import TASK_CONTRACTS, make_profile, validate_profile


def profile_for_execution(
    profile: dict, config: dict | None = None, *, contract_id=None
) -> dict | None:
    if (
        contract_id != "chat-v2"
        and profile.get("taskProfile", {}).get("contract", {}).get("id") == "chat-v2"
    ):
        import copy

        legacy = copy.deepcopy(profile)
        legacy.pop("taskProfile", None)
        if legacy.get("runtimePack", {}).get("taskContract") == "chat-v2":
            legacy["runtimePack"]["taskContract"] = "chat-v1"
        return profile_for_execution(legacy, config)
    if contract_id == "chat-v2":
        import copy

        from oryn_engine.chat_v2 import native_worker_supported, thinking_support

        source = copy.deepcopy(profile)
        existing = source.get("taskProfile")
        if existing and existing.get("contract", {}).get("id") == "chat-v2":
            return validate_profile(existing)
        # Only admitted chat-v1 execution is wrapped; unrelated native contracts
        # cannot be relabeled as chat by a caller's requested ID.
        base = profile_for_execution(source, config)
        if not base or base["contract"]["id"] != "chat-v1":
            return base
        runtime = source.get("runtimePack", {})
        pack = runtime.get("packId", "oryn.transformers-pytorch")
        standard = pack in {"oryn.llama-cpp", "oryn.mlx-lm", "oryn.transformers-pytorch"}
        maximum = 8192 if standard else 512
        legacy_worker = pack in {"oryn.llama-cpp", "oryn.mlx-lm"} and not native_worker_supported(
            pack
        )
        if legacy_worker:
            maximum = 512
        # Specialized reviewed adapters retain their smaller admitted ceilings.
        model_type = (config or {}).get("model_type")
        if pack == "oryn.mlx-lm":
            parameter = base["parameters"]["maxTokens"]["valueSchema"]
            caps = [
                part["maximum"] for part in parameter.get("allOf", [parameter]) if "maximum" in part
            ]
            if model_type in {
                "prism_hadamard_qwen35",
                "kolibri1",
                "gemma4",
                "gemma4_unified",
            } or any(cap < 512 for cap in caps):
                maximum = min([512, *caps])
        if "image" in base["inputModalities"]:
            maximum = min(maximum, 512)
        constraints = {}
        defaults = {}
        unsupported = []
        for name, parameter in base["parameters"].items():
            if name == "maxTokens":
                continue
            schema = parameter["valueSchema"]
            if "allOf" in schema:
                constraints[name] = schema["allOf"][-1]
            elif "const" in schema:
                constraints[name] = {"const": schema["const"]}
            if "default" in parameter:
                defaults[name] = parameter["default"]
            if parameter["handling"] == "unsupported":
                unsupported.append(name)
        constraints["maxTokens"] = {"minimum": 1, "maximum": maximum}
        defaults["maxTokens"] = min(4096, maximum)
        template = (config or {}).get("chat_template", "")
        support = thinking_support(model_type, template if isinstance(template, str) else "")
        if isinstance(source.get("_chatThinkingSupport"), dict):
            support = source["_chatThinkingSupport"]
        if pack == "oryn.mlx-lm" and model_type == "gpt_oss":
            support = thinking_support(fixed=True)
        if pack == "oryn.mlx-lm" and model_type in {"prism_hadamard_qwen35", "kolibri1"}:
            support = thinking_support(fixed=False)
        if legacy_worker:
            support = {"support": "unknown", "default": None}
        return make_profile(
            "chat-v2",
            features=base["features"],
            constraints=constraints,
            defaults=defaults,
            unsupported=unsupported,
            limits={
                **base["limits"],
                "contextTokens": min(8192, base["limits"].get("contextTokens", 8192)),
            },
            channels=["text", "reasoning"],
            input_modalities=base["inputModalities"],
            output_modalities=base["outputModalities"],
            thinking=support,
        )
    if profile.get("taskProfile"):
        return validate_profile(profile["taskProfile"])
    config = config or {}
    runtime = profile.get("runtimePack", {})
    technical = profile.get("technicalCapabilities", {})
    task = technical.get("task")
    contract_id = runtime.get("taskContract")
    if not contract_id:
        contract_id = next(
            (
                c["id"]
                for c in TASK_CONTRACTS.values()
                if task in [c["task"], *c.get("taskAliases", [])]
            ),
            None,
        )
    if contract_id not in TASK_CONTRACTS:
        return None
    c = TASK_CONTRACTS[contract_id]
    features = {
        k: ("supported" if v is True else "unsupported" if v is False else "unknown")
        for k, v in technical.get("capabilities", {}).items()
        if k in c["featureDefinitions"]
    }
    constraints, defaults, unsupported, channels, limits = {}, {}, [], [], {}
    pack = runtime.get("packId", "oryn.transformers-pytorch")
    if pack == "oryn.mlx-audio":
        from oryn_engine.mlx_audio_tasks import capability

        audio = capability(config, task)
        if audio is None:
            return None
        parameters = set(audio["parameters"])
        if contract_id == "speech-synthesis-v1":
            reference = "ref_audio" in parameters
            transcript = reference and "ref_text" in parameters
            native_voice = audio.get("defaults", {}).get("voice")
            voice = native_voice if isinstance(native_voice, str) and native_voice else "default"
            constraints = {
                "voice": {"enum": [voice, "reference"]} if reference else {"const": voice},
                # A shared API must not promise controls that a backend merely accepts
                # through **kwargs or ignores. Native defaults remain in charge.
                "language": {"const": "auto"},
                "speed": {"const": 1},
            }
            if reference:
                constraints["audioArtifactId"] = {"type": "string"}
            if transcript:
                constraints["referenceText"] = {"type": "string"}
            return make_profile(
                contract_id,
                constraints=constraints,
                defaults={"voice": voice, "language": "auto", "speed": 1},
                features={
                    "speechSynthesis": "supported",
                    "audioOutput": "supported",
                    "referenceAudio": "supported" if reference else "unsupported",
                },
                input_modalities=["text", "audio"] if reference else ["text"],
                output_modalities=["audio"],
                limits={
                    "textCharacters": 350,
                    "referenceAudioSeconds": 10,
                    "audioDurationSeconds": 30,
                },
            )
        if contract_id == "transcription-v1":
            return make_profile(
                contract_id,
                constraints={"language": {"const": "auto"}, "timestamps": {"const": False}},
                defaults={"language": "auto", "timestamps": False},
                features={"speechRecognition": "supported", "wordTimestamps": "unsupported"},
                input_modalities=["audio"],
                output_modalities=["text"],
                limits={"audioDurationSeconds": 30},
            )
        return None
    if contract_id == "text-classification-v1" and config.get("model_type") == "oryn_llama_turn":
        constraints["topK"] = {"minimum": 1, "maximum": 2}
        defaults["topK"] = 2
    from oryn_engine import mlx_bolding, mlx_kolibri, mlx_prism

    mlx_adapter = (
        mlx_kolibri
        if mlx_kolibri.supports(config)
        else mlx_bolding
        if mlx_bolding.supports(config)
        else mlx_prism
    )

    if contract_id == "chat-v1" and pack == "oryn.mlx-lm" and mlx_adapter.supports(config):
        mlx_adapter.validate_config(config)
        constraints["maxTokens"] = {"minimum": 1, "maximum": mlx_adapter.MAX_OUTPUT}
        defaults["maxTokens"] = 128
        limits["contextTokens"] = min(
            mlx_adapter.MAX_CONTEXT, config.get("text_config", config)["max_position_embeddings"]
        )
    if (
        contract_id == "chat-v1"
        and pack == "oryn.mlx-lm"
        and config.get("model_type") == "gemma4_unified"
    ):
        from oryn_engine.mlx_gemma import MAX_CONTEXT, MAX_OUTPUT, validate_config

        validate_config(config)
        constraints["maxTokens"] = {"minimum": 1, "maximum": MAX_OUTPUT}
        defaults["maxTokens"] = min(256, MAX_OUTPUT)
        limits["contextTokens"] = min(MAX_CONTEXT, config["text_config"]["max_position_embeddings"])
    if contract_id == "chat-v1" and pack == "oryn.mlx-lm" and config.get("model_type") == "gpt_oss":
        from oryn_engine.mlx_gpt_oss import MAX_CONTEXT, validate_config

        validate_config(config)
        limits["contextTokens"] = min(MAX_CONTEXT, config["max_position_embeddings"])
    if (
        pack == "oryn.mlx-lm"
        and config.get("model_type") == "gemma4"
        and task in {"image-text-to-text", "any-to-any"}
    ):
        from oryn_engine.mlx_gemma4 import MAX_CONTEXT, MAX_OUTPUT, validate_config

        validate_config(config)
        constraints["maxTokens"] = {
            "minimum": 1,
            "maximum": min(MAX_OUTPUT, 256)
            if contract_id == "multimodal-generation-v1"
            else MAX_OUTPUT,
        }
        defaults["maxTokens"] = 128
        if contract_id == "chat-v1":
            limits.update(contextTokens=MAX_CONTEXT, maxMessages=1)
    if contract_id == "zero-shot-classification-v1":
        from oryn_engine import decision_tasks

        if decision_tasks.supports(config, task):
            decision_tasks.validate_config(config)
            constraints["multiLabel"] = {"const": False}
            defaults["multiLabel"] = False
            limits["candidateLabels"] = decision_tasks.label_limit(config)
    if contract_id == "translation-v1":
        from oryn_engine import instruction_translation

        if pack == "oryn.llama-cpp":
            from oryn_engine import gguf_translation

            trusted = runtime.get("trustedMetadata", {})
            gguf_translation.validate_binding(
                trusted.get("adapterArchitecture"),
                trusted.get("chatTemplate"),
                trusted.get("projectorFile"),
            )
            constraints["targetLanguage"] = {"enum": sorted(gguf_translation.LANGUAGES)}
            defaults.update(targetLanguage="en", maxTokens=128)
            limits["contextTokens"] = min(
                gguf_translation.MAX_CONTEXT, trusted.get("contextLength", 4096)
            )
        elif instruction_translation.supports(config, task):
            translation = instruction_translation.adapter(config)
            translation.validate_config(config)
            constraints["targetLanguage"] = {"enum": sorted(translation.LANGUAGES)}
            defaults.update(targetLanguage="en", maxTokens=128)
            limits["contextTokens"] = min(
                translation.MAX_CONTEXT,
                translation.text_config(config)["max_position_embeddings"],
            )
        else:
            unsupported.append("targetLanguage")
    if (
        contract_id == "image-to-text-v1"
        and config.get("model_type") == "molparser_vision_encoder_decoder"
    ):
        from oryn_engine.molparser_tasks import token_limit, validate_config

        validate_config(config)
        maximum = token_limit(config)
        constraints["maxTokens"] = {"minimum": 1, "maximum": maximum}
        defaults["maxTokens"] = min(128, maximum)
    if contract_id == "summarization-v1" and runtime.get("runnerId") == "transformers.shared-tasks":
        defaults["maxTokens"] = 128
    if (
        contract_id == "image-to-text-v1"
        and config.get("model_type") == "vision-encoder-decoder"
        and isinstance(config.get("decoder"), dict)
        and config["decoder"].get("model_type") == "trocr"
    ):
        from oryn_engine.trocr_tasks import validate_config

        validate_config(config)
        maximum = min(512, config["decoder"]["max_position_embeddings"] - 1)
        constraints["maxTokens"] = {"minimum": 1, "maximum": maximum}
        defaults["maxTokens"] = min(128, maximum)
    if (
        contract_id in {"image-to-text-v1", "visual-question-answering-v1"}
        and config.get("model_type") == "blip"
    ):
        from oryn_engine.hdf5_blip import token_limit

        maximum = token_limit(config)
        constraints["maxTokens"] = {"minimum": 1, "maximum": maximum}
        defaults["maxTokens"] = min(128, maximum)
    if contract_id == "translation-v1" and config.get("model_type") == "marian":
        from oryn_engine.hdf5_marian import token_limit

        maximum = token_limit(config)
        constraints["maxTokens"] = {"minimum": 1, "maximum": maximum}
        defaults["maxTokens"] = min(128, maximum)
    if (
        contract_id in {"image-to-text-v1", "visual-question-answering-v1"}
        and config.get("model_type") == "llava_next"
    ):
        from oryn_engine.llava_next_tasks import MAX_OUTPUT, validate_config

        validate_config(config)
        constraints["maxTokens"] = {"minimum": 1, "maximum": MAX_OUTPUT}
        defaults["maxTokens"] = 128
    if contract_id == "multimodal-generation-v1":
        from oryn_engine.gemma4_tasks import input_modes, validate_config

        if runtime.get("runnerId") not in {"transformers.shared-tasks", "mlx.multimodal"}:
            return None
        validate_config(config)
        modes = input_modes(config)
        constraints.update(inputMode={"enum": modes}, outputMode={"const": "text"})
        defaults["outputMode"] = "text"
        limits["audioDurationSeconds"] = 8 if pack == "oryn.mlx-lm" else 30
        features.update(
            imageGrounding="supported" if "image" in modes else "unsupported",
            audioGrounding="supported" if "audio" in modes else "unsupported",
            audioOutput="unsupported",
        )
    if contract_id == "pose-estimation-v1":
        limits.update(poses=8, joints=len(config.get("id2label", {})))
    if contract_id == "table-question-answering-v1" and config.get("model_type") == "tapas":
        from oryn_engine.coverage_tasks import table_shape_limits

        rows, columns = table_shape_limits(config)
        limits.update(tableRows=rows, tableColumns=columns)
    if contract_id in {"tabular-classification-v1", "tabular-regression-v1"}:
        if runtime.get("runnerId") != "transformers.shared-tasks":
            return None
        limits.update(trainingRows=256, queryRows=64, featureColumns=32)
        if config.get("model_type") == "tabpfn_mix":
            limits["featureColumns"] = min(32, config["n_features"])
            if contract_id == "tabular-classification-v1":
                limits["classes"] = config["n_classes"]
        if config.get("model_type") == "mitra_tab2d" and contract_id == "tabular-classification-v1":
            limits["classes"] = config["dim_output"]
    if contract_id == "audio-question-answering-v1":
        from oryn_engine.audio_understanding import TASK_FAMILIES, validate_config

        validate_config(config)
        features["audioGrounding"] = (
            "supported"
            if runtime.get("runnerId") == "transformers.shared-tasks"
            and config.get("model_type") in TASK_FAMILIES["audio-text-to-text"]
            else "unknown"
        )
        limits["audioDurationSeconds"] = 30
        if config.get("model_type") == "voxtral":
            constraints["maxTokens"] = {"minimum": 1, "maximum": 128}
    if contract_id == "visual-document-retrieval-v1" and config.get("model_type") == "colqwen3":
        from oryn_engine.colqwen3_tasks import DIMENSIONS, PAGE_TOKENS, QUERY_TOKENS

        limits.update(
            embeddingDimensions=DIMENSIONS,
            documentTokens=PAGE_TOKENS,
            queryTokens=QUERY_TOKENS,
            batchItems=2,
        )
    if (
        contract_id == "visual-document-retrieval-v1"
        and config.get("model_type") == "jina_embeddings_v4"
    ):
        from oryn_engine.jina4_tasks import DIMENSIONS, PAGE_TOKENS, QUERY_TOKENS

        limits.update(
            embeddingDimensions=DIMENSIONS,
            documentTokens=PAGE_TOKENS,
            queryTokens=QUERY_TOKENS,
            batchItems=2,
        )
    if contract_id == "visual-document-retrieval-v1" and config.get("model_type") == "qwen3_5":
        from oryn_engine.colqwen35_tasks import (
            PAGE_TOKENS,
            QUERY_TOKENS,
            dimensions,
            validate_config,
        )

        validate_config(config)
        limits.update(
            embeddingDimensions=dimensions(config),
            documentTokens=PAGE_TOKENS,
            queryTokens=QUERY_TOKENS,
            batchItems=2,
        )
    if contract_id == "time-series-forecasting-v1":
        from oryn_engine.forecast_tasks import settings, validate_config

        if runtime.get("runnerId") != "transformers.shared-tasks":
            return None
        validate_config(config)
        config = settings(config)
        constraints.update(
            horizon={"maximum": min(config["horizon_length"], 512)},
            quantiles={"items": {"enum": config["quantiles"]}},
        )
        defaults["horizon"] = min(config["horizon_length"], 16)
        features.update(quantiles="supported", covariates="unsupported")
        unsupported.extend(["pastCovariates", "futureCovariates"])
        limits.update(
            contextLength=min(config["context_length"], 2048),
            forecastHorizon=min(config["horizon_length"], 512),
            covariateColumns=0,
        )
    if contract_id in {"video-classification-v1", "video-question-answering-v1"}:
        limits.update(videoDurationSeconds=30, videoFrames=32, videoPixels=1920 * 1080)
        if config.get("model_type") in {"videollama3_qwen2", "oryn_llava_video_qwen2"}:
            constraints["maxTokens"] = {"minimum": 1, "maximum": 256}
            defaults["maxTokens"] = 128
            limits.update(videoFrames=8)
    if contract_id == "chat-v1":
        features.update(
            chat="supported",
            streaming="supported",
            toolCalling="unsupported",
            structuredOutput="unsupported",
        )
        features["multimodalConversation"] = (
            "supported" if "image" in technical.get("inputModalities", []) else "unsupported"
        )
        channels = ["text"]
        if technical.get("contextLength"):
            limits["contextTokens"] = min(
                limits.get("contextTokens", technical["contextLength"]), technical["contextLength"]
            )
        if pack == "oryn.llama-cpp" and runtime.get("trustedMetadata", {}).get("projectorFile"):
            limits["maxMessages"] = 1
    if contract_id == "transcription-v1":
        architectures = technical.get("architectures", [])
        whisper = "WhisperForConditionalGeneration" in architectures
        qwen = any("Qwen3ASR" in a for a in architectures)
        heads = config.get("_orynGenerationConfig", {}).get("alignment_heads")
        features.update(
            speechRecognition="supported",
            languageDetection="supported",
            wordTimestamps="supported"
            if whisper and isinstance(heads, list) and bool(heads)
            else "unsupported"
            if qwen
            else "unknown",
        )
        if qwen:
            constraints["timestamps"] = {"const": False}
        limits["audioDurationSeconds"] = 30
        if (
            pack == "oryn.onnx-runtime"
            or "Wav2Vec2ForCTC" in architectures
            or config.get("model_type") in {"nemotron3_5_asr", "parakeet_tdt"}
        ):
            features.update(
                speechRecognition="supported",
                languageDetection="unsupported",
                wordTimestamps="unsupported",
            )
            constraints.update(language={"const": "auto"}, timestamps={"const": False})
            if pack == "oryn.onnx-runtime" and config.get("model_type") == "whisper":
                features["languageDetection"] = (
                    "supported" if config.get("vocab_size", 0) >= 51865 else "unsupported"
                )
        if config.get("model_type") == "audio8_asr_infinite":
            features.update(languageDetection="unsupported", wordTimestamps="unsupported")
            constraints.update(language={"enum": ["en", "zh"]}, timestamps={"const": False})
            defaults["language"] = "en"
        if config.get("model_type") == "oryn_ct2_whisper":
            features.update(languageDetection="supported", wordTimestamps="unsupported")
            constraints.update(language={"const": "auto"}, timestamps={"const": False})
    if contract_id == "forced-alignment-v1":
        features.update(forcedAlignment="supported", wordTimestamps="supported")
        languages = runtime.get("trustedMetadata", {}).get("languages")
        if languages:
            constraints["language"] = {"enum": languages}
    if contract_id == "text-to-music-v1":
        ace = pack == "oryn.ace-step-mlx"
        music3 = (
            pack == "oryn.diffusers-pytorch"
            and config.get("_class_name") == "MiniMaxMusic3ModularPipeline"
        )
        constraints["durationSeconds"] = {
            "minimum": 10 if ace else 1,
            "maximum": 180 if ace else 30,
        }
        features.update(
            audioOutput="supported",
            lyrics="supported" if ace or music3 else "unsupported",
            deterministicSeed="supported" if ace or music3 else "unsupported",
        )
        if not ace and not music3:
            unsupported.extend(["lyrics", "seed"])
        if music3:
            constraints["seed"] = {"minimum": 0, "maximum": 2**32 - 1}
            defaults.update(seed=0, lyrics="[Instrumental]")
    if contract_id == "speech-synthesis-v1" and config.get("model_type") == "breeze":
        from oryn_engine.breeze_tasks import validate_config

        validate_config(config)
        return make_profile(
            contract_id,
            constraints={
                "voice": {"const": "auto"},
                "language": {"const": "auto"},
                "speed": {"const": 1},
            },
            features={
                "speechSynthesis": "supported",
                "audioOutput": "supported",
                "referenceAudio": "unsupported",
            },
            input_modalities=["text"],
            output_modalities=["audio"],
            limits={"textCharacters": 300, "audioDurationSeconds": 30},
        )
    if contract_id == "speech-synthesis-v1" and config.get("model_type") == "omnivoice":
        from oryn_engine.omnivoice_tasks import LANGUAGES, validate_config

        validate_config(config)
        return make_profile(
            contract_id,
            constraints={
                "voice": {"const": "auto"},
                "language": {"enum": list(LANGUAGES)},
                "speed": {"const": 1},
            },
            defaults={"language": "en"},
            features={
                "speechSynthesis": "supported",
                "audioOutput": "supported",
                "referenceAudio": "unsupported",
            },
            input_modalities=["text"],
            output_modalities=["audio"],
            limits={"textCharacters": 300, "audioDurationSeconds": 30},
        )
    if contract_id == "speech-synthesis-v1" and config.get("model_type") == "oryn_sopro":
        from oryn_engine.sopro_tasks import LANGUAGES, validate_config

        validate_config(config)
        return make_profile(
            contract_id,
            constraints={
                "voice": {"const": "reference"},
                "language": {"enum": list(LANGUAGES)},
                "speed": {"const": 1},
                "audioArtifactId": {"type": "string"},
            },
            defaults={"language": "en"},
            features={
                "speechSynthesis": "supported",
                "audioOutput": "supported",
                "referenceAudio": "supported",
            },
            input_modalities=["text", "audio"],
            output_modalities=["audio"],
            limits={"textCharacters": 300, "referenceAudioSeconds": 10, "audioDurationSeconds": 30},
        )
    if contract_id == "speech-synthesis-v1":
        features.update(speechSynthesis="supported", audioOutput="supported")
        voices = runtime.get("trustedMetadata", {}).get("voices")
        if voices:
            constraints["voice"] = {"enum": [v["id"] for v in voices]}
        languages = runtime.get("trustedMetadata", {}).get("languages")
        if languages:
            constraints["language"] = {"enum": languages}
        if pack == "oryn.qwen-tts-pytorch":
            from oryn_engine.qwen_tts import controls

            speech = controls(config)
            constraints.update(
                voice={"enum": [v["id"] for v in speech["voices"]]},
                language={"enum": speech["languages"]},
                speed={"const": 1.0},
            )
            defaults.update(voice=speech["defaultVoice"], language="en-us", speed=1.0)
    if contract_id == "embeddings-v1":
        features["embeddings"] = "supported"
        if config.get("model_type") == "pplx_contextual_qwen3_5":
            from oryn_engine.contextual_embeddings import MAX_BATCH, validate_config

            validate_config(config)
            constraints.update(
                inputType={"enum": ["query", "document"]}, contextual={"type": "boolean"}
            )
            defaults.update(inputType="document", contextual=False)
            limits.update(
                batchItems=MAX_BATCH,
                textCharacters=8192,
                embeddingDimensions=config["embedding_dim"],
            )
        if config.get("model_type") in {"ministral3", "qwen3_bidirec"}:
            from oryn_engine.bidirectional_embeddings import MAX_BATCH, validate_config

            validate_config(config)
            limits.update(
                batchItems=MAX_BATCH, textCharacters=8192, embeddingDimensions=config["hidden_size"]
            )
        if config.get("model_type") == "qwen3_vl":
            from oryn_engine.qwen_retrieval import MAX_BATCH, validate_config

            validate_config(config)
            limits.update(
                batchItems=MAX_BATCH,
                textCharacters=8192,
                embeddingDimensions=config["text_config"]["hidden_size"],
            )
    if contract_id == "voice-activity-detection-v1":
        limits["audioDurationSeconds"] = 30
    if contract_id == "audio-classification-v1":
        from oryn_engine import speechx_tasks, wav2vec_heads

        if speechx_tasks.supports(config, task):
            speechx_tasks.validate_config(config)
            limits["audioDurationSeconds"] = speechx_tasks.duration_limit(config)
            constraints["classificationHead"] = {"enum": ["language", "gender"]}
            defaults["classificationHead"] = "language"

        if wav2vec_heads.supports(config, task):
            wav2vec_heads.validate_config(config)
            limits["audioDurationSeconds"] = wav2vec_heads.duration_limit(config)
    if contract_id == "audio-transformation-v1" and config.get("model_type") == "oryn_tiger_dnr":
        from oryn_engine.tiger_tasks import validate_config

        validate_config(config)
        return make_profile(
            contract_id,
            constraints={"operation": {"const": "separate"}, "sampleRate": {"const": 44100}},
            defaults={"operation": "separate", "sampleRate": 44100},
            limits={"audioDurationSeconds": 12},
        )
    if contract_id == "audio-transformation-v1" and config.get("model_type") == "oryn_reuse":
        from oryn_engine.reuse_tasks import validate_config

        validate_config(config)
        return make_profile(
            contract_id,
            constraints={"operation": {"const": "enhance"}, "sampleRate": {"const": 8000}},
            defaults={"operation": "enhance", "sampleRate": 8000},
            limits={"audioDurationSeconds": 2},
        )
    if contract_id == "audio-transformation-v1" and config.get("model_type") == "oryn_mpsenet":
        from oryn_engine.mpsenet_tasks import validate_config

        validate_config(config)
        return make_profile(
            contract_id,
            constraints={"operation": {"const": "enhance"}, "sampleRate": {"const": 16000}},
            defaults={"operation": "enhance", "sampleRate": 16000},
            limits={"audioDurationSeconds": 30},
        )
    if (
        contract_id == "audio-transformation-v1"
        and config.get("model_type") == "qwen3_tts_tokenizer_12hz"
    ):
        from oryn_engine.qwen_codec_tasks import validate_config

        validate_config(config)
        return make_profile(
            contract_id,
            constraints={
                "operation": {"enum": ["encode", "decode"]},
                "sampleRate": {"const": 24000},
            },
            defaults={"operation": "encode", "sampleRate": 24000},
            limits={"audioDurationSeconds": 30},
        )
    if contract_id == "audio-transformation-v1" and pack == "oryn.onnx-runtime":
        limits["audioDurationSeconds"] = 30
        constraints.update(operation={"const": "enhance"}, sampleRate={"const": 48000})
        defaults.update(operation="enhance", sampleRate=48000)
    if contract_id == "reranking-v1":
        features["reranking"] = "supported"
        if pack == "oryn.onnx-runtime":
            unsupported.append("instruction")
    generation = profile.get("generationProfile", {})
    if contract_id == "image-editing-v1":
        from oryn_engine.diffusion_tasks import reference_image_limits

        minimum, maximum = (
            reference_image_limits(
                (technical.get("architectures") or [""])[0],
                runtime.get("trustedMetadata", {}).get("composition"),
            )
            if pack == "oryn.diffusers-pytorch"
            else (1, 1)
        )
        limits.update(minReferenceImages=minimum, maxReferenceImages=maximum)
        features["multipleReferenceImages"] = "supported" if maximum > 1 else "unsupported"
        if maximum > 1:
            constraints["referenceImageArtifactIds"] = {
                "minItems": minimum - 1,
                "maxItems": maximum - 1,
            }
    for k, v in generation.get("parameters", {}).items():
        if k not in c["parameterNames"]:
            continue
        if v.get("supported", True) and v.get("editable", True):
            constraints[k] = {"minimum": v["minimum"], "maximum": v["maximum"]}
            if k in {"width", "height", "steps"}:
                constraints[k]["multipleOf"] = v["step"]
        else:
            constraints[k] = {"const": v["default"]}
        defaults[k] = v["default"]
    if contract_id in {
        "text-to-image-v1",
        "unconditional-image-generation-v1",
        "image-to-image-v1",
        "image-editing-v1",
        "image-layers-v1",
        "text-to-video-v1",
        "image-to-video-v1",
        "image-text-to-video-v1",
    }:
        features["deterministicSeed"] = "supported"
        if "negativePrompt" in features or "negativePrompt" in c["featureDefinitions"]:
            features["negativePrompt"] = "supported"
        if pack == "oryn.mflux-mlx" or (
            pack == "oryn.diffusers-pytorch"
            and (
                "Flux2KleinPipeline" in technical.get("architectures", [])
                or runtime.get("trustedMetadata", {})
                .get("composition", {})
                .get("sampling", {})
                .get("negativePrompt")
                == ""
            )
        ):
            # These sealed bindings accept only an empty negative prompt.
            constraints["negativePrompt"] = {"const": ""}
            features["negativePrompt"] = "unsupported"
    if (
        contract_id in {"text-to-video-v1", "image-to-video-v1", "image-text-to-video-v1"}
        and pack == "oryn.diffusers-pytorch"
    ):
        from oryn_engine.diffusion_tasks import VIDEO_PIPELINES

        if any(family in VIDEO_PIPELINES for family in technical.get("architectures", [])):
            ltx = "LTXPipeline" in technical.get("architectures", [])
            from oryn_engine.wan_video import expanded

            multiple = 32 if ltx or expanded(config) else 16
            constraints.update(
                width={"maximum": 512, "multipleOf": multiple},
                height={"maximum": 512, "multipleOf": multiple},
                frames={"enum": [1, 9, 17] if ltx else [1, 5, 9, 13, 17]},
                steps={"maximum": 30},
            )
            defaults.update(width=256, height=256, frames=17, steps=20, guidanceScale=6.0)
    if (
        contract_id in {"text-to-video-v1", "image-text-to-video-v1"}
        and pack == "oryn.diffusers-pytorch"
        and "MiniMaxH3ModularPipeline" in technical.get("architectures", [])
    ):
        from oryn_engine.minimax_h3 import task_profile

        return task_profile(contract_id)
    result = make_profile(
        contract_id,
        task=task if task in [c["task"], *c.get("taskAliases", [])] else c["task"],
        features=features,
        constraints=constraints,
        defaults=defaults,
        unsupported=unsupported,
        limits=limits,
        channels=channels,
        input_modalities=["text"]
        if contract_id == "speech-synthesis-v1"
        else [m for m in technical.get("inputModalities", []) if m in c["inputModalities"]] or None,
        output_modalities=[
            m for m in technical.get("outputModalities", []) if m in c["outputModalities"]
        ]
        or None,
    )
    return result

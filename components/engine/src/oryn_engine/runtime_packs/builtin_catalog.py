from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oryn_engine.generated_trust import DEVELOPMENT_KEY_ID, DEVELOPMENT_MODULUS
from oryn_engine.runtime_packs.manifests import SignedManifestVerifier, TrustedPublisherKey
from oryn_engine.runtime_packs.registry import RuntimePackRegistry

# Development-alpha publisher key. Release catalogs must use the separately protected release
# hierarchy; no private key is present in the application or repository.
DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-development-2026",
    22475301446459617703557098401057632606300504984885845974634855320857122310902282482269030143338376544237834461862932670246236180942010253410125000775012520433183612913337573616308826638326106133658304104399259415882507904548608142025614572650142172510944170827506452804906146709541172791915168812335139898962160495983845132924588050614980592134399357365644083997881959586178972696039878684512019289033192694693016225621466804935668072245829758120719038252177954866987799580691098147273504215075098114780170954704415161929458095617839234191672649470795006933775240412017113436512303188167744368709148567687113574519883,
)

ACE_STEP_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-ace-step-development-2026",
    25499628309423268490726672120171014620363832114365685376878016824814783331553084022535325008496420846347426221185198239102681705544236694508453565346227866976985595015384362016367133867194152410914444690338159846704327711432053789275141850342363007993912714297376034872437807485364008557179673936936531525234496456928662359417648134167878510877389514863582707037462567845919580529181932876617616298533887148730037006880823999650511559801822847155177848121640491766998002521534645590831938949340854650626103442700333088567504091636031631723194471196292558385971198112972178092128452570577483835918401130586615721159463,
)

SPRINT_30_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-sprint-30-development-2026",
    27933886347384254277261764733738308809212946210556704638427257577336800985524347344536960835300245375011728858222949855392966530022171971926493189819062014450537696752036316017804739907163062650814469560048818953816066967181454632077915887962770095772514560179788640017707860066000184092151290900601712450726336665529186068046973499417006466929658728722573431414704745916856459132735453045280879360520102695859635909748889469515833984012855106459256397898101647553095004229991459146255572315697947093134120070503965928650030678081245640859168644571786550621322375293306375333824147828055752167171279204945713914252581,
)

SPRINT_31_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-sprint-31-development-2026",
    26689625404289651693040273691708211049831478091699678831104149791587143023566764503817593873436121155834919774088525422468064828389617097818771670841968806552001694088280356976674985780620466825558804409948985290997399909674075754429471882318286308970528367634773662847734431824321576441309442495732861946241383109366925471929581136584134162593636446006034417648825144387280397905299033066206934861947605586532360143322372225061115591871954571766067442538866000600471134254549686733918194177184424811211777651000252397675164173512361070284408932005032675475957029563246935086862406754539297829510560407956835300536317,
)

SPRINT_32_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-sprint-32-development-2026",
    25378447214867415227951470733979850451969707906619425230412462235275728063343624110854504503954655781362472455167858344126241369601379842327280036067769120565624221912432358234496943779983738134463160311596799895804024035215397360447015384403065522551533563054338988937096208731808376003966048667555555847501332682550321288375384329646915895778968277322177741732146867598275282103231792697560192413113700997962543588933646841708590572398695803631579481006816974078840477629797075105087846408970344404155451615198269896141392430746229185041461462555189579004211675393285359883236648385501031274939268058683684559001207,
)

GENERIC_DIFFUSERS_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-generic-diffusers-development-2026",
    16974393205291890715488982953597313626670805155688991691697136015751867652186398596535522588953349641374111198663389947104610632816623917791493720081728985166494738566175418272685088451982299701306937948170363311954909564185479113043204821063872370062606204727504355539773275160557034577328882638861118783278313257008248599865876534143181960914144267240618004968348783226364187857552246222722777931471875098773560808454233170138280425922848645946324599220759043368903173829687339085011359956230015976138187637775571724971092395034998234804948597118745767015057634293572136035989995771092718466160734203560797672480773,
)

SPRINT_36_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-sprint-36-development-2026",
    18587801353945734821265608448512744077395039273198969368729686919542914322261710954545093832414035179301328617088288117257993090129426721410413534952483451360313582992651358961170533296049325197601568088979075388123684551602203702722981505438353026895790155115628476517197385160077899287581333508097362041226928406364573947064016196288140908556523669024627464295574180643111799243006847471556270632554073717789037408654464214507839836126209237754446540737326389705929423526898491951985325928251611394941895607945257590458928084173502454589255493479869217868864608177537201950408616691517403714970300772210107819069467,
)

BITSANDBYTES_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-bitsandbytes-development-2026",
    20495391919518967233188423799300250913516682572102753863441375270843792231578332098908684138516252477895072816359067119243433951185308694549179908354769971623703991837264944089240173606988108797729642245352593938079265307481649812722944740298890840022910903527193898522677057493365708674047872470555312850716577699517874747542328543006609002881820902674468142059120079035088810557243325144111433703840286755784414998505565323719822691988016969996186265160757149016471954267973177141125369229449462597561436045554622509728977876237840937715348458066080045089225299515286783110565877335091956654776939190112774808611459,
)

GENERATION_PROFILES_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-generation-profiles-development-2026",
    31868436996247802248358594875049572457203538230754917287147674726138186853078250025993389541197082072630160055745413520451052666235586411665521643681126900891380912882637813177532811457127278436570853491767221213327064524432967055742328696616677700866852656854273733336319443158602188456851471513393189734399505239401837579516175594863453390916501488112335017920741428393996925885122991736534127520539585415141081275311962002756017927424432876952311863342178868097186172065058434550038744585323477502279742335607022146681152708341339125566587829264003921182149171405078077971008667941783369092620613345617869259628347,
)

PACK: dict[str, Any] = {
    "payload": {
        "kind": "runtime-pack",
        "schemaVersion": 1,
        "packId": "oryn.transformers-pytorch",
        "version": "0.1.0",
        "publisher": "oryn.first-party",
        "abi": "1",
        "runtime": {"kind": "python", "entrypoint": "oryn_engine.runtime_packs.fixture_worker"},
        "dependencyClosure": "oryn-engine-py312-transformers-5",
        "runners": [
            {
                "runnerId": "transformers.chat",
                "taskContracts": ["chat-v1"],
                "backends": ["mps", "cuda", "cpu"],
            },
            {
                "runnerId": "transformers.transcription",
                "taskContracts": ["transcription-v1"],
                "backends": ["mps", "cuda", "cpu"],
            },
            {
                "runnerId": "transformers.musicgen",
                "taskContracts": ["text-to-music-v1"],
                "backends": ["mps", "cuda", "cpu"],
            },
            {
                "runnerId": "transformers.embeddings",
                "taskContracts": ["embeddings-v1"],
                "backends": ["mps", "cuda", "cpu"],
            },
            {
                "runnerId": "transformers.reranking",
                "taskContracts": ["reranking-v1"],
                "backends": ["mps", "cuda", "cpu"],
            },
        ],
        "policy": {
            "network": "deny",
            "filesystemRoots": ["model", "artifact-input", "artifact-output", "scratch"],
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-sprint-30-development-2026",
        "digest": "3a1dd26b3441ec34e081f059a5c61bfe31ae644fddd24c28fddb1bbd95759d39",
        "signature": "O4MEwm1XxiYJqb81vDTY9gLW1rPWTxE6yB4Cs6eWzxfWxFdnFWVpvwOVc/2h6AslgjmBdcVa/AIT+1evC8ArUwCa9e5ItW0zLgP518vM1albmo1lDBEnurSVbyg8QyLFNq4F15Wvt9LbCDIXo3br1JBEmt7aT2IovQgIXv/5acpfkz9r2WDXdvBSrl8SZAtfHFyVpcyjwdryiicmXQxLKkJ1B6FdJ0FDSPX4sGT6kf9C3qGD1Tb5DgqIp8bogH8g7qwzoozatcx/XV5F4HxZFr/xDZCRBh3hb/7G2KsoKVCrdhA7Ol9mrBwH4h6h4CRZ/G7txJK3BdL+JG++wXNgkA==",  # noqa: E501
    },
}

COMPATIBILITY: tuple[dict[str, Any], ...] = (
    {
        "payload": {
            "kind": "model-compatibility",
            "schemaVersion": 1,
            "manifestId": "oryn.compat.tiny-qwen2",
            "runtime": {
                "packId": "oryn.transformers-pytorch",
                "packVersion": "0.1.0",
                "runnerId": "transformers.chat",
            },
            "model": {
                "provider": "huggingface",
                "repository": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "revision": "f56c2fbf7fabc03ffff3494d3c2f8d06c774493d",
            },
            "match": {"architectures": ["Qwen2ForCausalLM"], "tasks": ["text-generation"]},
            "taskContract": "chat-v1",
            "level": "native-declarative",
            "requiredFiles": ["config.json", "*.safetensors", "tokenizer*"],
            "capabilities": {
                "task": "text-generation",
                "architectures": ["Qwen2ForCausalLM"],
                "inputModalities": ["text"],
                "outputModalities": ["text"],
                "capabilities": {"chat": True, "streaming": True},
            },
            "resources": {"estimatedResidentBytes": 32000000, "workingMemoryBytes": 8000000},
        },
        "integrity": {
            "algorithm": "rsa-sha256",
            "keyId": "oryn-development-2026",
            "digest": "f019623e3c5161e417e886efbd37de906eb6cd337a626067103684d126ef9605",
            "signature": "q9KT4FwROeTvTt6/EXlhGKBSLb8TDsWt2aseKOFj+OsiiHBQC+1mdkKC7T+B+d8pJuaz+IEEgh2THamHJFhgOfswjQymqAlBcTp989CSRY6IuQIC78XwFfTVpjd8HLmNmNv1Y4Pt4M04NKRfk5JI6noc1eZa+C+5Hr4IXupjYiGXMn78chBULU+JdnEcEsOXB/JVuKfWXIRKP+M8CBy5hYYw9FTmu60Frc7b78bUEk+6UbP7QmLzlPiKqeUp0OHO1kDlK0pQiDMoBxydZbOJCjvHa2qjuHSnUlEznfv6G6wNEW51xV7tWk2FN/Q4Y7PHo90n9SJDPoU5jtOeQZafIA==",  # noqa: E501
        },
    },
    {
        "payload": {
            "kind": "model-compatibility",
            "schemaVersion": 1,
            "manifestId": "oryn.compat.tiny-whisper",
            "runtime": {
                "packId": "oryn.transformers-pytorch",
                "packVersion": "0.1.0",
                "runnerId": "transformers.transcription",
            },
            "model": {
                "provider": "huggingface",
                "repository": "Xenova/tiny-random-WhisperForConditionalGeneration",
                "revision": "e1a9e01a339a6c8ba2aadb92a271624e46f86d05",
            },
            "match": {
                "architectures": ["WhisperForConditionalGeneration"],
                "tasks": ["automatic-speech-recognition"],
            },
            "taskContract": "transcription-v1",
            "level": "native-declarative",
            "requiredFiles": [
                "config.json",
                "*.safetensors",
                "tokenizer*",
                "preprocessor_config.json",
            ],
            "capabilities": {
                "task": "automatic-speech-recognition",
                "architectures": ["WhisperForConditionalGeneration"],
                "inputModalities": ["audio"],
                "outputModalities": ["text"],
                "capabilities": {"speechRecognition": True},
            },
            "resources": {"estimatedResidentBytes": 32000000, "workingMemoryBytes": 8000000},
        },
        "integrity": {
            "algorithm": "rsa-sha256",
            "keyId": "oryn-development-2026",
            "digest": "1826b48111be7b901f28e9581137c88071c7101551ab9e48853d95c9ca5d6b3d",
            "signature": "FtcF+HISNrlo0Y0cdJfsuyslXXBs6ITjvPC3PbNuqqdcsi3is1phu9oldYKxhm0lhf79eueryRa6Ix5IqAR/S+L0mj2PdsuT8YEcmAcRIXTbD5zv/yTnU2fgvc5JqPdlvXhoU7DvoC92nJwF+we5MkNdTtAeWrvpAyLBpCVoYZQEGVxnavddEvmMaS5pNT4/3/qOwpSkrFul6N5s9n+jPBuCfNYJbNZ1dSggiu/InrcGWXuUDjktHmE066Q42/STEebOCYhd33qlRkk+fr+/07LgTUPOc30bX1M5E5fkAsz+3Jb2hNuLcLb2TKnYPqGbw2uZUFm5+QynTVshIOJdMw==",  # noqa: E501
        },
    },
    {
        "payload": {
            "kind": "model-compatibility",
            "schemaVersion": 1,
            "manifestId": "oryn.compat.tiny-musicgen",
            "runtime": {
                "packId": "oryn.transformers-pytorch",
                "packVersion": "0.1.0",
                "runnerId": "transformers.musicgen",
            },
            "model": {
                "provider": "huggingface",
                "repository": "hf-tiny-v2/tiny-random-MusicgenForConditionalGeneration",
                "revision": "4606e560821ed4c10d76d6a6513c893ec3cf2ae0",
            },
            "match": {
                "architectures": ["MusicgenForConditionalGeneration"],
                "tasks": ["text-to-audio"],
            },
            "taskContract": "text-to-music-v1",
            "level": "native-declarative",
            "requiredFiles": [
                "config.json",
                "*.safetensors",
                "tokenizer*",
                "preprocessor_config.json",
            ],
            "capabilities": {
                "task": "text-to-audio",
                "architectures": ["MusicgenForConditionalGeneration"],
                "inputModalities": ["text"],
                "outputModalities": ["audio"],
                "capabilities": {"audioOutput": True},
            },
            "resources": {"estimatedResidentBytes": 64000000, "workingMemoryBytes": 16000000},
        },
        "integrity": {
            "algorithm": "rsa-sha256",
            "keyId": "oryn-development-2026",
            "digest": "473c0963085669faaac102fb15c6640790fa919c7b15090ae3f54348e1d5378d",
            "signature": "kEgVjpLcmLTIlvh7lWrX0BAPoyjOMQJcuTvYcEWyiIxdY4VBAdx4YbvVN9DXblWJB9w5i74l1Jt/ilPLVJmnLhlYYfdb4XSnLK97CT0jJck9MrhbAlBlgIthDgVbv1jHLCRVhhQl5B8rylZZH1Fd6U0foiNnXKDJsbE+M7yHn7PKc3LIsHGemQLUOAeJXWmC7HJ9U61d5GZM045V5NuzHWHmFQIl/+ujqL6/wSPtb2DnpRWKoo0f3F0NLUpyWLkAztwH3GqSsry1V2Zm75Cm6ZCQbSOQDA3tYat8JU9gzBofLzCKeyMRgY5KNo1hc4iWdSgUkLOKsE+UF9+ZjtdY8w==",  # noqa: E501
        },
    },
)

ACE_STEP_PACK: dict[str, Any] = {
    "payload": {
        "kind": "runtime-pack",
        "schemaVersion": 1,
        "packId": "oryn.ace-step-mlx",
        "version": "0.1.0",
        "publisher": "oryn.first-party",
        "abi": "1",
        "runtime": {
            "kind": "python",
            "entrypoint": "oryn_engine.runtime_packs.ace_step_worker",
        },
        "dependencyClosure": "ace-step-v1.5.0-dce621408bee-py312",
        "runners": [
            {
                "runnerId": "ace-step.music",
                "taskContracts": ["text-to-music-v1"],
                "backends": ["mlx"],
            }
        ],
        "policy": {
            "network": "deny",
            "filesystemRoots": ["model", "artifact-output", "scratch"],
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-ace-step-development-2026",
        "digest": "6d360179ac5c9f04b65613ab06c65ca6f6c334cb92dedb64c6a7570c3907f7fd",
        "signature": "jZzid3qOJp6PN8X0gPNiJk+N1ikYj5m5JsvhDCWBqK9iMHPZXt8P2XBk9TMrx8GLmrGcFCfTi2SxYlVxGTfriB60KDymZZdMG2iJmwexiOF4cv50XCQWtO4hlIzGwnrjui5Ser/7li34p6UjGNXXUxEemx5ecA87kIHdJlPcibIgUBl0A0pwPVfAJZhOhKGMc3T+1wb/FQRMtOZhKmHtb1tHedUdaLU8ZA9Pkm3wVqO5qaU369r2GyzZaanZu3Zxuxq2X3RuPXoVEAys4/UT14pJojYSjZbwJqW1ol2Kjcg6bsfXubjy0Yopbcy7MRgFPjMunC9s6beGju4DmnW7lw==",  # noqa: E501
    },
}

ACE_STEP_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.ace-step-1.5",
        "runtime": {
            "packId": "oryn.ace-step-mlx",
            "packVersion": "0.1.0",
            "runnerId": "ace-step.music",
        },
        "model": {
            "provider": "huggingface",
            "repository": "ACE-Step/Ace-Step1.5",
            "revision": "19671f406d603126926c1b7e2adc169acbcade22",
        },
        "match": {
            "architectures": ["AceStepConditionGenerationModel"],
            "tasks": ["text-to-audio"],
        },
        "taskContract": "text-to-music-v1",
        "level": "audited-adapter",
        "requiredFiles": [
            "config.json",
            "acestep-v15-turbo/config.json",
            "acestep-v15-turbo/model.safetensors",
            "acestep-v15-turbo/silence_latent.pt",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "Qwen3-Embedding-0.6B/*.json",
            "Qwen3-Embedding-0.6B/*.jinja",
            "Qwen3-Embedding-0.6B/*.txt",
            "Qwen3-Embedding-0.6B/*.safetensors",
        ],
        "capabilities": {
            "task": "text-to-audio",
            "architectures": ["AceStepConditionGenerationModel"],
            "inputModalities": ["text"],
            "outputModalities": ["audio"],
            "capabilities": {
                "audioOutput": True,
                "lyrics": True,
                "deterministicSeed": True,
                "streaming": False,
            },
        },
        "resources": {
            "estimatedResidentBytes": 12_000_000_000,
            "workingMemoryBytes": 8_000_000_000,
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-ace-step-development-2026",
        "digest": "45ca4a05e65782a379e29d50c7a80d0891c9db610e83649d7694d00b7a36c77f",
        "signature": "MHrCPlBwJwyaywtvnged5HUe4nfFdH5c42d1qaeVuzmXERnJIWvAsiNjhTx0EpgxPBGyAYZfhPI9wY8Y237IKstqfvFNrJ7sjFAS1Ss8B6zwVXDjCLT53ZjI1vWjBIS3QXAGX7U7RzdN2V/JOP1HILGWyI9d5uCmQROlgKctR2tswaFDA7+K4EihJWMbPQMt9V4X6YgQaeDq+oQqRxAMmV8X4ZvQq+dLzei8gj1UgW6gYkz9dZ05fI0estp0WUuNI/z2b4pEor0k9gpSocYbjGPqWSY5K8CYqtdhLIgcwfCtMh2SphSQm/G1Fwzug9eJXrqcRAtsEpg5ZlmhiLEDtQ==",  # noqa: E501
    },
}

MFLUX_PACK: dict[str, Any] = {
    "payload": {
        "kind": "runtime-pack",
        "schemaVersion": 1,
        "packId": "oryn.mflux-mlx",
        "version": "0.1.0",
        "publisher": "oryn.first-party",
        "abi": "1",
        "runtime": {
            "kind": "python",
            "entrypoint": "oryn_engine.runtime_packs.mflux_image_worker",
        },
        "dependencyClosure": "mflux0.18.1-adapter-py312",
        "runners": [
            {
                "runnerId": "mflux.image",
                "taskContracts": ["text-to-image-v1"],
                "backends": ["mlx"],
            }
        ],
        "policy": {
            "network": "deny",
            "filesystemRoots": ["model", "artifact-output", "scratch"],
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-sprint-36-development-2026",
        "digest": "f61e6555fd14d95a825aed6e3a6b473c2e7c37072b0b3610c97705e289ea2c16",
        "signature": "QiUFM3S6+MDI/u++TQmi8PcCF7HtltLKRB2RNOCmcxSt6hC08+pog/zGEWGdMBGPJj42sgiNvykYpfQUF21clEVHDvzBKPAMmoB13I090oNK7tNGr4JprimdHpbusw5uJyk3pxOzyR3MLkktA07C7awL6B7Br1N03n2FbKO9NPMjmjI7hJuhXssb0WgCZ0IYvR8f52ruRwxInKsBP39S+ObuyY5ylsPgsmSZjGT4hHrRfjP7D3d/mDR6DpmL4NSlyR2el+wz5Ql5D2/YmzPwGN8RC4GSPH9oKxjWoUSDeEnDdIzYKOvinHuM/t503N7L04qke3KAy2eaUI8LBssykg==",  # noqa: E501
    },
}

MFLUX_KREA2_Q4_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.krea-2-turbo-mflux-q4",
        "runtime": {
            "packId": "oryn.mflux-mlx",
            "packVersion": "0.1.0",
            "runnerId": "mflux.image",
        },
        "model": {
            "provider": "huggingface",
            "repository": "mflux-community/krea-2-turbo-mflux-q4",
            "revision": "f30071b91ba672767e49e83276fe44cc29bc6a8e",
        },
        "match": {"architectures": [], "tasks": ["text-to-image"]},
        "taskContract": "text-to-image-v1",
        "level": "converted-artifact",
        "requiredFiles": [
            "*.safetensors",
            "model.safetensors.index.json",
            "text_encoder/*.safetensors",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/*.json",
            "vae/*.safetensors",
            "vae/model.safetensors.index.json",
        ],
        "capabilities": {
            "task": "text-to-image",
            "architectures": ["Krea2Pipeline"],
            "inputModalities": ["text"],
            "outputModalities": ["image"],
            "capabilities": {"deterministicSeed": True},
        },
        "resources": {
            "estimatedResidentBytes": 19_327_352_832,
            "workingMemoryBytes": 6_442_450_944,
        },
        "metadata": {
            "license": "Krea-2-Community-License",
            "quantization": "Q4",
            "adapterArchitecture": "krea2-mflux-v1",
            "inferenceDefaults": {
                "width": 512,
                "height": 512,
                "steps": 8,
                "guidanceScale": 1.0,
            },
            "source": {
                "provider": "huggingface",
                "repository": "krea/Krea-2-Turbo",
                "revision": "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",
            },
            "conformance": {
                "status": "passed",
                "platforms": ["macos-arm64-mlx"],
            },
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-sprint-36-development-2026",
        "digest": "acd725139508bc08454725ecef36e170823c359df490c3acbdb4fb88708acda0",
        "signature": "dB8UuZpHaUJR7UvUxoPC7WRQAp9zqNMmdMxMhzjOqBn5AN6FXE2r98fLXbDTaBKiEiIbtYVcAJHK68HDn6tQ6WlnehQ99q8xtfS8DYGC2+4pJRsUp8ceGhdgaavqIrML/wglrNJTp5o/LxB0KRDX8fOoFqnFFMYV79G66fDlLOjNJbOH87wOms3ay3INoa96n60hWmoeW+UPRa6N4rGt1fp12cajHwRFMdkl6Bh5YI+XJUKbYEBKwBLDXTuKNb1GsGn6kIqiu5i7pKc6xvtNvqnE7YHAqK+3/QUxjcZNIcXYUA2tWllOgLrXQAmi+oWHPaxMMviYp810L4GgSKx0FA==",  # noqa: E501
    },
}

SPRINT_30_COMPATIBILITY: tuple[dict[str, Any], ...] = (
    {
        "payload": {
            "kind": "model-compatibility",
            "schemaVersion": 1,
            "manifestId": "oryn.compat.qwen3-embedding-0.6b",
            "runtime": {
                "packId": "oryn.transformers-pytorch",
                "packVersion": "0.1.0",
                "runnerId": "transformers.embeddings",
            },
            "model": {
                "provider": "huggingface",
                "repository": "Qwen/Qwen3-Embedding-0.6B",
                "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            },
            "match": {"architectures": ["Qwen3ForCausalLM"], "tasks": ["feature-extraction"]},
            "taskContract": "embeddings-v1",
            "level": "native-declarative",
            "requiredFiles": [
                "config.json",
                "model.safetensors",
                "tokenizer*",
                "merges.txt",
                "vocab.json",
            ],
            "capabilities": {
                "task": "feature-extraction",
                "architectures": ["Qwen3ForCausalLM"],
                "contextLength": 32768,
                "inputModalities": ["text"],
                "outputModalities": ["embedding"],
                "capabilities": {"structuredOutput": True},
            },
            "resources": {"estimatedResidentBytes": 2400000000, "workingMemoryBytes": 800000000},
            "metadata": {
                "license": "Apache-2.0",
                "languages": ["multilingual-100-plus"],
                "conformance": {"status": "passed", "platforms": ["macos-arm64-mps", "cpu"]},
            },
        },
        "integrity": {
            "algorithm": "rsa-sha256",
            "keyId": "oryn-sprint-30-development-2026",
            "digest": "330819b3a455924bb027f8559d57ebc26c2b874f63d7c5530a60e37a3a0689c9",
            "signature": "Z+jNeweKpbnJoj1VHohZ40KqVtHE+I16w+3Td5oS+Gzj/ypC5b3mXvIzq8TCVcOQbfa3JlBuBhJrJkf+cC0S6aDHXre6P4dU4zv8M38tjskte1SZpiIuM3/jfUsmwCeWdz0J5InB3ewbe/V4c2aCoBQllcMLd5sEDiPbJ2fm0NWQ25kQYok3GHQC40n7SFjxzXBJX22iisNHZ5ftEHM6PChVLOJvS4f4g+Iqxcm7RN5XL6OsXjAQongIilHwYJkwAGOTHe4/swFQhkxYHNDmamFDwJY1xd2jt7jaygmTGIu/bEyrx/lGREe5r3xEOeNMCEgCA3LPVOLSkzajDRxAGw==",  # noqa: E501
        },
    },
    {
        "payload": {
            "kind": "model-compatibility",
            "schemaVersion": 1,
            "manifestId": "oryn.compat.qwen3-reranker-0.6b",
            "runtime": {
                "packId": "oryn.transformers-pytorch",
                "packVersion": "0.1.0",
                "runnerId": "transformers.reranking",
            },
            "model": {
                "provider": "huggingface",
                "repository": "Qwen/Qwen3-Reranker-0.6B",
                "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
            },
            "match": {"architectures": ["Qwen3ForCausalLM"], "tasks": ["text-ranking"]},
            "taskContract": "reranking-v1",
            "level": "native-declarative",
            "requiredFiles": [
                "config.json",
                "model.safetensors",
                "tokenizer*",
                "merges.txt",
                "vocab.json",
                "chat_template.jinja",
            ],
            "capabilities": {
                "task": "text-ranking",
                "architectures": ["Qwen3ForCausalLM"],
                "contextLength": 32768,
                "inputModalities": ["text"],
                "outputModalities": ["structured"],
                "capabilities": {"structuredOutput": True},
            },
            "resources": {"estimatedResidentBytes": 2400000000, "workingMemoryBytes": 800000000},
            "metadata": {
                "license": "Apache-2.0",
                "languages": ["multilingual-100-plus"],
                "conformance": {"status": "passed", "platforms": ["macos-arm64-mps", "cpu"]},
            },
        },
        "integrity": {
            "algorithm": "rsa-sha256",
            "keyId": "oryn-sprint-30-development-2026",
            "digest": "770466b3d3c2e7a114a0a620b757bb5ec871bedd857f735057676017b730937a",
            "signature": "SaU1HYwBrv4HxrIFQTdBO42s7NVVZd2anI9hbJuo/NH4KU4Kq/KJIDpDmOqEyrFB6pwccs4E80KbsDaIvkQB3JOK0ny4fvxpFVSaP1k+REf4gpcCuWtHf0OsyvvoHS2TKvsLEpXYyQdeLeQD0n218ee4AumD+zCcYpNq1WlrnUELboxjCUsuIifWE1KyAhitiV1TZ9LBMaN2H9A4bKpdBhf5s/XQslnL3Qkl0vlsdfLQRTALrVCBa6MMlgmPUUKxEbho6CS15DpO2247QIGWoMbT6gHoghoDBCrVQq1Cob8p+wsk+2atqk0rj4SMwDW5hQANFogUuUlX+usii/BGZQ==",  # noqa: E501
        },
    },
)

KOKORO_ONNX_PACK: dict[str, Any] = {
    "payload": {
        "kind": "runtime-pack",
        "schemaVersion": 1,
        "packId": "oryn.kokoro-onnx",
        "version": "0.1.0",
        "publisher": "oryn.first-party",
        "abi": "1",
        "runtime": {
            "kind": "python",
            "entrypoint": "oryn_engine.runtime_packs.kokoro_onnx_worker",
        },
        "dependencyClosure": "kokoro-onnx-ort1.22-espeak3.3.2-py312",
        "runners": [
            {
                "runnerId": "kokoro.speech-synthesis",
                "taskContracts": ["speech-synthesis-v1"],
                "backends": ["coreml", "cpu"],
            }
        ],
        "policy": {
            "network": "deny",
            "filesystemRoots": ["model", "artifact-output", "scratch"],
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-sprint-30-development-2026",
        "digest": "febcffdc1878aba13b2a0e457c9820a64386046e85845b10ddfd2b99d25221f6",
        "signature": "TkoLBP5z3gM7O5ya/LarvV94Qp+DkWLtC15yhn795s9Ev9UOsauvc72N+2PStxOinoM0EYFhEvyzvx3OOHYESoJD+ZzDqgVb3APMTjXAy3NAlL7lRM+fywGilSYvfYTz18dvSDWEDTNHdXkC1lx97cklWIqx6S2NfLJEF8KH1gxW084NNBHpeAEBYArxH1cr8uNP7Pzx9XI7PhClhg536DxJaLvZOq1+ZjBK/Xt1BfN1zxu9M6t1psXG9ega6Io/cvMAtk43u//+ktK4kzy8LMRAdNeeZbD0r89z66/Ict95aHzYGH8SvRZOFq7fno7sSxTHpatmIpVbwFhsfzrrtA==",  # noqa: E501
    },
}

KOKORO_ONNX_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.kokoro-82m-v1-onnx-q8",
        "runtime": {
            "packId": "oryn.kokoro-onnx",
            "packVersion": "0.1.0",
            "runnerId": "kokoro.speech-synthesis",
        },
        "model": {
            "provider": "huggingface",
            "repository": "onnx-community/Kokoro-82M-v1.0-ONNX",
            "revision": "1939ad2a8e416c0acfeecc08a694d14ef25f2231",
        },
        "match": {"architectures": [], "tasks": ["text-to-speech"]},
        "taskContract": "speech-synthesis-v1",
        "level": "converted-artifact",
        "requiredFiles": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "onnx/model_quantized.onnx",
            "voices/af_heart.bin",
            "voices/af_bella.bin",
            "voices/af_nicole.bin",
            "voices/af_sarah.bin",
            "voices/am_adam.bin",
            "voices/am_michael.bin",
            "voices/bf_emma.bin",
            "voices/bf_isabella.bin",
            "voices/bm_george.bin",
            "voices/bm_lewis.bin",
        ],
        "capabilities": {
            "task": "text-to-speech",
            "architectures": ["Kokoro82MOnnx"],
            "inputModalities": ["text"],
            "outputModalities": ["audio"],
            "capabilities": {"speechSynthesis": True, "audioOutput": True},
        },
        "resources": {"estimatedResidentBytes": 256000000, "workingMemoryBytes": 256000000},
        "metadata": {
            "license": "Apache-2.0",
            "source": {
                "provider": "huggingface",
                "repository": "hexgrad/Kokoro-82M",
                "revision": "f3ff3571791e39611d31c381e3a41a3af07b4987",
            },
            "languages": ["en-us", "en-gb"],
            "voices": [
                {"id": "af_heart", "language": "en-us", "name": "Heart"},
                {"id": "af_bella", "language": "en-us", "name": "Bella"},
                {"id": "af_nicole", "language": "en-us", "name": "Nicole"},
                {"id": "af_sarah", "language": "en-us", "name": "Sarah"},
                {"id": "am_adam", "language": "en-us", "name": "Adam"},
                {"id": "am_michael", "language": "en-us", "name": "Michael"},
                {"id": "bf_emma", "language": "en-gb", "name": "Emma"},
                {"id": "bf_isabella", "language": "en-gb", "name": "Isabella"},
                {"id": "bm_george", "language": "en-gb", "name": "George"},
                {"id": "bm_lewis", "language": "en-gb", "name": "Lewis"},
            ],
            "conformance": {"status": "passed", "platforms": ["macos-arm64-coreml", "cpu"]},
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-sprint-30-development-2026",
        "digest": "132e01d9a7fc38d887222f1b60735f19c6a656b6e01ba20341288cce08bfb6cc",
        "signature": "wAPROqZhjzCOH+kCAmZh49ZSN8jOjZPzwzFI0E1vqbqVAUGth+MbZ4YQUYX038ToTtj137+uJV4I5bnuPgAZeWxZXBvUCTz8wW5pF19hDm6TxLv7w6y2WODPFyCMks33/lYwL+8m1VK0DG4TT7gE6gmw+GS/O4dECtOWw2UQax8gNTY8YjTNHB7QRYuhYilYoYwSBS4xLzFzFATRGl7LkUeq11bNQi4tBuonRsw3agVnXQUCsX3FIMl+8BXrfp3Y8p2TpzcMoNNMsONFMQGvzWLkgjakj5TFrQInZFf04ePV3AwfBEh0odd5NTm0NOmdnuv3Onp8DNH6IFi/+ixolQ==",  # noqa: E501
    },
}

DIFFUSERS_PACK: dict[str, Any] = {
    "payload": {
        "kind": "runtime-pack",
        "schemaVersion": 1,
        "packId": "oryn.diffusers-pytorch",
        "version": "0.2.0",
        "publisher": "oryn.first-party",
        "abi": "1",
        "runtime": {
            "kind": "python",
            "entrypoint": "oryn_engine.runtime_packs.diffusers_worker",
        },
        "dependencyClosure": ("diffusers0.39-bitsandbytes0.50.1-imageio2.37-base-torch2.13-py312"),
        "runners": [
            {
                "runnerId": "diffusers.pipeline",
                "taskContracts": [
                    "text-to-image-v1",
                    "image-to-image-v1",
                    "text-to-video-v1",
                    "image-to-video-v1",
                ],
                "backends": ["mps", "cuda", "cpu"],
            }
        ],
        "policy": {
            "network": "deny",
            "filesystemRoots": ["model", "artifact-input", "artifact-output", "scratch"],
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-bitsandbytes-development-2026",
        "digest": "90ae4e70406b39b15d53613429ae89faa2357cf302506c8dcc2aed9165c1393a",
        "signature": "buG35iIFqUdCNhsatflnulNnn34ivAa4csKBZzM9KUQ69okg0skMwv/vY694N7v+bcGALsXDvA82jAeJAUZiPsipCkKgVYxSStNZ9lYZtP35UnraeFagqRXNkRdhJeUr1R5L0LTnYBR5cgF7LwpNygvBtyTDIjL70VlWfJfHcWC/fPETr2x9Rn0JxkZcbXeXRywSL/F+ZKd3032BDeLspuy5gmdh2jRI4/b5vSp1u3R3NDSR3JUC3uWy1Ywm+hyDDNLAiCTNj5TD0Ds0nUfZz+Vu22NlBZ0NBRkzIGwbMVheKK82SkV+668OObV5A9S+CR8t89XTLIw/PCx2eYkVHw==",  # noqa: E501
    },
}

DIFFUSERS_IMAGE_FIXTURE_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.tiny-random-sdxl-safetensors",
        "runtime": {
            "packId": "oryn.diffusers-pytorch",
            "packVersion": "0.2.0",
            "runnerId": "diffusers.pipeline",
        },
        "model": {
            "provider": "huggingface",
            "repository": "dg845/tiny-random-stable-diffusion-xl",
            "revision": "7ae769dd7f63b06308f2f23881d329a0cb1c89d2",
        },
        "match": {"architectures": [], "tasks": ["text-to-image"]},
        "taskContract": "text-to-image-v1",
        "level": "native-declarative",
        "requiredFiles": [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.safetensors",
            "tokenizer/*",
            "tokenizer_2/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
        "capabilities": {
            "task": "text-to-image",
            "architectures": ["StableDiffusionXLPipeline"],
            "inputModalities": ["text"],
            "outputModalities": ["image"],
            "capabilities": {"deterministicSeed": True},
        },
        "resources": {"estimatedResidentBytes": 268435456, "workingMemoryBytes": 536870912},
        "metadata": {
            "license": "Apache-2.0",
            "conformance": {"status": "fixture-only", "platforms": ["macos-arm64-mps", "cpu"]},
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-bitsandbytes-development-2026",
        "digest": "9a422e0ae90ac0f54568d2c369c67cc860eb1044891c27f4ef7eeed5c9b6d1dc",
        "signature": "eY/nOPtCSkLZ1hge8H08ntFKT6FmwhI2lKNAn1zfvgWCKIxNgUWaTMQbjLD1hKC+bIdNpe8FFbUT5ESTzt1jusxXPyBH4s93I7OX0ah0Sj5ubQUnBt9LE/aS69Ff8SwDh4XFueZ1YQL28f1AQpB6CLjJcxvi3UZ5C2ui1tzGAUF+NfpWNhR7r4WhnGNzlbNqSRd0pOT2NrG56IQAYtNKR/iQRqs0RyfxO4AUpN94toy7l4ptUopvUEV2uAE+D+RFeWAVDDs1aAQk+kTEZsVAW8VJUQpgtqoVAB84RBmuFCLSOYAPDFkJkgRJSzkudmDVhLFR3iPOCG4/i4SIHHcRsQ==",  # noqa: E501
    },
}

GENERIC_DIFFUSERS_IMAGE_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.generic-safe-diffusers-image",
        "runtime": {
            "packId": "oryn.diffusers-pytorch",
            "packVersion": "0.2.0",
            "runnerId": "diffusers.pipeline",
        },
        "model": None,
        "match": {"architectures": [], "tasks": ["text-to-image"]},
        "taskContract": "text-to-image-v1",
        "level": "native-declarative",
        "requiredFiles": [
            "model_index.json",
            "*/*.json",
            "*.safetensors",
            "tokenizer*/*",
        ],
        "capabilities": {
            "task": "text-to-image",
            "architectures": [],
            "inputModalities": ["text"],
            "outputModalities": ["image"],
            "capabilities": {"deterministicSeed": True},
        },
        "resources": {"estimatedResidentBytes": 0, "workingMemoryBytes": 0},
        "metadata": {
            "license": "Repository-defined",
            "conformance": {
                "status": "fixture-only",
                "platforms": ["macos-arm64-mps", "cuda", "cpu"],
            },
            "generationProfiles": [
                {
                    "match": {"_class_name": "Krea2Pipeline", "is_distilled": True},
                    "profile": {
                        "profileId": "diffusers.krea2.distilled",
                        "recommended": True,
                        "parameters": {
                            "width": {
                                "default": 512,
                                "minimum": 64,
                                "maximum": 2048,
                                "step": 8,
                                "supported": True,
                                "editable": True,
                            },
                            "height": {
                                "default": 512,
                                "minimum": 64,
                                "maximum": 2048,
                                "step": 8,
                                "supported": True,
                                "editable": True,
                            },
                            "steps": {
                                "default": 8,
                                "minimum": 1,
                                "maximum": 100,
                                "step": 1,
                                "supported": True,
                                "editable": True,
                            },
                            "guidanceScale": {
                                "default": 0.0,
                                "minimum": 0.0,
                                "maximum": 30.0,
                                "step": 0.1,
                                "supported": True,
                                "editable": True,
                            },
                        },
                    },
                },
                {
                    "match": {"_class_name": "Krea2Pipeline", "is_distilled": False},
                    "profile": {
                        "profileId": "diffusers.krea2.base",
                        "recommended": True,
                        "parameters": {
                            "width": {
                                "default": 512,
                                "minimum": 64,
                                "maximum": 2048,
                                "step": 8,
                                "supported": True,
                                "editable": True,
                            },
                            "height": {
                                "default": 512,
                                "minimum": 64,
                                "maximum": 2048,
                                "step": 8,
                                "supported": True,
                                "editable": True,
                            },
                            "steps": {
                                "default": 28,
                                "minimum": 1,
                                "maximum": 100,
                                "step": 1,
                                "supported": True,
                                "editable": True,
                            },
                            "guidanceScale": {
                                "default": 4.5,
                                "minimum": 0.0,
                                "maximum": 30.0,
                                "step": 0.1,
                                "supported": True,
                                "editable": True,
                            },
                        },
                    },
                },
            ],
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-generation-profiles-development-2026",
        "digest": "9da0ff8a46757c8635adc723f4712c05b3243b16ab360e407a9cf895bc58e269",
        "signature": "iomugJR1jWMaG19OT2uW2LiJSyWFkvFnIZsop391tHZ9S5TZtmiRpYh4/JL3ko+7O0kxTBJ1Of4JeKR9S49Pgi2JEZZ3YhjBItLqnpTv+vhENolvzL9s37nadooIe3Pd2DeoWEaZXUg2kdaTZwBXGrkKbit1EON75gkE/XjioPzpco8C09F9tOa7qh6mVN/7jGsr+SCpZMPr447QtCYaoNWfFq0zneNGsASJLCzWkHjELJ3VInNqG+XPjVreYBMjAumvMGmlrmf4w4pHvKO4+RHQqkJNtShoKRxXCVPDeSl6mWMFRwJFW9gdT6ow0Bex5bpRDO8RG5D4LTOdoStMFA==",  # noqa: E501
    },
}

DIFFUSERS_VIDEO_FIXTURE_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.tiny-random-wan2.1-t2v",
        "runtime": {
            "packId": "oryn.diffusers-pytorch",
            "packVersion": "0.2.0",
            "runnerId": "diffusers.pipeline",
        },
        "model": {
            "provider": "huggingface",
            "repository": "Erland/tiny-wan2.1-t2v-debug",
            "revision": "f7f0c2439a5728ad0ef1a009053128ee6eeb1889",
        },
        "match": {"architectures": [], "tasks": ["text-to-video"]},
        "taskContract": "text-to-video-v1",
        "level": "native-declarative",
        "requiredFiles": [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/*",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ],
        "capabilities": {
            "task": "text-to-video",
            "architectures": ["WanPipeline"],
            "inputModalities": ["text"],
            "outputModalities": ["video"],
            "capabilities": {"deterministicSeed": True},
        },
        "resources": {"estimatedResidentBytes": 1073741824, "workingMemoryBytes": 2147483648},
        "metadata": {
            "license": "Apache-2.0",
            "conformance": {"status": "fixture-only", "platforms": ["macos-arm64-mps", "cpu"]},
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-bitsandbytes-development-2026",
        "digest": "3c0538c74af0a877501f2d2d2f70b91c97253006809b9ba76b9d2ae1c8badb02",
        "signature": "oEMw4KlDilVEg9+SvENiWy5Uv4dQVTJwkZwT5w8vjeZ/7ezX+eH0IpslKx9e4p75etb+rhcd8QnUytvsdC9QP1yIDr5HeneBOmXpzcqViNPas5OiSxuMfehf+4RlRsLEEXvLnsKk6APt/PVGeZO+GI5Odk4n8LnDwVFmCpIv2PT6yo3zVr14p0A/xqt31bdoEBm6DbhYuJshoASGf58XB9vj09fT+0RfCX4DgViKMB/Eu9LQxEDiqm2iKDiUx8MG++J6WFK1oMT+vZUfxPJxso45WiqKrQ/molHZcPyBTD6CJz7prOGR9lX7PoNORvJ0XxkglR2VLZSye+KJ4vftkw==",  # noqa: E501
    },
}

DIFFUSERS_VEGA_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.segmind-vega-fp16",
        "runtime": {
            "packId": "oryn.diffusers-pytorch",
            "packVersion": "0.2.0",
            "runnerId": "diffusers.pipeline",
        },
        "model": {
            "provider": "huggingface",
            "repository": "segmind/Segmind-Vega",
            "revision": "7714c4363e5856ff974a4f4b068e8691f26d0b40",
        },
        "match": {"architectures": [], "tasks": ["text-to-image"]},
        "taskContract": "text-to-image-v1",
        "level": "native-declarative",
        "requiredFiles": [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.fp16.safetensors",
            "tokenizer/*",
            "tokenizer_2/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
        "capabilities": {
            "task": "text-to-image",
            "architectures": ["StableDiffusionXLPipeline"],
            "inputModalities": ["text"],
            "outputModalities": ["image"],
            "capabilities": {"deterministicSeed": True},
        },
        "resources": {"estimatedResidentBytes": 7516192768, "workingMemoryBytes": 4294967296},
        "metadata": {
            "license": "Apache-2.0",
            "conformance": {"status": "passed", "platforms": ["macos-arm64-mps"]},
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-bitsandbytes-development-2026",
        "digest": "3feab9c414ed8a6696ada2f5a592b1f3b14e1a8a81fdc7b43a142f312b43f5ec",
        "signature": "nDllQijfMP0AUWwBEV7nMi7dE02AmeTI3+t1ZNc5Ym6ewsT4ehArfW5rG8J5gLPt+dSwoz0ZNKYb5scBVu9Q87U37tIvDhkHZ5jpBzAlcibSW10uQ+nmpTtw4P5yvFBeDKFnuBOaOhqEIDupG70ipvWG4l9+UX5+ph+yRZAhGc/WNJ1WUixwn67sR3OjUrk/jLRPf+Ikjc+LV62e6xSbIBBCESMgUw2hVWV/+v0Pk9AUj/mGTTFwcg6RkOAVkBAuv3t8WemK/f1h0jXGVDCk2fF1WwBOYrs+VOBqA1j5im+s8I6ofHEVwFMZdLr8N+UF8HUXID5kwCuAPxOtNufo2w==",  # noqa: E501
    },
}

LLAMA_CPP_PACK: dict[str, Any] = {
    "payload": {
        "kind": "runtime-pack",
        "schemaVersion": 1,
        "packId": "oryn.llama-cpp",
        "version": "0.1.0",
        "publisher": "oryn.first-party",
        "abi": "1",
        "runtime": {
            "kind": "python",
            "entrypoint": "oryn_engine.runtime_packs.llama_cpp_worker",
        },
        "dependencyClosure": "llama.cpp-b10218-de699957-abi1",
        "runners": [
            {
                "runnerId": "llama.cpp.chat",
                "taskContracts": ["chat-v1"],
                "backends": ["metal", "cuda", "cpu"],
            }
        ],
        "policy": {"network": "deny", "filesystemRoots": ["model", "scratch"]},
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-sprint-32-development-2026",
        "digest": "af93dac50c7698b390b31047c7cd75a3c484940755b54585593c1c7f06bbfd0e",
        "signature": "lBZhdqAcyRAEC7ufe4tgzNSG7Hs/5O2sG+g6pjMy9cbV7Wm8Qo64Mzt7TcSAjvPjn/NuR9lBUzCa03/17w3afXLCN0ujSUyqpspsiOsE7RCp/S1udvp2tBeYLwy5x9Cdt8zLXWLTLlqmI3eSu+9vUFBKuWkz1oeEZepLmNhhsPdqv1WGwH6sCBJB4rrKmY+lhQqF5wOlS1vvnFgOodxCOMhVAg7HPfKqrL9FKPvDTYfbN3GSQI3Lyfv5GTnM0mw3QOJtITWr6k+YXUnnC7DXZ9bur7m8qgY5I+cyyR4jXk5bvTS8Rn8V7wr+eX9kZd+rBfjo7XR55nD+uQqcsNx4oQ==",  # noqa: E501
    },
}

LLAMA_CPP_FIXTURE_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.tiny-gemma3-gguf-q8-0",
        "runtime": {
            "packId": "oryn.llama-cpp",
            "packVersion": "0.1.0",
            "runnerId": "llama.cpp.chat",
        },
        "model": {
            "provider": "huggingface",
            "repository": "ggml-org/tinygemma3-GGUF",
            "revision": "c287502cd9e278dac8eed805c112cce5d0081e0b",
        },
        "match": {"architectures": [], "tasks": ["text-generation"]},
        "taskContract": "chat-v1",
        "level": "converted-artifact",
        "requiredFiles": ["tinygemma3-Q8_0.gguf"],
        "capabilities": {
            "task": "text-generation",
            "architectures": ["gemma3"],
            "contextLength": 131072,
            "inputModalities": ["text"],
            "outputModalities": ["text"],
            "capabilities": {"chat": True, "streaming": True},
        },
        "resources": {"estimatedResidentBytes": 134217728, "workingMemoryBytes": 268435456},
        "metadata": {
            "license": "WTFPL",
            "quantization": "Q8_0",
            "contextLength": 131072,
            "chatTemplate": "gemma3",
            "modelFile": "tinygemma3-Q8_0.gguf",
            "conformance": {"status": "fixture-only", "platforms": ["macos-arm64-metal", "cpu"]},
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-sprint-32-development-2026",
        "digest": "81f38538a01a5b06bbacef8edd3d43e4e4054f5ddf6b040574906d768e796cd1",
        "signature": "mWYb4Cy1dx5LRix8MSeUGiNc2MoK0c8GcPNg5IHF+owigsbSxTSFbYaPZqaPmMHKQ+0OrOf3Obt0xGaU7YjWdGbv/C9lJGtsr6JCehN1PdEOR0cnImD5kFagKdjPdRs3MJvC9SDSNMiaoao0zp5MLhDZUvn0V6RHpO+GTX6oON2iCxRvi8cY8FVO6C66cfknojR8yqUCV+enPcanp2Gd40w2WLCZfe6ZhpwP6pZtvhGBQrga/AU3JAr/MmGaMV3JEZ5nrmLiuxb2IC1AZuCawDJk89braslNotBoHjxt+RggfPPhYJLo1cnHMW8YP1/9nQvrSxUaDLlP3PQAj8i+2g==",  # noqa: E501
    },
}

LLAMA_CPP_QWEN_COMPATIBILITY: dict[str, Any] = {
    "payload": {
        "kind": "model-compatibility",
        "schemaVersion": 1,
        "manifestId": "oryn.compat.qwen2.5-0.5b-instruct-gguf-q4-k-m",
        "runtime": {
            "packId": "oryn.llama-cpp",
            "packVersion": "0.1.0",
            "runnerId": "llama.cpp.chat",
        },
        "model": {
            "provider": "huggingface",
            "repository": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "revision": "9217f5db79a29953eb74d5343926648285ec7e67",
        },
        "match": {"architectures": [], "tasks": ["text-generation"]},
        "taskContract": "chat-v1",
        "level": "converted-artifact",
        "requiredFiles": ["qwen2.5-0.5b-instruct-q4_k_m.gguf"],
        "capabilities": {
            "task": "text-generation",
            "architectures": ["qwen2"],
            "contextLength": 32768,
            "inputModalities": ["text"],
            "outputModalities": ["text"],
            "capabilities": {"chat": True, "streaming": True},
        },
        "resources": {"estimatedResidentBytes": 805306368, "workingMemoryBytes": 1073741824},
        "metadata": {
            "license": "Apache-2.0",
            "quantization": "Q4_K_M",
            "contextLength": 32768,
            "chatTemplate": "chatml",
            "modelFile": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "conformance": {"status": "passed", "platforms": ["macos-arm64-metal"]},
        },
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-sprint-32-development-2026",
        "digest": "a4ed8c7bb439ac0d3fd38e162b8d33dc3dc044b6f5a1c881679e6060776d25b4",
        "signature": "RZ/8TD+Na1nrP4YCfH0bCoO96g8n5zvc7qLBhCMSOVkJRJMIDkzTUMqmJFcfi73DjkC0DJPwu9rPImlpKkubXRi7RoRMUCHd0nx5f+32ZvchF+So6YCh4FROIr/+vSNXC1Bj1gqca055S3pIx35CXVFP1f07TgGlJKr0QqIslvI3OgcH/9KS/q3B/q5bSWWQxB7smpeV3847/2QB+J5SvE/MrsnPCZSLJBkl88f/9RNHJVD6w7U4Tw17Rwhf6XrV54GAqnFzTCylXSuM2vk32ceSTofZ/sVpPLnx4nRVNncwiNXQ4VHSZSyQqIACSgmhX0kp3dF6qpjPv8i/IGDRDw==",  # noqa: E501
    },
}


def builtin_registry() -> RuntimePackRegistry:
    registry = RuntimePackRegistry(
        SignedManifestVerifier(
            {
                DEVELOPMENT_KEY_ID: TrustedPublisherKey(DEVELOPMENT_KEY_ID, DEVELOPMENT_MODULUS),
                DEVELOPMENT_PUBLISHER.key_id: DEVELOPMENT_PUBLISHER,
                ACE_STEP_DEVELOPMENT_PUBLISHER.key_id: ACE_STEP_DEVELOPMENT_PUBLISHER,
                SPRINT_30_DEVELOPMENT_PUBLISHER.key_id: SPRINT_30_DEVELOPMENT_PUBLISHER,
                SPRINT_31_DEVELOPMENT_PUBLISHER.key_id: SPRINT_31_DEVELOPMENT_PUBLISHER,
                SPRINT_32_DEVELOPMENT_PUBLISHER.key_id: SPRINT_32_DEVELOPMENT_PUBLISHER,
                GENERIC_DIFFUSERS_DEVELOPMENT_PUBLISHER.key_id: (
                    GENERIC_DIFFUSERS_DEVELOPMENT_PUBLISHER
                ),
                SPRINT_36_DEVELOPMENT_PUBLISHER.key_id: SPRINT_36_DEVELOPMENT_PUBLISHER,
                BITSANDBYTES_DEVELOPMENT_PUBLISHER.key_id: (BITSANDBYTES_DEVELOPMENT_PUBLISHER),
                GENERATION_PROFILES_DEVELOPMENT_PUBLISHER.key_id: (
                    GENERATION_PROFILES_DEVELOPMENT_PUBLISHER
                ),
            }
        )
    )
    registry.register_pack(
        json.loads((Path(__file__).parents[1] / "data/mlx-pack.json").read_text())
    )
    data_root = Path(__file__).parents[1] / "data"
    registry.register_pack(json.loads((data_root / "transformers-pack.json").read_text()))
    registry.register_pack(json.loads((data_root / "onnx-pack.json").read_text()))
    registry.register_pack(json.loads((data_root / "qwen-tts-pack.json").read_text()))
    registry.register_pack(json.loads((data_root / "mlx-audio-pack.json").read_text()))
    for compatibility in json.loads((data_root / "transformers-compatibility.json").read_text()):
        registry.register_compatibility(
            compatibility,
            publish_to_catalog=(
                compatibility["payload"]["manifestId"] != "oryn.compat.tiny-musicgen"
            ),
        )
    registry.register_pack(ACE_STEP_PACK)
    registry.register_compatibility(ACE_STEP_COMPATIBILITY)
    registry.register_pack(json.loads((data_root / "mflux-pack.json").read_text()))
    for compatibility in json.loads((data_root / "mflux-compatibility.json").read_text()):
        registry.register_compatibility(compatibility)
    registry.register_pack(json.loads((data_root / "kokoro-pack.json").read_text()))
    for compatibility in json.loads((data_root / "kokoro-compatibility.json").read_text()):
        registry.register_compatibility(compatibility)
    for compatibility in SPRINT_30_COMPATIBILITY:
        if compatibility["payload"]["runtime"]["packId"] != "oryn.transformers-pytorch":
            registry.register_compatibility(compatibility)
    registry.register_pack(json.loads((data_root / "diffusers-pack.json").read_text()))
    for compatibility in json.loads((data_root / "diffusers-compatibility.json").read_text()):
        registry.register_compatibility(
            compatibility,
            publish_to_catalog=compatibility["payload"]["manifestId"]
            == DIFFUSERS_VEGA_COMPATIBILITY["payload"]["manifestId"],
        )
    registry.register_pack(json.loads((data_root / "llama-pack.json").read_text()))
    for compatibility in json.loads((data_root / "llama-compatibility.json").read_text()):
        registry.register_compatibility(
            compatibility,
            publish_to_catalog=(
                compatibility["payload"]["manifestId"]
                != LLAMA_CPP_FIXTURE_COMPATIBILITY["payload"]["manifestId"]
            ),
        )
    return registry
